# keep each GPT-2 XL task inside one disposable process
import argparse
import gc
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_utils import read_json
from scripts.grpo_worker import classify_failure, save_status
from scripts.s12_eval import IntegrityError, atomic_write_json, sha256


def _latest_checkpoint_step(out_dir):
    steps = []
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if name.startswith("checkpoint-"):
                try:
                    steps.append(int(name.rsplit("-", 1)[1]))
                except ValueError:
                    pass
    return max(steps, default=0)


def _cuda_memory_point(label):
    import torch

    if not torch.cuda.is_available():
        return {"label": label, "available": False}
    torch.cuda.synchronize()
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    return {
        "label": label,
        "available": True,
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "free_gib": free / 1024**3,
        "total_gib": total / 1024**3,
    }


def _instrument_resume_memory(trainer, target):
    import torch

    # checkpoint restoration is measured separately from ordinary training
    for method_name in ("_load_from_checkpoint", "_load_optimizer_and_scheduler"):
        original = getattr(trainer, method_name, None)
        if not callable(original):
            raise IntegrityError(f"experiment 046 trainer has no {method_name}")

        def measured(*args, _original=original, _name=method_name, **kwargs):
            target.append(_cuda_memory_point(f"before {_name}"))
            value = _original(*args, **kwargs)
            target.append(_cuda_memory_point(f"after {_name}"))
            gc.collect()
            torch.cuda.empty_cache()
            target.append(_cuda_memory_point(f"after {_name} cleanup"))
            return value

        setattr(trainer, method_name, measured)


def _wandb_callback(result_dir, cfg, stage, target):
    from scripts.s42_worker import _capture_wandb_run
    from transformers import TrainerCallback

    # each resumed online run gets its own record beside the local history
    class wandb_record(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step > 0 and not target:
                target.append(_capture_wandb_run(
                    result_dir, cfg, state.global_step,
                    package="046", stage=stage))
            return control

    return wandb_record()


def _check_lora_setup(trainer, stage, package="046"):
    import torch
    from train.model_utils import SCRATCH_LORA_TARGET_MODULES

    model = trainer.accelerator.unwrap_model(trainer.model)
    # count every target block before accepting the smaller trainable surface
    trainable = [(name, value) for name, value in model.named_parameters()
                 if value.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise IntegrityError(f"experiment {package} found a trainable non-LoRA parameter")
    if any(value.dtype != torch.float32 for _, value in trainable):
        raise IntegrityError(f"experiment {package} LoRA parameters are not float32")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    expected_layers = getattr(base.config, "n_layer", None)
    if not isinstance(expected_layers, int) or expected_layers <= 0:
        raise IntegrityError(f"experiment {package} model layer count is unavailable")
    target_counts = {}
    for target in SCRATCH_LORA_TARGET_MODULES:
        modules = [name for name, _ in model.named_modules()
                   if name.endswith(target)]
        if len(modules) != expected_layers:
            raise IntegrityError(f"experiment {package} LoRA target count differs: {target}")
        target_counts[target] = len(modules)
    if base.get_output_embeddings().weight is not base.get_input_embeddings().weight:
        raise IntegrityError(f"experiment {package} language model weights are not tied")
    if stage == "sft":
        config = base.config
        if (config.embd_pdrop != 0.1 or config.attn_pdrop != 0.1
                or config.resid_pdrop != 0.1):
            raise IntegrityError(f"experiment {package} SFT dropout config changed")
    # walking the wrapped model covers the reference too, since lora reaches it by
    # disabling the adapter over these same dropout modules
    elif any(module.p != 0.0 for module in model.modules()
             if isinstance(module, torch.nn.Dropout)):
        raise IntegrityError(f"experiment {package} GRPO dropout was not disabled")
    trainable_parameters = sum(value.numel() for _, value in trainable)
    total_parameters = sum(value.numel() for value in model.parameters())
    return {
        "verified": True,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "trainable_tensors": len(trainable),
        "targets": list(SCRATCH_LORA_TARGET_MODULES),
        "target_module_counts": target_counts,
        "tied_language_model_weights": True,
    }


def _reference_behaviour_report(before, after, package="046"):
    # keep the disabled reference fixed while the trained adapter moves the policy
    def maximum_difference(left, right):
        if (len(left) != len(right)
                or any(len(a) != len(b) for a, b in zip(left, right))):
            raise IntegrityError(f"experiment {package} reference rows do not align")
        values = [abs(a - b) for row_a, row_b in zip(left, right)
                  for a, b in zip(row_a, row_b)]
        return max(values, default=0.0)

    reference_drift = maximum_difference(
        before["reference_logprobs"], after["reference_logprobs"])
    adapter_effect = maximum_difference(
        after["policy_logprobs"], after["reference_logprobs"])
    if reference_drift > 1.0e-3:
        raise IntegrityError(f"experiment {package} adapter disabled reference changed")
    if adapter_effect == 0.0:
        raise IntegrityError(f"experiment {package} adapter disable has no behavioural effect")
    return {
        "verified": True,
        "reference_max_abs_drift": reference_drift,
        "adapter_effect_max_abs": adapter_effect,
        "sampled_logprobs": sum(len(row) for row in after["reference_logprobs"]),
    }


def _check_eval_source(branch, adapter_dir, model_path, package="046"):
    if branch == "base":
        valid = model_path is not None and adapter_dir is None
    elif branch in ("sft", "grpo"):
        valid = adapter_dir is not None and model_path is None
    else:
        valid = False
    if not valid:
        raise IntegrityError(f"experiment {package} {branch} evaluation source differs")


def _convert(cfg, scratch_dir, out_dir, record_path, package="046"):
    from checkpoint import write_artifact_type
    from gpt2_scratch.weights import reference_checkpoint
    from scripts.s42_worker import commit_full_artifact
    from train.model_utils import (load_scratch_model_and_tokenizer,
                                   resolve_model_id, write_model_base)

    if os.path.exists(record_path):
        from scripts.s42_worker import verify_artifact_record
        return verify_artifact_record(record_path)
    if os.path.exists(scratch_dir):
        shutil.rmtree(scratch_dir)
    os.makedirs(scratch_dir)
    source = resolve_model_id(cfg)
    model, tokenizer = load_scratch_model_and_tokenizer(
        cfg, source, dtype="float32")
    model.save_pretrained(scratch_dir)
    tokenizer.save_pretrained(scratch_dir)
    write_model_base(scratch_dir, "hf_id", source)
    write_artifact_type(
        scratch_dir, "full", family="scratch_gpt", scale=cfg.scale.name)
    _, revision = reference_checkpoint(source)
    return commit_full_artifact(
        scratch_dir, out_dir, record_path, "immutable_base",
        {"experiment": package, "source": source, "source_revision": revision})


def _reference_outputs(model, input_ids, attention_mask, position_ids):
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids, attention_mask=attention_mask,
            position_ids=position_ids, use_cache=False).logits.cpu()
        labels = input_ids.clone()
        labels[0, :2] = -100
        loss = float(model(
            input_ids, attention_mask=attention_mask,
            position_ids=position_ids, labels=labels,
            use_cache=False).loss.item())
        generation = model.generate(
            input_ids[:1], attention_mask=attention_mask[:1],
            max_new_tokens=32, do_sample=False, pad_token_id=50256,
            return_dict_in_generate=True, output_scores=True)
        prefix = model(
            input_ids[:, :-1], attention_mask=attention_mask[:, :-1],
            position_ids=position_ids[:, :-1], use_cache=True)
        cached = model(
            input_ids[:, -1:], attention_mask=attention_mask,
            position_ids=position_ids[:, -1:],
            past_key_values=prefix.past_key_values, use_cache=True).logits[:, -1].cpu()
    cache_error = float((cached - logits[:, -1]).abs().max().item())
    scores = tuple(value[0].float().cpu() for value in generation.scores)
    return logits, loss, generation.sequences.cpu(), scores, cached, cache_error


def _generation_diagnostic(expected, actual, expected_scores, actual_scores,
                           prefix_length):
    import torch

    left = expected[0, prefix_length:]
    right = actual[0, prefix_length:]
    # record the first changed greedy choice even when one sequence ends early
    common = min(left.numel(), right.numel())
    different = torch.nonzero(left[:common] != right[:common], as_tuple=False)
    if different.numel() == 0 and left.numel() == right.numel():
        return {"first_divergence": None}
    step = int(different[0].item()) if different.numel() else common

    def margin(scores):
        if step >= len(scores):
            return None
        values = torch.topk(scores[step], 2).values
        return float((values[0] - values[1]).item())

    max_error = None
    if step < len(expected_scores) and step < len(actual_scores):
        max_error = float(
            (expected_scores[step] - actual_scores[step]).abs().max().item())
    return {
        "first_divergence": step,
        "reference_token": int(left[step].item()) if step < left.numel() else None,
        "scratch_token": int(right[step].item()) if step < right.numel() else None,
        "reference_top_two_margin": margin(expected_scores),
        "scratch_top_two_margin": margin(actual_scores),
        "max_abs_logit_error": max_error,
    }


def _parity(cfg, base_record_path, record_path, package="046"):
    import torch
    from transformers import GPT2LMHeadModel
    from gpt2_scratch.weights import reference_checkpoint
    from scripts.s42_worker import _write_once
    from train.model_utils import load_scratch_model, resolve_model_id

    base = read_json(base_record_path)
    source, revision = reference_checkpoint(resolve_model_id(cfg))
    input_ids = torch.tensor([
        [50256, 50256, 15496, 11, 314, 716],
        [464, 2068, 7586, 21831, 318, 13779],
    ], device="cuda")
    attention_mask = input_ids.ne(50256).long()
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    # unload the official model before the submitted model reaches the card
    reference = GPT2LMHeadModel.from_pretrained(
        source, revision=revision, dtype=torch.float32).to("cuda")
    (expected_logits, expected_loss, expected_generation, expected_scores,
     expected_cached, reference_cache_error) = _reference_outputs(
        reference, input_ids, attention_mask, position_ids)
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    scratch = load_scratch_model(
        cfg, base["artifact"], dtype="float32", device="cuda")
    (actual_logits, actual_loss, actual_generation, actual_scores,
     actual_cached, scratch_cache_error) = _reference_outputs(
        scratch, input_ids, attention_mask, position_ids)
    valid = attention_mask.bool().cpu().unsqueeze(-1).expand_as(actual_logits)
    padded_logit_error = float(
        (expected_logits[valid] - actual_logits[valid]).abs().max().item())
    unpadded_logit_error = float(
        (expected_logits[1] - actual_logits[1]).abs().max().item())
    cached_logit_error = float(
        (expected_cached - actual_cached).abs().max().item())
    loss_error = abs(expected_loss - actual_loss)
    exact_generation = bool(torch.equal(expected_generation, actual_generation))
    generation_diagnostic = _generation_diagnostic(
        expected_generation, actual_generation, expected_scores, actual_scores,
        input_ids[:1].shape[1])
    tied = bool(
        scratch.get_output_embeddings().weight is scratch.get_input_embeddings().weight
        and scratch.get_output_embeddings().weight.data_ptr()
        == scratch.get_input_embeddings().weight.data_ptr())
    del scratch
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = load_scratch_model(
        cfg, base["artifact"], dtype="float32", device="cuda")
    reloaded.eval()
    with torch.no_grad():
        reloaded_logits = reloaded(
            input_ids, attention_mask=attention_mask,
            position_ids=position_ids, use_cache=False).logits.cpu()
    reload_error = float((actual_logits - reloaded_logits).abs().max().item())
    reload_tied = bool(
        reloaded.get_output_embeddings().weight is reloaded.get_input_embeddings().weight
        and reloaded.get_output_embeddings().weight.data_ptr()
        == reloaded.get_input_embeddings().weight.data_ptr())
    if (unpadded_logit_error > 1.0e-4 or padded_logit_error > 1.0e-4
            or loss_error > 1.0e-5 or cached_logit_error > 3.0e-4
            or reference_cache_error > 3.0e-4 or scratch_cache_error > 3.0e-4
            or reload_error > 1.0e-7 or not exact_generation
            or not tied or not reload_tied):
        raise IntegrityError(f"experiment {package} scratch GPT parity failed")
    payload = {
        "completed": True,
        "experiment": package,
        "source": source,
        "source_revision": revision,
        "base_record_sha256": sha256(base_record_path),
        "max_abs_unpadded_logit_error": unpadded_logit_error,
        "max_abs_padded_logit_error": padded_logit_error,
        "abs_loss_error": loss_error,
        "max_abs_cached_logit_error": cached_logit_error,
        "reference_cache_error": reference_cache_error,
        "scratch_cache_error": scratch_cache_error,
        "generation_tokens_equal": exact_generation,
        "generation_diagnostic": generation_diagnostic,
        "tied_parameter_and_storage": tied and reload_tied,
        "reload_logit_error": reload_error,
        "max_new_tokens": 32,
    }
    return _write_once(record_path, payload)


def _stop_callback(stop_after):
    from transformers import TrainerCallback

    class stop_at_update(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= stop_after:
                control.should_training_stop = True
            return control

    return stop_at_update()


def _merge_logit_check(cfg, adapter_dir, merged_dir):
    import torch
    from train.model_utils import (load_model_and_tokenizer,
                                   merge_model_dtype)

    dtype = merge_model_dtype(cfg)
    live, tokenizer = load_model_and_tokenizer(
        cfg, "cuda", adapter_path=adapter_dir, dtype=dtype)
    baked, _ = load_model_and_tokenizer(
        cfg, "cuda", model_path=merged_dir, dtype=dtype)
    encoded = tokenizer("Adapter merge check", return_tensors="pt").to("cuda")
    live.eval()
    baked.eval()
    with torch.no_grad():
        left = live(**encoded).logits
        right = baked(**encoded).logits
    error = float((left - right).abs().max().item())
    if error > 1.0e-4:
        raise IntegrityError(
            "experiment 046 adapter and merged logits differ: "
            f"max abs error {error:.6g} exceeds 0.0001")
    del live, baked, left, right
    gc.collect()
    torch.cuda.empty_cache()
    return {"max_abs_logit_error": error, "comparison_dtype": dtype}


def _adapter_merge_check(cfg, adapter_dir, merged_dir, grpo_dir):
    import json
    import torch
    from data.splits import build_grpo_split
    from eval.harness import run_logprob_rows
    from scripts.s28_compose import ensure_bound_merge, verify_binding
    from train.model_utils import merge_adapter

    merged = ensure_bound_merge(
        adapter_dir, merged_dir, grpo_dir,
        merge_fn=lambda: merge_adapter(cfg, adapter_dir, merged_dir))
    local_check = _merge_logit_check(cfg, adapter_dir, merged)

    # run the two pool probes separately after the local comparison is released
    diagnostic_dir = os.path.join(os.path.dirname(adapter_dir), "merge_diagnostic")
    shutil.rmtree(diagnostic_dir, ignore_errors=True)
    rows = build_grpo_split()
    run_logprob_rows(
        cfg, "cuda", rows, adapter_path=adapter_dir,
        condition="adapter", out_dir=diagnostic_dir, include_scores=True)
    gc.collect()
    torch.cuda.empty_cache()
    run_logprob_rows(
        cfg, "cuda", rows, model_path=merged,
        condition="merged", out_dir=diagnostic_dir, include_scores=True)
    with open(os.path.join(diagnostic_dir, "adapter_logprob_preds.jsonl")) as handle:
        adapter_rows = [json.loads(line) for line in handle]
    with open(os.path.join(diagnostic_dir, "merged_logprob_preds.jsonl")) as handle:
        merged_rows = [json.loads(line) for line in handle]
    if (len(adapter_rows) != len(rows) or len(merged_rows) != len(rows)
            or any((left["subject"], left["gold"]) != (right["subject"], right["gold"])
                   for left, right in zip(adapter_rows, merged_rows))):
        raise IntegrityError("experiment 046 merge diagnostic rows do not align")

    def top_two_margin(row):
        values = sorted(row["letter_logits"].values(), reverse=True)
        return values[0] - values[1]

    changed = [i for i, (left, right) in enumerate(zip(adapter_rows, merged_rows))
               if left["pred"] != right["pred"]]
    changed_set = set(changed)
    unchanged = [i for i in range(len(rows)) if i not in changed_set]
    agreement = 1.0 - len(changed) / len(rows)
    adapter_acc = sum(row["correct"] for row in adapter_rows) / len(rows)
    merged_acc = sum(row["correct"] for row in merged_rows) / len(rows)
    if agreement < 0.90 or abs(adapter_acc - merged_acc) > 0.05:
        raise IntegrityError("experiment 046 adapter and merged development probes differ grossly")
    margins = {
        "changed": {
            "adapter": ([top_two_margin(adapter_rows[i]) for i in changed]),
            "merged": ([top_two_margin(merged_rows[i]) for i in changed]),
        },
        "unchanged": {
            "adapter": ([top_two_margin(adapter_rows[i]) for i in unchanged]),
            "merged": ([top_two_margin(merged_rows[i]) for i in unchanged]),
        },
    }
    margin_means = {
        group: {kind: (sum(values) / len(values) if values else None)
                for kind, values in by_kind.items()}
        for group, by_kind in margins.items()
    }
    shutil.rmtree(diagnostic_dir)
    binding = verify_binding(adapter_dir, merged_dir)
    return {
        "verified": True,
        **local_check,
        "development_rows": len(rows),
        "answer_agreement": agreement,
        "adapter_accuracy": adapter_acc,
        "merged_accuracy": merged_acc,
        "changed_rows": len(changed),
        "top_two_margin_means": margin_means,
        "gross_mismatch_limits": {"minimum_agreement": 0.90,
                                  "maximum_accuracy_difference": 0.05},
        "binding": binding,
    }


def _merge(cfg, adapter_dir, merged_dir, grpo_dir):
    from scripts.s28_compose import ensure_bound_merge, verify_binding
    from train.model_utils import merge_adapter

    # keep an existing checkpoint tied to the same merged SFT start
    ensure_bound_merge(
        adapter_dir, merged_dir, grpo_dir,
        merge_fn=lambda: merge_adapter(cfg, adapter_dir, merged_dir))
    return verify_binding(adapter_dir, merged_dir)


def _run_sft(cfg, base_record_path, out_dir, result_dir, pilot=False,
             pilot_merged_dir=None, pilot_grpo_dir=None):
    from data.gsm8k import load_sft
    from scripts.s46_gpt2_xl import commit_sft_training
    from train.sft import run_sft_rows

    base = read_json(base_record_path)
    run_name = f"{os.path.basename(result_dir)}_sft"
    recorded = []
    setup = {}

    def trainer_setup(trainer):
        setup.update(_check_lora_setup(trainer, "sft"))
        if pilot:
            trainer.add_callback(_stop_callback(5))
        else:
            trainer.add_callback(_wandb_callback(
                result_dir, cfg, "sft", recorded))

    if pilot:
        cfg.log.save_steps = 5
        cfg.log.save_total_limit = 1
        cfg.train.sft_val_examples = 0
    rows = load_sft(limit=32 if pilot else None)
    eval_rows = None if pilot else load_sft(
        split="test", limit=cfg.train.sft_val_examples)
    run_sft_rows(
        cfg, "cuda", rows, base["artifact"], "local",
        os.path.relpath(base["artifact"], cfg.output_root), out_dir=out_dir,
        report_to=[] if pilot else ["wandb"], log_dir=result_dir,
        run_name=run_name, allow_oom_fallback=False, eval_rows=eval_rows,
        trainer_setup=trainer_setup)
    if pilot:
        if not pilot_merged_dir or not pilot_grpo_dir:
            raise ValueError("experiment 046 pilot SFT needs its merge and GRPO paths")
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        merge = _adapter_merge_check(
            cfg, out_dir, pilot_merged_dir, pilot_grpo_dir)
        return {"updates": 5, "setup": setup, "merge": merge,
                "adapter_reload_verified": True,
                "merged_dir": pilot_merged_dir}
    if not recorded:
        raise IntegrityError("experiment 046 SFT W&B run was not recorded")
    from scripts.s42_worker import _wandb_records
    return commit_sft_training(
        cfg, result_dir, out_dir, base_record_path,
        _wandb_records(result_dir, cfg, package="046", stage="sft"))


def _run_grpo(cfg, start_dir, out_dir, result_dir, precision_path,
              sft_record_path=None, pilot_stop=None):
    from data.splits import build_grpo_split
    from scripts.s35_matched_grpo import PrecisionAudit, parity_fingerprint
    from scripts.s42_worker import _completion_length_gate
    from scripts.s46_gpt2_xl import (commit_grpo_training,
                                     verify_grpo_adapter_base)
    from train.grpo import run_grpo_rows

    run_name = f"{os.path.basename(result_dir)}_grpo"
    recorded = []
    length_seen = []
    audit = PrecisionAudit(stop_after=pilot_stop)
    setup = {}
    rows = build_grpo_split()
    prompts = [row["prompt"] for row in rows[:2]]
    reference_before = {}
    resume_memory = []
    initial_step = _latest_checkpoint_step(out_dir)
    if pilot_stop == 15 and initial_step >= 15:
        if not os.path.exists(precision_path):
            raise IntegrityError("experiment 046 pilot phase A has no precision record")
        return {"updates": 15, "precision": read_json(precision_path),
                "resume_verified": False, "phase_start_step": initial_step}
    if pilot_stop == 30 and initial_step != 15:
        raise IntegrityError("experiment 046 pilot phase B must resume checkpoint 15")

    def trainer_setup(trainer):
        resume_memory.append(_cuda_memory_point("trainer ready"))
        if pilot_stop == 30:
            _instrument_resume_memory(trainer, resume_memory)
        setup.update(_check_lora_setup(trainer, "grpo"))
        audit.attach(trainer)
        reference_before.update(parity_fingerprint(
            trainer, trainer.processing_class, prompts,
            {"experiment": "046", "stage": "merged_sft",
             "path": os.path.abspath(start_dir)}, cfg, "A"))
        trainer.add_callback(_completion_length_gate(length_seen, package="046"))
        if pilot_stop is None:
            trainer.add_callback(_wandb_callback(
                result_dir, cfg, "grpo", recorded))

    def trainer_finished(trainer):
        resume_memory.append(_cuda_memory_point("training finished"))
        report = audit.finish(trainer)
        reference_after = parity_fingerprint(
            trainer, trainer.processing_class, prompts,
            {"experiment": "046", "stage": "merged_sft",
             "path": os.path.abspath(start_dir)}, cfg, "A")
        report["reference_behaviour"] = _reference_behaviour_report(
            reference_before, reference_after)
        report["lora_setup"] = setup
        report["wandb_run_record"] = recorded[0] if recorded else None
        report["completion_length_verified"] = bool(length_seen)
        report["resume_memory"] = resume_memory
        atomic_write_json(precision_path, report)

    # the two pilot phases use real checkpoints and the unchanged 500 update schedule
    if pilot_stop:
        cfg.log.save_steps = 15
        cfg.log.save_total_limit = 2
        resume_memory.append(_cuda_memory_point("before trainer"))
    run_grpo_rows(
        cfg, "cuda", rows, start_dir, "local",
        os.path.relpath(start_dir, cfg.output_root), out_dir=out_dir,
        report_to=[] if pilot_stop else ["wandb"], log_dir=result_dir,
        run_name=run_name, allow_oom_fallback=False,
        trainer_setup=trainer_setup, trainer_finished=trainer_finished,
        save_final=pilot_stop in (30, None))
    if pilot_stop != 15:
        precision = read_json(precision_path)
        precision["adapter_base"] = verify_grpo_adapter_base(
            cfg, out_dir, start_dir)
        atomic_write_json(precision_path, precision)
    if pilot_stop:
        end_step = _latest_checkpoint_step(out_dir)
        return {"updates": pilot_stop, "precision": read_json(precision_path),
                "resume_verified": pilot_stop == 30
                and initial_step == 15 and end_step >= 30,
                "phase_start_step": initial_step}
    if not recorded:
        raise IntegrityError("experiment 046 GRPO W&B run was not recorded")
    return commit_grpo_training(
        cfg, result_dir, out_dir, sft_record_path, start_dir, precision_path)


def _run_eval(cfg, branch, scratch_dir, adapter_dir=None, model_path=None,
              package="046"):
    from data.splits import build_eval_split
    from eval.harness import run_eval_rows, run_logprob_rows

    _check_eval_source(branch, adapter_dir, model_path, package=package)
    os.makedirs(scratch_dir, exist_ok=True)
    if os.listdir(scratch_dir):
        raise IntegrityError(f"{scratch_dir}: evaluation scratch is not empty")
    # both endpoints finish in scratch before their manifest can commit
    rows = build_eval_split()
    generated = run_eval_rows(
        cfg, "cuda", rows, adapter_path=adapter_dir, model_path=model_path,
        condition=branch, out_dir=scratch_dir)
    probe = run_logprob_rows(
        cfg, "cuda", rows, adapter_path=adapter_dir, model_path=model_path,
        condition=branch, out_dir=scratch_dir)
    atomic_write_json(os.path.join(scratch_dir, f"{branch}_eval.json"), generated)
    atomic_write_json(os.path.join(scratch_dir, f"{branch}_logprob.json"), probe)
    return {"generated": generated, "probe": probe}


latest_checkpoint_step = _latest_checkpoint_step
check_lora_setup = _check_lora_setup
reference_behaviour_report = _reference_behaviour_report
run_eval = _run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=(
        "convert", "parity", "pilot_sft", "pilot_grpo_a", "pilot_grpo_b",
        "sft", "merge", "grpo", "eval"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--scratch-dir")
    parser.add_argument("--record")
    parser.add_argument("--base-record")
    parser.add_argument("--sft-record")
    parser.add_argument("--start-dir")
    parser.add_argument("--precision-report")
    parser.add_argument("--branch", choices=("base", "sft", "grpo"))
    parser.add_argument("--adapter-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--grpo-dir")
    args = parser.parse_args()

    import torch
    from config import load_config
    from obs import _versions, gpu_snapshot
    from scripts.s46_gpt2_xl import (ALLOCATOR_CONFIG, check_card_lock,
                                     check_card_vram, check_registered_config,
                                     commit_eval, eval_state)
    from train.model_utils import set_all_seeds

    cfg = check_registered_config(load_config(args.config))
    allocator_config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if allocator_config != ALLOCATOR_CONFIG:
        raise IntegrityError("experiment 046 CUDA allocator setting changed")
    # scientific tasks stay blocked until the technical pilot selects this card
    if args.task not in ("convert", "parity", "pilot_sft",
                         "pilot_grpo_a", "pilot_grpo_b"):
        lock = check_card_lock(cfg)
        total_vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        check_card_vram(lock, total_vram_gib)
    set_all_seeds(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    status = {"completed": False, "task": args.task, "seed": cfg.train.seed,
              "allocator_config": allocator_config,
              "gpu": gpu_snapshot(), "versions": _versions()}
    # write one status for success, interruption or failure before the process exits
    started = time.time()
    failure = None
    result = None
    start_step = _latest_checkpoint_step(args.out_dir or "")
    try:
        if args.task == "convert":
            result = _convert(
                cfg, args.scratch_dir, args.out_dir, args.record)
        elif args.task == "parity":
            result = _parity(cfg, args.base_record, args.record)
        elif args.task in ("pilot_sft", "sft"):
            result = _run_sft(
                cfg, args.base_record, args.out_dir, args.result_dir,
                pilot=args.task == "pilot_sft",
                pilot_merged_dir=args.start_dir,
                pilot_grpo_dir=args.grpo_dir)
        elif args.task == "merge":
            result = _merge(
                cfg, args.adapter_dir, args.start_dir, args.out_dir)
        elif args.task in ("pilot_grpo_a", "pilot_grpo_b", "grpo"):
            stop = 15 if args.task == "pilot_grpo_a" else (
                30 if args.task == "pilot_grpo_b" else None)
            result = _run_grpo(
                cfg, args.start_dir, args.out_dir, args.result_dir,
                args.precision_report, sft_record_path=args.sft_record,
                pilot_stop=stop)
        else:
            binding = read_json(args.record)
            state = eval_state(args.result_dir, args.branch, binding)
            if state == "tampered":
                raise IntegrityError(f"{args.result_dir}: endpoint changed")
            if state == "committed":
                result = read_json(os.path.join(
                    args.result_dir, f"046_{args.branch}_eval_manifest.json"))
            else:
                _run_eval(
                    cfg, args.branch, args.scratch_dir,
                    adapter_dir=args.adapter_dir, model_path=args.model_path)
                result = commit_eval(
                    args.result_dir, args.scratch_dir, args.branch, binding)
        status["completed"] = True
    except BaseException as exc:
        failure = exc
        status.update(classify_failure(exc, torch))
    finally:
        status["wall_seconds"] = round(time.time() - started, 1)
        status["start_step"] = start_step
        status["end_step"] = _latest_checkpoint_step(args.out_dir or "")
        status["completed_updates"] = max(0, status["end_step"] - start_step)
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            peak = torch.cuda.max_memory_reserved()
            peak_allocated = torch.cuda.max_memory_allocated()
            status["peak_reserved_gib"] = round(peak / 1024**3, 2)
            status["peak_reserved_fraction"] = peak / total
            status["peak_allocated_gib"] = round(peak_allocated / 1024**3, 2)
            status["peak_allocated_fraction"] = peak_allocated / total
            status["current_reserved_gib"] = round(
                torch.cuda.memory_reserved() / 1024**3, 2)
            status["current_allocated_gib"] = round(
                torch.cuda.memory_allocated() / 1024**3, 2)
            status["total_vram_gib"] = round(total / 1024**3, 2)
        if result is not None:
            status["result"] = result
        save_status(args.status, status)
        if failure is not None:
            raise failure


if __name__ == "__main__":
    main()
