# lora sft on the tagged gsm8k data, teaches the response template
import os

from data.gsm8k import load_sft
from train.model_utils import (resolve_model_id, make_lora, set_all_seeds, resolve_dtype,
                               write_adapter_base, merge_adapter, is_scratch_gpt,
                               precision_settings, load_scratch_model,
                               load_scratch_tokenizer,
                               record_scratch_artifact)


def to_sft_text(example):
    # what the model sees during sft: the question then the tagged answer to copy
    return f"{example['prompt']}\n{example['target']}"


def build_sft_rows(rows, eos, completion_only):
    # two dataset shapes: one text field (loss on everything, the default), or
    # prompt/completion so trl can put the loss on the completion only
    out = []
    for r in rows:
        if completion_only:
            comp = r["target"]
            if eos and not comp.endswith(eos):
                comp = comp + eos
            out.append({"prompt": r["prompt"] + "\n", "completion": comp})
        else:
            t = to_sft_text(r)
            if eos and not t.endswith(eos):
                t = t + eos
            out.append({"text": t})
    return out


def mc_sft_rows(grpo_rows):
    # Supervised targets for the mc-sft control: the exact prompts grpo trains on,
    # a fixed neutral think and the gold letter in the tags. Tests whether studying
    # the answer key reproduces the grpo gain
    from data.prompts import MC_SFT_THINK
    from data.tagging import wrap_continuation
    return [{"prompt": r["prompt"], "target": wrap_continuation(MC_SFT_THINK, r["gold"])} for r in grpo_rows]


def resolve_sft_start(cfg):
    # Experiment 008: when sft_from_adapter is set, bake that adapter into the base
    # first and train the new lora on the merged model, same pattern grpo uses for
    # chaining. Returns the model source plus the provenance to record
    if not cfg.train.sft_from_adapter:
        return resolve_model_id(cfg), "hf_id", resolve_model_id(cfg)
    adapter_dir = os.path.join(cfg.output_root, cfg.train.sft_from_adapter)
    if not os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")):
        # fail loudly before hours of colab time, a missing drive sync is the
        # likely cause and a cryptic peft error would hide it
        raise FileNotFoundError(f"sft_from_adapter points at {adapter_dir} but there is no adapter there")
    source_merged = os.path.join(os.path.dirname(adapter_dir), "sft_merged")
    if os.path.exists(os.path.join(source_merged, "config.json")):
        # reuse the merged base made from this exact adapter by its parent run
        return source_merged, "local", os.path.relpath(source_merged, cfg.output_root)
    # keep fallback merges shared only when the source adapter is the same
    source_tag = os.path.basename(os.path.dirname(adapter_dir))
    merged_dir = os.path.join(cfg.output_root, "checkpoints",
                              f"{cfg.experiment}_{cfg.scale.name}_{source_tag}_merged_base")
    merge_adapter(cfg, adapter_dir, merged_dir)
    return merged_dir, "local", os.path.relpath(merged_dir, cfg.output_root)


def _run_sft_rows(cfg, rows, model_src, base_kind, base_val, out_dir,
                  report_to, log_dir, run_name, allow_oom_fallback,
                  eval_rows=None, eval_rows_fn=None, trainer_setup=None,
                  use_cpu=None):
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTTrainer, SFTConfig
    from checkpoint import save_metrics, has_checkpoint, check_config_hash

    # add eos to the end of each target so the model learns to stop after </answer>
    settings = precision_settings(cfg, "sft")
    if is_scratch_gpt(cfg):
        tok = load_scratch_tokenizer(cfg, model_src)
    else:
        tok = AutoTokenizer.from_pretrained(model_src)
    eos = tok.eos_token or ""
    ds = Dataset.from_list(build_sft_rows(rows, eos, cfg.train.sft_completion_only))

    if eval_rows_fn is not None:
        eval_rows = eval_rows_fn()
    eval_ds = None
    if eval_rows:
        eval_ds = Dataset.from_list(build_sft_rows(
            eval_rows, eos, cfg.train.sft_completion_only))

    from obs import stamp_trainer_logs, oom_ladder, prepare_wandb, wandb_finish

    def attempt(batch, accum, checkpointing):
        # the frozen quantity is the effective batch (16), the split is memory plumbing
        assert batch * accum == cfg.train.sft_batch_size * cfg.train.sft_grad_accum, \
            "sft batch x accum must keep the effective batch"
        trainer_model = (load_scratch_model(
            cfg, model_src, settings["model_dtype"])
            if is_scratch_gpt(cfg) else model_src)
        sft_kwargs = dict(
            output_dir=out_dir,
            max_steps=cfg.train.max_steps,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=accum,
            gradient_checkpointing=checkpointing,
            learning_rate=cfg.train.sft_lr,
            max_length=cfg.train.max_seq_len,
            model_init_kwargs={"dtype": resolve_dtype(cfg)},
            logging_steps=1,
            save_steps=cfg.log.save_steps,
            save_total_limit=cfg.log.save_total_limit,
            report_to=report_to if report_to is not None else [],
            run_name=run_name,
            seed=cfg.train.seed,
        )
        if is_scratch_gpt(cfg):
            import torch
            sft_kwargs.pop("model_init_kwargs")
            sft_kwargs["loss_type"] = "nll"
            sft_kwargs["use_cpu"] = (
                not torch.cuda.is_available() if use_cpu is None else use_cpu)
            if settings["bf16"]:
                sft_kwargs["bf16"] = True
        if not cfg.train.sft_completion_only:
            sft_kwargs["dataset_text_field"] = "text"
        if eval_ds is not None:
            sft_kwargs["eval_strategy"] = "steps"
            sft_kwargs["eval_steps"] = cfg.log.save_steps
            sft_kwargs["per_device_eval_batch_size"] = batch
        args = SFTConfig(**sft_kwargs)
        trainer_kwargs = dict(
            model=trainer_model,
            args=args,
            train_dataset=ds,
            eval_dataset=eval_ds,
            peft_config=(make_lora(cfg) if settings["parameter_mode"] == "lora" else None),
        )
        if is_scratch_gpt(cfg):
            trainer_kwargs["processing_class"] = tok
        trainer = SFTTrainer(**trainer_kwargs)
        stamp_trainer_logs(trainer)
        if trainer_setup:
            trainer_setup(trainer)
        trainer.train(resume_from_checkpoint=True if has_checkpoint(out_dir) else None)
        return trainer

    tracking = prepare_wandb(cfg, report_to)
    try:
        check_config_hash(out_dir, cfg)
        # use the same oom fallback order as grpo
        # 7b runs disable it because the no-mutation rule covers the full run
        if allow_oom_fallback:
            trainer = oom_ladder(attempt,
                                 per_device=cfg.train.sft_batch_size,
                                 accum=cfg.train.sft_grad_accum,
                                 checkpointing=cfg.exec.gradient_checkpointing)
        else:
            trainer = attempt(cfg.train.sft_batch_size, cfg.train.sft_grad_accum,
                              cfg.exec.gradient_checkpointing)
        trainer.save_model(out_dir)
        if is_scratch_gpt(cfg):
            record_scratch_artifact(
                cfg, tok, out_dir, "sft", base_kind, base_val)
        else:
            write_adapter_base(out_dir, base_kind, base_val)
        if log_dir:
            # keep the full per step training log so loss can be plotted later without rerunning
            save_metrics(f"{log_dir}/sft_trainlog.json", trainer.state.log_history)
    finally:
        if tracking["enabled"]:
            wandb_finish()
    return out_dir


def run_sft_rows(cfg, device, rows, model_src, base_kind, base_val,
                 out_dir="checkpoints/sft", report_to=None, log_dir=None,
                 run_name=None, allow_oom_fallback=True, eval_rows=None,
                 trainer_setup=None):
    from checkpoint import write_config_hash

    set_all_seeds(cfg.train.seed)
    write_config_hash(out_dir, cfg)
    return _run_sft_rows(
        cfg, rows, model_src, base_kind, base_val, out_dir,
        report_to, log_dir, run_name, allow_oom_fallback,
        eval_rows=eval_rows, trainer_setup=trainer_setup,
        use_cpu=device == "cpu")


def run_sft(cfg, device, out_dir="checkpoints/sft", report_to=None, log_dir=None, run_name=None,
            allow_oom_fallback=True):
    from checkpoint import write_config_hash

    if cfg.train.sft_data == "synthetic_facts":
        raise ValueError("synthetic facts require run_sft_rows")
    set_all_seeds(cfg.train.seed)
    write_config_hash(out_dir, cfg)
    model_src, base_kind, base_val = resolve_sft_start(cfg)

    limit = 8 if cfg.mode.name == "SMOKE" else None
    if cfg.train.sft_data == "mmlu_mc":
        from data.splits import build_grpo_split
        rows = mc_sft_rows(build_grpo_split(limit_per_subject=4 if limit else None))
    else:
        rows = load_sft(limit=limit)

    # held out gsm8k test slice gives a val loss curve next to the train loss
    eval_rows_fn = None
    if cfg.train.sft_val_examples:
        eval_rows_fn = lambda: load_sft(split="test", limit=cfg.train.sft_val_examples)
    return _run_sft_rows(
        cfg, rows, model_src, base_kind, base_val, out_dir,
        report_to, log_dir, run_name, allow_oom_fallback,
        eval_rows_fn=eval_rows_fn, use_cpu=device == "cpu")
