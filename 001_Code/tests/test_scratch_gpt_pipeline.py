import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from checkpoint import (check_artifact_complete, config_hash,
                        write_artifact_type)
from config import MODEL_IDS, RunConfig, load_config
from enums import Condition, Mode, Scale


SCRATCH_PACKAGE = Path(__file__).parents[2] / "002_GPT2_From_Scratch"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def scratch_cfg(root, sft_mode="full", grpo_mode="full"):
    cfg = RunConfig()
    cfg.experiment = "042"
    cfg.exploratory = True
    cfg.mode = Mode.SMOKE
    cfg.scale = Scale.M124
    cfg.condition = Condition.SFT_GRPO
    cfg.output_root = str(root)
    cfg.train.dtype = "float32"
    cfg.train.max_steps = 1
    cfg.train.max_seq_len = 32
    cfg.train.sft_batch_size = 1
    cfg.train.sft_grad_accum = 1
    cfg.train.sft_val_examples = 0
    cfg.train.sft_parameter_mode = sft_mode
    if sft_mode == "full":
        cfg.train.sft_precision_mode = "fp32_master_bf16"
    cfg.train.grpo_parameter_mode = grpo_mode
    if grpo_mode == "full":
        cfg.train.grpo_precision_mode = "fp32_master_bf16"
    cfg.train.num_generations = 8
    cfg.grpo.max_completion_length = 4
    cfg.exec.grpo_batch_size = 8
    cfg.exec.grpo_grad_accum = 1
    cfg.log.save_steps = 5
    cfg.log.save_total_limit = 1
    cfg.log.log_completions = False
    cfg.log.trackers = ()
    cfg.eval.batch_size = 1
    cfg.eval.max_new_tokens = 2
    return cfg


@pytest.fixture
def scratch_start(tmp_path):
    # use tiny local model so the pipeline checks never touch MMLU
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from transformers import GPT2TokenizerFast
    from gpt2_scratch import ScratchGPT2Config, ScratchGPT2ForCausalLM

    start = tmp_path / "start"
    start.mkdir()
    model = ScratchGPT2ForCausalLM(ScratchGPT2Config(
        vocab_size=50257,
        n_positions=64,
        n_embd=16,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    ))
    model.save_pretrained(start)
    tokenizer = GPT2TokenizerFast.from_pretrained(
        "gpt2", revision=TOKENIZER_REVISION, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(start)
    (start / "model_base.json").write_text(
        json.dumps({"kind": "hf_id", "value": "gpt2"}))
    write_artifact_type(
        str(start), "full", family="scratch_gpt", scale="M124")
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    assert torch.isfinite(model.get_input_embeddings().weight).all()
    return tmp_path, start


def test_scratch_scales_use_their_gpt2_checkpoints():
    from train.model_utils import resolve_model_id

    expected = {
        Scale.M124: "gpt2",
        Scale.M355: "gpt2-medium",
        Scale.M774: "gpt2-large",
        Scale.GPT2_B15: "gpt2-xl",
    }
    assert {scale: MODEL_IDS[scale] for scale in expected} == expected
    for scale, model_id in expected.items():
        cfg = RunConfig()
        cfg.mode = Mode.SMOKE
        cfg.scale = scale
        assert resolve_model_id(cfg) == model_id


def test_historical_config_hashes_stay_unchanged():
    # keep every pre 042 resume identity unchanged
    digest = hashlib.sha256()
    paths = []
    for path in sorted(Path("configs").glob("0*.yaml")):
        cfg = load_config(path)
        if cfg.experiment < "042":
            paths.append(path)
            digest.update(f"{path}\0{config_hash(cfg)}\n".encode())
    assert len(paths) == 97
    assert digest.hexdigest() == "1c40a3e19e579b3c8280d7f858c7ba4ea09a63db814bd8758e04b09f1c3ab4e0"


def test_scratch_lora_targets_cover_attention_and_feed_forward():
    from train.model_utils import make_lora

    cfg = RunConfig()
    cfg.scale = Scale.M124
    assert set(make_lora(cfg).target_modules) == {
        "W_query", "W_key", "W_value", "out_proj", "layers.0", "layers.2",
    }


def test_scratch_config_modes_are_narrow(tmp_path):
    # alow the scratch modes without opening the same route for Qwen
    path = tmp_path / "config.yaml"
    path.write_text("""
experiment: '042'
exploratory: true
mode: SMOKE
scale: M124
condition: SFT_GRPO
train:
  sft_parameter_mode: full
  sft_precision_mode: fp32_master_bf16
  grpo_parameter_mode: full
  grpo_precision_mode: fp32_master_bf16
  a2_action: observe
""")
    cfg = load_config(path)
    assert cfg.scale == Scale.M124
    assert cfg.train.sft_parameter_mode == "full"
    assert cfg.train.grpo_parameter_mode == "full"

    unsafe = path.read_text().replace(
        "sft_precision_mode: fp32_master_bf16", "sft_precision_mode: legacy")
    path.write_text(unsafe)
    with pytest.raises(ValueError, match="full-model SFT"):
        load_config(path)

    path.write_text(unsafe.replace("scale: M124", "scale: B05"))
    with pytest.raises(ValueError, match="scratch GPT"):
        load_config(path)


def test_artifact_type_separates_full_models_and_adapters(tmp_path):
    # let the committed type decide which files are required
    from train.model_utils import load_scratch_model_and_tokenizer

    full = tmp_path / "full"
    full.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors"):
        (full / name).write_text("x")
    (full / "model_base.json").write_text(
        json.dumps({"kind": "hf_id", "value": "gpt2"}))
    write_artifact_type(
        str(full), "full", family="scratch_gpt", scale="M124")
    assert check_artifact_complete(str(full), expected="full")["family"] == "scratch_gpt"
    with pytest.raises(ValueError, match="expected adapter"):
        check_artifact_complete(str(full), expected="adapter")
    wrong_scale = RunConfig()
    wrong_scale.scale = Scale.M355
    with pytest.raises(ValueError, match="different scale"):
        load_scratch_model_and_tokenizer(wrong_scale, str(full))

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        (adapter / name).write_text("x")
    (adapter / "adapter_base.json").write_text(
        json.dumps({"kind": "local", "value": "start"}))
    write_artifact_type(
        str(adapter), "adapter", family="scratch_gpt", scale="M124")
    assert check_artifact_complete(str(adapter), expected="adapter")["kind"] == "adapter"
    with pytest.raises(ValueError, match="expected full"):
        check_artifact_complete(str(adapter), expected="full")


def test_starvation_fields_are_required_at_the_first_update():
    from obs import make_starvation_guard, normalise_starvation_metrics

    logs = {
        "rewards/accuracy_reward/mean": 0.5,
        "rewards/format_reward/mean": 0.25,
        "frac_reward_zero_std": 0.75,
    }
    assert normalise_starvation_metrics(logs) == {
        "mean_accuracy_reward": 0.5,
        "mean_format_reward": 0.25,
        "share_zero_reward_spread": 0.75,
    }
    with pytest.raises(RuntimeError, match="format_reward"):
        normalise_starvation_metrics({
            "rewards/accuracy_reward/mean": 0.5,
            "frac_reward_zero_std": 0.75,
        })

    callback = make_starvation_guard()
    callback.on_log(None, SimpleNamespace(global_step=1), None,
                    logs={"loss": 0.0, **logs})
    with pytest.raises(RuntimeError, match="format_reward"):
        callback.on_log(None, SimpleNamespace(global_step=2), None, logs={
            "loss": 0.0,
            "rewards/accuracy_reward/mean": 0.5,
            "frac_reward_zero_std": 0.75,
        })
    callback.on_train_end(None, SimpleNamespace(global_step=1), None)


def test_scratch_sft_merge_and_evaluation(scratch_start):
    torch = pytest.importorskip("torch")
    from eval.harness import run_eval_rows, run_logprob_rows
    from train.grpo import run_grpo_rows
    from train.model_utils import (load_model_and_tokenizer, merge_adapter)
    from train.sft import resolve_sft_start, run_sft_rows

    root, start = scratch_start
    rows = [
        {"prompt": "Question?", "target": "<think>ok</think>\n<answer>A</answer>"},
        {"prompt": "A longer question for padding?",
         "target": "<think>still ok</think>\n<answer>B</answer>"},
    ]
    outputs = {}
    for mode in ("full", "lora"):
        cfg = scratch_cfg(root, sft_mode=mode)
        cfg.train.sft_batch_size = 2
        out = root / f"sft_{mode}"
        log_dir = root / f"sft_{mode}_logs"
        label_checks = []
        loss_checks = []

        def check_labels(trainer):
            assert trainer.args.loss_type == "nll"
            assert trainer.args.use_cpu is True
            loss_checks.append(True)
            if mode != "full":
                return
            batch = next(iter(trainer.get_train_dataloader()))
            labels = batch["labels"]
            mask = batch["attention_mask"].bool()
            assert torch.all(labels[~mask] == -100)
            eos = trainer.processing_class.eos_token_id
            assert all(bool(((row_labels == eos) & row_mask).any())
                       for row_labels, row_mask in zip(labels, mask))
            label_checks.append(True)

        run_sft_rows(
            cfg, "cpu", rows, str(start), "local", "start",
            out_dir=str(out), report_to=[], log_dir=str(log_dir),
            allow_oom_fallback=False,
            trainer_setup=check_labels)
        outputs[mode] = out
        expected = "full" if mode == "full" else "adapter"
        assert check_artifact_complete(str(out), expected=expected)
        history = json.loads((log_dir / "sft_trainlog.json").read_text())
        losses = [float(row["loss"]) for row in history if "loss" in row]
        assert losses and all(math.isfinite(loss) for loss in losses)
        assert loss_checks == [True]
        if mode == "full":
            assert label_checks == [True]

    cfg = scratch_cfg(root)
    start_model, _ = load_model_and_tokenizer(cfg, "cpu", model_path=str(start))
    trained_model, _ = load_model_and_tokenizer(
        cfg, "cpu", model_path=str(outputs["full"]))
    start_state = start_model.state_dict()
    trained_state = trained_model.state_dict()
    # make sure a full update is not only changing one part of the model
    families = {
        "embeddings": ("tok_emb.", "pos_emb."),
        "attention": (".att.",),
        "feed_forward": (".ff.",),
        "normalisation": ("norm",),
    }
    for prefixes in families.values():
        names = [name for name in start_state
                 if any(prefix in name for prefix in prefixes)]
        assert names
        assert any(not torch.equal(start_state[name], trained_state[name])
                   for name in names)
    assert trained_model.get_input_embeddings().weight is trained_model.get_output_embeddings().weight
    assert (trained_model.get_input_embeddings().weight.data_ptr()
            == trained_model.get_output_embeddings().weight.data_ptr())

    cfg = scratch_cfg(root, sft_mode="lora")
    merged = root / "sft_merged"
    # compare the live adapter with the full model chaining will load
    merge_adapter(cfg, str(outputs["lora"]), str(merged))
    assert check_artifact_complete(str(merged), expected="full")
    cfg.train.sft_from_adapter = str(outputs["lora"].relative_to(root))
    resolved, kind, value = resolve_sft_start(cfg)
    assert (resolved, kind, value) == (
        str(merged), "local", str(merged.relative_to(root)))

    live_model, tok = load_model_and_tokenizer(
        cfg, "cpu", adapter_path=str(outputs["lora"]))
    merged_model, _ = load_model_and_tokenizer(
        cfg, "cpu", model_path=str(merged))
    enc = tok("merge check", return_tensors="pt")
    live_model.eval()
    merged_model.eval()
    with torch.no_grad():
        live_logits = live_model(**enc).logits
        merged_logits = merged_model(**enc).logits
    assert torch.allclose(live_logits, merged_logits, atol=1e-6, rtol=1e-6)

    eval_rows = [{
        "prompt": "Question?\n<think>",
        "probe": "Question?\nAnswer:",
        "gold": "A",
        "subject": "test_subject",
        "tier": "easy",
    }]
    probe = run_logprob_rows(
        cfg, "cpu", eval_rows, model_path=str(merged),
        condition="scratch_probe", out_dir=str(root / "eval"))
    generated = run_eval_rows(
        cfg, "cpu", eval_rows, model_path=str(merged),
        condition="scratch", out_dir=str(root / "eval"))
    assert set(probe) == {
        "condition", "metric", "n_items", "per_tier_acc", "per_subject_acc",
        "per_tier_ci", "overall_ci",
    }
    assert set(generated) == {
        "condition", "n_items", "per_tier_acc", "per_subject_acc",
        "per_tier_ci", "overall_ci", "format_compliance", "mean_len",
        "extraction_failure_rate",
    }

    # pass the adapter derived full start through the same GRPO route as science
    grpo_out = root / "grpo_from_merged"
    run_grpo_rows(
        cfg, "cpu", [{"prompt": "Question?\n<think>", "gold": "A",
                      "tier": "easy", "subject": "test_subject"}],
        str(merged), "local", "sft_merged", out_dir=str(grpo_out),
        report_to=[], log_dir=str(root / "grpo_from_merged_logs"),
        allow_oom_fallback=False)
    assert check_artifact_complete(str(grpo_out), expected="full")
    assert json.loads((grpo_out / "model_base.json").read_text()) == {
        "kind": "local", "value": "sft_merged"}


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_scratch_sft_oom_retry_rebuilds_the_model(scratch_start, mode):
    torch = pytest.importorskip("torch")
    from train.sft import run_sft_rows

    root, start = scratch_start
    cfg = scratch_cfg(root, sft_mode=mode)
    cfg.train.sft_batch_size = 2
    rows = [
        {"prompt": "Question?", "target": "<think>ok</think>\n<answer>A</answer>"},
        {"prompt": "Another question?", "target": "<think>ok</think>\n<answer>B</answer>"},
    ]
    models = []
    starts = []

    def fail_first_attempt(trainer):
        model = (trainer.model.get_base_model()
                 if hasattr(trainer.model, "get_base_model") else trainer.model)
        models.append(model)
        weight = model.get_input_embeddings().weight
        starts.append(weight.detach().clone())
        if len(models) == 1:
            # change the first model so a reused retry cannot pass unnoticed
            with torch.no_grad():
                weight.add_(1.0)
            raise RuntimeError("CUDA out of memory")

    run_sft_rows(
        cfg, "cpu", rows, str(start), "local", "start",
        out_dir=str(root / f"sft_retry_{mode}"), report_to=[],
        allow_oom_fallback=True, trainer_setup=fail_first_attempt)
    assert len(models) == 2
    assert models[0] is not models[1]
    assert torch.equal(starts[0], starts[1])


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_scratch_grpo_runs_with_bound_reference(scratch_start, mode):
    torch = pytest.importorskip("torch")
    from train.grpo import run_grpo_rows

    root, start = scratch_start
    cfg = scratch_cfg(root, grpo_mode=mode)
    rows = [{
        "prompt": "Question?\n<think>",
        "gold": "A",
        "tier": "easy",
        "subject": "test_subject",
    }]
    setup_seen = []

    def check_setup(trainer):
        # verify the reference path before the first GRPO update
        setup_seen.append(True)
        assert trainer.args.use_cpu is True
        assert trainer.args.disable_dropout is True
        assert trainer.model.config.resid_pdrop == 0.1
        assert trainer.model.config.embd_pdrop == 0.1
        assert trainer.model.config.attn_pdrop == 0.1
        assert all(module.p == 0.0 for module in trainer.model.modules()
                   if isinstance(module, torch.nn.Dropout))
        if mode == "full":
            assert trainer.ref_model is not None
            assert all(module.p == 0.0 for module in trainer.ref_model.modules()
                       if isinstance(module, torch.nn.Dropout))
            model_state = trainer.model.state_dict()
            ref_state = trainer.ref_model.state_dict()
            assert model_state.keys() == ref_state.keys()
            assert all(torch.equal(model_state[name], ref_state[name])
                       for name in model_state)
            assert not any(param.requires_grad for param in trainer.ref_model.parameters())
        else:
            assert trainer.ref_model is None

    out = root / f"grpo_{mode}"
    log_dir = root / f"grpo_{mode}_logs"
    run_grpo_rows(
        cfg, "cpu", rows, str(start), "local", "start",
        out_dir=str(out), report_to=[], log_dir=str(log_dir),
        allow_oom_fallback=False, trainer_setup=check_setup)
    assert setup_seen == [True]
    expected = "full" if mode == "full" else "adapter"
    assert check_artifact_complete(str(out), expected=expected)
    history = json.loads((log_dir / "grpo_trainlog.json").read_text())
    first = next(row for row in history if row.get("step") == 1)
    assert {
        "rewards/accuracy_reward/mean",
        "rewards/format_reward/mean",
        "frac_reward_zero_std",
    } <= set(first)


def test_scratch_grpo_resumes_the_same_two_step_schedule(scratch_start):
    pytest.importorskip("torch")
    from transformers import TrainerCallback
    from train.grpo import run_grpo_rows

    root, start = scratch_start
    cfg = scratch_cfg(root)
    cfg.train.max_steps = 2
    cfg.log.save_steps = 1
    out = root / "grpo_resume"
    logs = root / "grpo_resume_logs"
    rows = [{"prompt": "Question?\n<think>", "gold": "A",
             "tier": "easy", "subject": "test_subject"}]

    class stop_after_one(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_training_stop = True
            return control

    def stop_setup(trainer):
        trainer.add_callback(stop_after_one())

    run_grpo_rows(
        cfg, "cpu", rows, str(start), "local", "start",
        out_dir=str(out), report_to=[], log_dir=str(logs),
        allow_oom_fallback=False, trainer_setup=stop_setup)
    assert (out / "checkpoint-1" / "trainer_state.json").exists()

    # resume through the unchanged config instead of starting another schedule
    run_grpo_rows(
        cfg, "cpu", rows, str(start), "local", "start",
        out_dir=str(out), report_to=[], log_dir=str(logs),
        allow_oom_fallback=False)
    history = json.loads((logs / "grpo_trainlog.json").read_text())
    assert [row["step"] for row in history if "loss" in row] == [1, 2]
