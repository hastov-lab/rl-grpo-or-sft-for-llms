import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import load_config
from file_utils import read_json
from scripts import s51_b1_grpo as worker
from scripts.s12_eval import IntegrityError, sha256


def config(seed=0):
    return load_config(worker.CODE / "configs" / f"051_colab_gpt2_b15_b1_seed{seed}.yaml")


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_settings_match_historical_cold_xl(seed):
    cfg = config(seed)
    worker.validate_config(cfg)
    assert cfg.train.a2_action == "observe"
    assert cfg.train.max_steps == 500
    assert cfg.exec.gradient_checkpointing is False
    assert cfg.exec.allow_oom_fallback is False
    cfg.train.max_steps = 50
    with pytest.raises(IntegrityError, match="beyond"):
        worker.validate_config(cfg)


def test_paths_cannot_resume_049(tmp_path):
    # a symlink must not turn a new run name into the historical output directory
    cfg = config()
    cfg.output_root = str(tmp_path)
    root, result, checkpoint = worker.paths_for(cfg)
    assert result == root / "results/051_COLAB_FULL_GPT2_B15_base_grpo_seed0"
    assert checkpoint == root / "checkpoints/051_COLAB_FULL_GPT2_B15_base_grpo_seed0/grpo"
    result.parent.mkdir()
    old = root / "results/049_base_grpo_COLAB_FULL_GPT2_B15_base_grpo_seed0"
    old.mkdir()
    result.symlink_to(old, target_is_directory=True)
    with pytest.raises(IntegrityError, match="redirected"):
        worker.paths_for(cfg)


def test_generic_transaction_helpers_are_reused():
    # keep completion, resume and evaluation behaviour shared with 050
    from scripts import s50_b1_grpo as original
    for name in ("bind_run", "checkpoint_step", "digest", "evaluate", "paths_for",
                 "run_lock", "stopped_outcome", "verify_record"):
        assert getattr(worker, name) is getattr(original, name)


@pytest.mark.parametrize("role,source,accepted", [
    ("immutable_base", "gpt2-xl", True), ("sft", "gpt2-xl", False),
    ("immutable_base", "gpt2-medium", False)])
def test_source_requires_original_pretrained_xl(tmp_path, monkeypatch, role, source, accepted):
    # the same architecture is not enough when the starting weights differ
    import checkpoint
    import scripts.s46_gpt2_xl as original
    cfg = config()
    cfg.output_root = str(tmp_path)
    directory = tmp_path / "results" / worker.BASE_TAG
    directory.mkdir(parents=True)
    technical = tmp_path / "results/046_technical"
    technical.mkdir()
    for path in (directory / "046_base_manifest.json", directory / "046_base_eval_manifest.json",
                 technical / "046_xl_parity.json", technical / "046_token_audit.json"):
        path.write_text("{}")
    base = {"completed": True, "experiment": "046", "role": role, "source": source,
            "artifact": str(tmp_path / "base"), "artifact_sha256": {"weights": "test"}}
    monkeypatch.setattr(original, "verify_technical_records", lambda *a: {"base": base})
    monkeypatch.setattr(original, "eval_state", lambda *a: "committed")
    monkeypatch.setattr(checkpoint, "check_artifact_complete", lambda *a, **k: {
        "family": "scratch_gpt", "scale": "GPT2_B15"})
    if accepted:
        result = worker.source_for(cfg)
        assert result["path"] == base["artifact"]
        assert len(result["records"]) == 4
    else:
        with pytest.raises(IntegrityError, match="source"):
            worker.source_for(cfg)


def test_binding_is_b1_and_binds_scratch_sources(monkeypatch):
    import obs
    monkeypatch.setattr(worker, "source_for", lambda cfg: {"kind": "immutable_base", "path": "/base"})
    monkeypatch.setattr(obs, "versions", lambda: dict(worker.PINS))
    result = worker.run_binding(config(), [{}] * 245, [{}] * 1884)
    assert result["parser"] == "B1" and result["historical_comparator"] == "049_base_grpo_seed0"
    assert "scripts/s50_b1_grpo.py" in result["code"]
    assert "model.py" in result["scratch_code"]
    with pytest.raises(IntegrityError, match="245"):
        worker.run_binding(config(), [], [{}] * 1884)


def test_notebook_uses_xl_worker_at_launch_or_closeout():
    notebook = read_json(worker.CODE / "notebooks/051_colab_gpt2_xl_b1_base_grpo.ipynb")
    text = "\n".join("".join(c["source"]) for c in notebook["cells"])
    values, executed = {}, False
    fields = {"SEEDS_TO_RUN", "cu_before", "cu_after", "unit_rate"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            count = cell["execution_count"]
            assert count is None or (type(count) is int and count >= 0)
            assert all(output.get("output_type") != "error" for output in cell["outputs"])
            executed = executed or count is not None or bool(cell["outputs"])
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Name) and target.id in fields:
                        assert target.id not in values
                        values[target.id] = ast.literal_eval(node.value)
    assert set(values) == fields
    # Completed notebooks retain outputs and their entered compute fields.
    if executed:
        seeds = values["SEEDS_TO_RUN"]
        assert isinstance(seeds, tuple) and seeds
        assert len(seeds) == len(set(seeds))
        assert all(type(seed) is int and seed in (0, 1, 2) for seed in seeds)
        before, after, rate = (values[k] for k in ("cu_before", "cu_after", "unit_rate"))
        assert type(before) in (int, float) and before >= 0
        assert type(rate) in (int, float) and rate > 0
        assert after is None or (type(after) in (int, float) and 0 <= after <= before)
    else:
        assert values == {"SEEDS_TO_RUN": (0,), "cu_before": None,
                          "cu_after": None, "unit_rate": None}
    for phrase in ("75 <= total_gb < 90", "free_gb >= 45",
                   "scripts/s51_b1_grpo.py", "051_colab_gpt2_b15_b1_seed",
                   "sys.path.insert(0, scratch_package)"):
        assert phrase in text
    assert "scripts/s49_worker.py" not in text and "scripts/run.py" not in text
    assert "snapshot_download" not in text
    assert text.index("PYTORCH_CUDA_ALLOC_CONF") < text.index("import subprocess, torch")


def test_general_runner_rejects_051_before_loading_data():
    source = (worker.CODE / "scripts/run.py").read_text()
    assert 'if cfg.experiment == "051":' in source
    assert 'raise ValueError("experiment 051 requires scripts/s51_b1_grpo.py")' in source


@pytest.mark.parametrize("old,new", [("exploratory: true", "exploratory: false"),
                                     ("scale: GPT2_B15", "scale: M774"),
                                     ("condition: BASE_GRPO", "condition: SFT_GRPO")])
def test_observe_mode_stays_limited_to_registered_xl_cold(tmp_path, old, new):
    text = (worker.CODE / "configs/051_colab_gpt2_b15_b1_seed0.yaml").read_text()
    path = tmp_path / "wrong.yaml"
    path.write_text(text.replace(old, new))
    with pytest.raises((ValueError, KeyError)):
        load_config(path)


@pytest.mark.parametrize("kind", ("complete", "a1", "technical"))
def test_training_keeps_xl_gates_and_separate_terminal_records(tmp_path, monkeypatch, kind):
    # exercise callback wiring and terminal records without allocating the XL model
    import obs
    import train.grpo as grpo
    import train.parser_diagnostics as diagnostic_module
    import scripts.s35_matched_grpo as precision
    import scripts.s42_worker as records
    import scripts.s46_worker as gates
    import scripts.s49_gpt2_xl_symmetry as lineage
    from scripts.s12_eval import ADAPTER_FILES

    cfg = config()
    cfg.output_root = str(tmp_path)
    _, result, checkpoint = worker.paths_for(cfg)
    result.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    binding = {"parser": "B1", "source": {"kind": "immutable_base", "path": str(tmp_path / "base")}}
    events = []
    diagnostics = {"completed_updates": 0, "responses_checked": 0,
                   "h_only_accuracy_reward": 0, "b1_only_accuracy_reward": 0}
    diagnostic = SimpleNamespace(record={"steps": []}, summary=lambda: dict(diagnostics),
                                 close_attempt=lambda: events.append("close"))

    def attach(trainer, passed_cfg, out, cp, digest):
        assert passed_cfg is cfg and out == str(result) and cp == str(checkpoint)
        assert digest == worker.digest(binding)
        events.append("diagnostics")
        return diagnostic

    class audit:
        def __init__(self, **kwargs):
            assert kwargs == {"allow_initial_zero_gradient": True}

        def attach(self, trainer):
            events.append("precision_attach")

        def finish(self, trainer):
            events.append("precision_finish")
            return {"gradient_dtypes": ["float32"]}

    monkeypatch.setattr(diagnostic_module, "attach_parser_diagnostics", attach)
    monkeypatch.setattr(precision, "PrecisionAudit", audit)
    monkeypatch.setattr(precision, "parity_fingerprint", lambda *a: {"reference_logprobs": [[0.0]]})
    monkeypatch.setattr(gates, "check_lora_setup", lambda *a, **k: events.append("lora") or {"verified": True})
    monkeypatch.setattr(gates, "reference_behaviour_report", lambda *a, **k: events.append("reference") or {"verified": True})
    monkeypatch.setattr(records, "completion_length_gate", lambda seen, **k: seen.append(512) or object())
    monkeypatch.setattr(records, "capture_wandb_run", lambda *a, **k: events.append("wandb"))
    monkeypatch.setattr(records, "wandb_records", lambda *a, **k: [{"run_id": "test"}])
    monkeypatch.setattr(records, "step_rows", lambda *a: [{"step": i} for i in range(1, 501)])
    monkeypatch.setattr(records, "starvation_summary", lambda rows: {"severely_reward_starved": False})
    monkeypatch.setattr(lineage, "verify_adapter_base", lambda *a: events.append("adapter_base") or {"verified": True})
    monkeypatch.setattr(obs, "resolve_report_to", lambda cfg: ["wandb"])
    monkeypatch.setattr(obs, "write_manifest", lambda cfg, out, extra: (Path(out) / "manifest.json").write_text("{}"))
    monkeypatch.setattr(obs, "verify_a2_verdict", lambda path, cfg, rows: events.append("a2") or True)
    monkeypatch.setattr(obs, "gpu_snapshot", lambda: {})

    def run(passed_cfg, device, rows, model_src, base_kind, base_value, **kwargs):
        assert passed_cfg is cfg and device == "cuda" and model_src == binding["source"]["path"]
        assert base_kind == "local" and base_value == "base"
        assert kwargs["allow_oom_fallback"] is False and kwargs["save_final"] is True
        callbacks = []
        trainer = SimpleNamespace(processing_class=object(), add_callback=callbacks.append)
        kwargs["trainer_setup"](trainer)
        for callback in callbacks:
            if hasattr(callback, "on_log"):
                callback.on_log(None, SimpleNamespace(global_step=1), None, logs={"loss": 0.1})
        steps = 50 if kind == "a1" else 500
        diagnostics.update(completed_updates=steps, responses_checked=steps * 8)
        diagnostic.record["steps"] = [{"training_log": {"completions/clipped_ratio": .5}} for _ in range(steps)]
        (result / "parser_diagnostics.json").write_text(json.dumps(diagnostic.record))
        if kind == "a1":
            raise RuntimeError("sentinel abort: a1: over 30%")
        if kind == "technical":
            raise RuntimeError("CUDA out of memory")
        kwargs["trainer_finished"](trainer)
        for name in (*ADAPTER_FILES, "a2_verdict.json"):
            (checkpoint / name).write_text("{}")
        (result / "grpo_trainlog.json").write_text("{}")

    monkeypatch.setattr(grpo, "run_grpo_rows", run)
    if kind == "technical":
        with pytest.raises(RuntimeError, match="out of memory"):
            worker.train(cfg, [{"prompt": "development"}], result, checkpoint, binding)
        assert (result / "last_failure.json").exists()
    else:
        outcome = worker.train(cfg, [{"prompt": "development"}], result, checkpoint, binding)
        assert outcome["completed"] is (kind == "complete")
        assert (result / "terminal_stop.json").exists() is (kind == "a1")
        if kind == "complete":
            assert all(e in events for e in ("reference", "a2", "precision_finish", "adapter_base"))
            assert outcome["files"]["results/051_precision.json"] == sha256(str(result / "051_precision.json"))
        else:
            assert outcome["outcome"] == {"kind": "A1", "step": 50, "mean_clipped_ratio": .5}
            assert "precision_finish" not in events and "a2" not in events
    assert all(e in events for e in ("lora", "precision_attach", "diagnostics", "wandb", "close"))
    assert (result / "training_complete.json").exists() is (kind == "complete")
