import ast
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import load_config
from file_utils import read_json
from scripts import s52_b1_warm_grpo as worker
from scripts.s12_eval import ADAPTER_FILES, IntegrityError


def config(seed=0):
    return load_config(worker.CODE / "configs" / f"052_colab_b05_b1_warm_seed{seed}.yaml")


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_settings_match_historical_warm_b05(seed):
    cfg = config(seed)
    worker.validate_config(cfg)
    assert cfg.condition.name == "SFT_GRPO"
    assert cfg.train.max_steps == 500 and cfg.train.a2_action == "abort"
    assert cfg.exec.gradient_checkpointing is False
    assert cfg.exec.allow_oom_fallback is False
    cfg.train.grpo_lr *= 2
    with pytest.raises(IntegrityError, match="beyond"):
        worker.validate_config(cfg)


@pytest.mark.parametrize("experiment", ("005", "006", "050", "051"))
def test_historical_run_identity_is_rejected(experiment):
    cfg = config()
    cfg.experiment = experiment
    with pytest.raises(IntegrityError, match="052 requires"):
        worker.validate_config(cfg)


def parent_fixture(tmp_path, seed=0):
    # Keep source records separate from new GRPO output tree
    cfg = config(seed)
    cfg.output_root = str(tmp_path)
    tag = f"006_COLAB_FULL_B05_sft_grpo_seed{seed}"
    parent = tmp_path / "checkpoints" / tag
    adapter, merged = parent / "sft", parent / "sft_merged"
    adapter.mkdir(parents=True)
    merged.mkdir()
    (parent / "grpo").mkdir()
    result = tmp_path / "results" / tag
    result.mkdir(parents=True)
    old = load_config(worker.CODE / "configs" / f"006_colab_full_b05_seed{seed}.yaml")
    manifest = {"run_tag": tag, "experiment": "006", "seed": seed, "model_id": worker.MODEL_ID,
                "condition": "SFT_GRPO", "scale": "B05", "env": "COLAB", "mode": "FULL",
                "config": {name: asdict(getattr(old, name)) for name in ("train", "grpo", "lora", "reward")}}
    core = {name: manifest[name] for name in ("experiment", "env", "mode", "scale", "condition")}
    core.update(manifest["config"])
    for name in ADAPTER_FILES:
        (adapter / name).write_text("{}")
    (adapter / "adapter_base.json").write_text(json.dumps({"kind": "hf_id", "value": worker.MODEL_ID}))
    (adapter / "config_hash.txt").write_text(worker.digest(core))
    (parent / "grpo/adapter_base.json").write_text(json.dumps({
        "kind": "local", "value": str(merged.relative_to(tmp_path))}))
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors", "chat_template.jinja"):
        (merged / name).write_text("{}")
    (result / "manifest.json").write_text(json.dumps(manifest))
    return cfg, parent, result


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_source_reuses_seed_matched_sft_without_remerging(tmp_path, seed):
    cfg, parent, result = parent_fixture(tmp_path, seed)
    source = worker.source_for(cfg)
    assert source["path"] == str(parent / "sft_merged")
    assert source["source_run"].endswith(f"seed{seed}")
    assert source["binding_level"] == "path_and_provenance" and source["byte_reconstruction"] is False
    assert "chat_template.jinja" in source["merged_sha256"]
    assert source["records"][str((result / "manifest.json").relative_to(tmp_path))]
    (parent / "sft_merged/model.safetensors").write_text("changed")
    assert worker.source_for(cfg) != source


@pytest.mark.parametrize("damage", ("seed", "model", "route", "settings", "hash", "grpo_base", "missing", "symlink"))
def test_wrong_or_missing_parent_is_rejected(tmp_path, damage):
    # Matching model size is insufficient if the seed or starting route is different
    cfg, parent, result = parent_fixture(tmp_path)
    manifest_path = result / "manifest.json"
    manifest = read_json(manifest_path)
    if damage in ("seed", "model", "route", "settings"):
        if damage == "settings":
            manifest["config"]["train"]["sft_lr"] = 0.1
        else:
            key = {"seed": "seed", "model": "model_id", "route": "condition"}[damage]
            manifest[key] = "wrong"
        manifest_path.write_text(json.dumps(manifest))
    elif damage == "hash":
        (parent / "sft/config_hash.txt").write_text("wrong")
    elif damage == "grpo_base":
        (parent / "grpo/adapter_base.json").write_text(json.dumps({"kind": "local", "value": "models/050_qwen_b05"}))
    elif damage == "missing":
        (parent / "sft_merged/model.safetensors").unlink()
    else:
        merged = parent / "sft_merged"
        merged.rename(parent / "other")
        merged.symlink_to(parent / "other", target_is_directory=True)
    with pytest.raises(IntegrityError):
        worker.source_for(cfg)


def test_shared_helpers_and_historical_subject_order_are_reused():
    # Preserve seed 0's earlier flattened subject order without changing tier labels.
    from scripts import s50_b1_grpo as original
    for name in ("bind_run", "checkpoint_step", "comparator_order", "evaluate", "paths_for",
                 "run_lock", "stopped_outcome", "verify_record"):
        assert getattr(worker, name) is getattr(original, name)
    for seed in (0, 1, 2):
        path = worker.CODE / "results" / f"006_COLAB_FULL_B05_sft_grpo_seed{seed}" / "grpo_preds.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        subjects = list(dict.fromkeys(row["subject"] for row in rows))
        ordered = worker.comparator_order(config(seed), [{"subject": s} for s in sorted(subjects)])
        assert [row["subject"] for row in ordered] == subjects


def test_new_paths_cannot_redirect_to_historical_outputs(tmp_path):
    cfg = config()
    cfg.output_root = str(tmp_path)
    root, result, checkpoint = worker.paths_for(cfg)
    assert result == root / "results/052_COLAB_FULL_B05_sft_grpo_seed0"
    assert checkpoint == root / "checkpoints/052_COLAB_FULL_B05_sft_grpo_seed0/grpo"
    result.parent.mkdir()
    old = result.parent / "006_COLAB_FULL_B05_sft_grpo_seed0"
    old.mkdir()
    result.symlink_to(old, target_is_directory=True)
    with pytest.raises(IntegrityError, match="redirected"):
        worker.paths_for(cfg)


def test_binding_contains_new_worker_and_source(monkeypatch):
    import obs
    monkeypatch.setattr(worker, "source_for", lambda cfg: {"kind": "gsm8k_sft", "path": "/sft"})
    monkeypatch.setattr(obs, "versions", lambda: dict(worker.PINS))
    binding = worker.run_binding(config(), [{}] * 245, [{}] * 1884)
    assert binding["historical_comparator"] == "006_seed0" and binding["parser"] == "B1"
    assert "scripts/s52_b1_warm_grpo.py" in binding["code"]
    assert "scripts/s50_b1_grpo.py" in binding["code"]
    with pytest.raises(IntegrityError, match="245"):
        worker.run_binding(config(), [], [{}] * 1884)


@pytest.mark.parametrize("kind", ("complete", "a1", "technical", "wrong_base"))
def test_training_uses_only_bound_sft_and_commits_expected_outcome(tmp_path, monkeypatch, kind):
    # Technical failures must not become registered stops or evaluated endpoints.
    import obs
    import train.grpo as grpo
    import train.sft as sft
    import train.parser_diagnostics as tracing
    import scripts.s42_worker as records
    cfg = config()
    cfg.output_root = str(tmp_path)
    _, result, checkpoint = worker.paths_for(cfg)
    result.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    binding = {"source": {"path": str(tmp_path / "old_sft"), "base_value": "old_sft"}}
    events, diagnostics = [], {}
    diagnostic = SimpleNamespace(summary=lambda: diagnostics, record={"steps": []},
                                 close_attempt=lambda: events.append("close"))
    monkeypatch.setattr(tracing, "attach_parser_diagnostics", lambda *a: events.append("B1") or diagnostic)
    monkeypatch.setattr(records, "capture_wandb_run", lambda *a, **k: events.append(k["package"]))
    monkeypatch.setattr(records, "wandb_records", lambda *a, **k: [])
    monkeypatch.setattr(obs, "resolve_report_to", lambda cfg: ["wandb"])
    monkeypatch.setattr(obs, "write_manifest", lambda cfg, out, extra: (Path(out) / "manifest.json").write_text("{}"))
    monkeypatch.setattr(obs, "gpu_snapshot", lambda: {})
    monkeypatch.setattr(sft, "run_sft", lambda *a, **k: pytest.fail("SFT must not run"))

    def run(passed_cfg, device, rows, model_src, base_kind, base_value, **kwargs):
        assert passed_cfg is cfg and device == "cuda" and model_src == binding["source"]["path"]
        assert base_kind == "local" and base_value == "old_sft"
        assert kwargs["out_dir"] == str(checkpoint) and kwargs["allow_oom_fallback"] is False
        callbacks = []
        kwargs["trainer_setup"](SimpleNamespace(add_callback=callbacks.append))
        for callback in callbacks:
            callback.on_log(None, SimpleNamespace(global_step=1), None, logs={"loss": 0.1})
        steps = 50 if kind == "a1" else 500
        diagnostics.update(completed_updates=steps, responses_checked=steps * 8)
        diagnostic.record["steps"] = [{"training_log": {"completions/clipped_ratio": .5}} for _ in range(steps)]
        (result / "parser_diagnostics.json").write_text(json.dumps(diagnostic.record))
        if kind == "a1":
            raise RuntimeError("sentinel abort: a1: over 30%")
        if kind == "technical":
            raise RuntimeError("CUDA out of memory")
        for name in ADAPTER_FILES:
            (checkpoint / name).write_text("{}")
        (checkpoint / "adapter_base.json").write_text(json.dumps({
            "kind": "local", "value": "wrong" if kind == "wrong_base" else base_value}))
        (result / "grpo_trainlog.json").write_text("{}")

    monkeypatch.setattr(grpo, "run_grpo_rows", run)
    if kind in ("technical", "wrong_base"):
        with pytest.raises((RuntimeError, IntegrityError)):
            worker.train(cfg, [{"prompt": "development"}], result, checkpoint, binding)
        assert (result / "last_failure.json").exists()
    else:
        status = worker.train(cfg, [{"prompt": "development"}], result, checkpoint, binding)
        assert status["completed"] is (kind == "complete")
        assert (result / "terminal_stop.json").exists() is (kind == "a1")
        monkeypatch.setattr(grpo, "run_grpo_rows", lambda *a, **k: pytest.fail("Do not repeat a terminal run"))
        assert worker.train(cfg, [], result, checkpoint, binding) == status
    assert events == ["B1", "052", "close"]
    assert (result / "training_complete.json").exists() is (kind == "complete")


def test_notebook_uses_warm_worker_at_launch_or_closeout():
    path = worker.CODE / "notebooks/052_colab_0p5b_b1_warm_grpo.ipynb"
    cells = json.loads(path.read_text())["cells"]
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    values, executed = {}, False
    fields = {"SEEDS_TO_RUN", "cu_before", "cu_after", "unit_rate"}
    for cell in cells:
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
    assert "scripts/s52_b1_warm_grpo.py" in code
    assert "scripts/run.py" not in code and "snapshot_download" not in code
    assert "merge_adapter" not in code and "run_sft" not in code


def test_generic_runner_refuses_052_before_creating_outputs():
    # The ordinary runner otherwise would select historical rewards and SFT
    text = (worker.CODE / "scripts/run.py").read_text()
    gate = text.index('if cfg.experiment == "052":')
    assert gate < text.index("tag = run_tag(cfg)", gate)
    assert 'raise ValueError("experiment 052 requires scripts/s52_b1_warm_grpo.py")' in text
