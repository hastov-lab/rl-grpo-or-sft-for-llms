import ast
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import load_config
from file_utils import read_json
from scripts import s50_b1_grpo as worker
from scripts.s12_eval import IntegrityError, atomic_write_json, sha256


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_configs_change_only_identity_and_fallback(seed):
    cfg = load_config(worker.CODE / "configs" / f"050_colab_b05_b1_seed{seed}.yaml")
    worker.validate_config(cfg)
    assert cfg.experiment == "050" and cfg.exploratory
    assert cfg.exec.allow_oom_fallback is False
    cfg.grpo.beta += 0.01
    with pytest.raises(IntegrityError, match="beyond"):
        worker.validate_config(cfg)


def test_isolation_and_unbound_existing_files(tmp_path):
    # A renamed notebook must not inherit an old adapter or completion marker
    cfg = load_config(worker.CODE / "configs/050_colab_b05_b1_seed0.yaml")
    cfg.output_root = str(tmp_path)
    _, result, checkpoint = worker.paths_for(cfg)
    assert "050_COLAB_FULL_B05_base_grpo_seed0" in str(result)
    assert "005_" not in str(result)
    with worker.run_lock(result):
        worker.bind_run(result, checkpoint, {"parser": "B1"})
        worker.bind_run(result, checkpoint, {"parser": "B1"})
        with pytest.raises(IntegrityError, match="changed"):
            worker.bind_run(result, checkpoint, {"parser": "H"})
    (result / "run_binding.json").unlink()
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"old")
    with worker.run_lock(result), pytest.raises(IntegrityError, match="unbound"):
        worker.bind_run(result, checkpoint, {"parser": "B1"})


def test_output_symlink_cannot_target_historical_results(tmp_path):
    cfg = load_config(worker.CODE / "configs/050_colab_b05_b1_seed0.yaml")
    cfg.output_root = str(tmp_path)
    old = tmp_path / "historical"
    old.mkdir()
    (tmp_path / "results").symlink_to(old, target_is_directory=True)
    with pytest.raises(IntegrityError, match="redirected"):
        worker.paths_for(cfg)


def test_committed_artifacts_are_verified_not_just_skipped(tmp_path):
    # a completion marker still has to agree with the bytes actually saved
    result, cp = tmp_path / "result", tmp_path / "cp"
    result.mkdir()
    (result / "metric.json").write_text("{}")
    binding = {"parser": "B1"}
    path = result / "done.json"
    record = {"binding_sha256": worker.digest(binding),
              "files": {"results/metric.json": sha256(str(result / "metric.json"))}}
    atomic_write_json(str(path), record)
    assert worker.verify_record(path, result, cp, binding) == record
    (result / "metric.json").write_text('{"changed": true}')
    with pytest.raises(IntegrityError, match="changed"):
        worker.verify_record(path, result, cp, binding)


def test_incomplete_latest_checkpoint_does_not_fall_back(tmp_path):
    # never read an interrupted newer save as permission to replay an older one
    (tmp_path / "checkpoint-50").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    with pytest.raises(IntegrityError, match="incomplete"):
        worker.checkpoint_step(tmp_path)


@pytest.mark.parametrize("step,kind,clipped,acc,kl", [
    (50, "A1", 0.5, 0.1, 0.1), (150, "A2", 0.0, 0.1, 0.0)])
def test_stop_record_requires_matching_logged_evidence(step, kind, clipped, acc, kl):
    cfg = load_config(worker.CODE / "configs/050_colab_b05_b1_seed0.yaml")
    diagnostic = SimpleNamespace(record={"steps": [
        {"training_log": {"completions/clipped_ratio": clipped,
                           "rewards/accuracy_reward/mean": acc, "kl": kl}}
        for _ in range(step)]})
    exc = RuntimeError(f"sentinel abort: {kind.lower()}: example")
    assert worker.stopped_outcome(exc, cfg, diagnostic)["kind"] == kind
    assert worker.stopped_outcome(RuntimeError("CUDA out of memory"), cfg, diagnostic) is None
    diagnostic.record["steps"].pop()
    assert worker.stopped_outcome(exc, cfg, diagnostic) is None


def test_stopped_arm_cannot_be_evaluated(tmp_path, monkeypatch):
    import eval.harness as harness
    result, cp = tmp_path / "result", tmp_path / "cp"
    result.mkdir()
    binding = {"parser": "B1"}
    (result / "diagnostic.json").write_text("{}")
    record = {"binding_sha256": worker.digest(binding), "completed": False, "outcome": {"kind": "A1", "step": 50},
              "files": {"results/diagnostic.json": sha256(str(result / "diagnostic.json"))}}
    atomic_write_json(str(result / "terminal_stop.json"), record)
    monkeypatch.setattr(harness, "run_eval_rows", lambda *a, **k: pytest.fail("must not evaluate"))
    assert worker.evaluate(None, [], result, cp, binding)["evaluation"] == "not applicable"
    (result / "grpo_preds.jsonl").write_text("")
    with pytest.raises(IntegrityError, match="stopped arm"):
        worker.evaluate(None, [], result, cp, binding)


def test_endpoint_transaction_and_historical_scoring_bridge(tmp_path, monkeypatch):
    import eval.harness as harness
    result, cp = tmp_path / "result", tmp_path / "cp"
    result.mkdir()
    binding = {"parser": "B1"}
    (result / "training.json").write_text("{}")
    atomic_write_json(str(result / "training_complete.json"), {
        "binding_sha256": worker.digest(binding), "completed": True,
        "files": {"results/training.json": sha256(str(result / "training.json"))}})
    calls = []
    row = {"gold": "D", "subject": "marketing", "tier": "easy"}

    def generation(cfg, device, rows, **kwargs):
        calls.append("generation")
        assert kwargs["extractor"]("The answer is unspecified.") is None
        prediction = dict(row, output="The answer is unspecified.", pred=None,
                          correct=False, format_ok=False, n_tokens=7)
        (result / "grpo_preds.jsonl").write_text(json.dumps(prediction) + "\n")
        return {"n_items": 1}

    def probe(cfg, device, rows, **kwargs):
        calls.append("probe")
        assert "extractor" not in kwargs
        (result / "grpo_logprob_preds.jsonl").write_text(json.dumps(dict(row, pred="D", correct=True)) + "\n")
        return {"n_items": 1}

    # both real transaction branches, no model generation involved
    monkeypatch.setattr(harness, "run_eval_rows", generation)
    monkeypatch.setattr(harness, "run_logprob_rows", probe)
    worker.evaluate(None, [row], result, cp, binding)
    worker.evaluate(None, [row], result, cp, binding)
    assert calls == ["generation", "probe"]
    assert json.loads((result / "grpo_historical_preds.jsonl").read_text())["pred"] == "D"
    (result / "grpo_logprob.json").write_text("{}")
    with pytest.raises(IntegrityError, match="changed"):
        worker.evaluate(None, [row], result, cp, binding)


def test_partial_test_pass_requires_review(tmp_path, monkeypatch):
    import eval.harness as harness
    result = tmp_path
    binding = {"parser": "B1"}
    (result / "train.json").write_text("{}")
    atomic_write_json(str(result / "training_complete.json"), {
        "binding_sha256": worker.digest(binding), "completed": True,
        "files": {"results/train.json": sha256(str(result / "train.json"))}})
    (result / "generation_pending.json").write_text("{}")
    monkeypatch.setattr(harness, "run_eval_rows", lambda *a, **k: pytest.fail("no repeated test pass"))
    with pytest.raises(IntegrityError, match="test contact"):
        worker.evaluate(None, [], result, result / "cp", binding)


def test_notebook_cells_and_launch_targets():
    path = worker.CODE / "notebooks/050_colab_0p5b_b1_base_grpo.ipynb"
    notebook = read_json(path)
    text = "\n".join("".join(c["source"]) for c in notebook["cells"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert all(output.get("output_type") != "error" for output in cell["outputs"])
    assert "SEEDS_TO_RUN = (0,)" in text
    assert "cu_before =" in text and "cu_after =" in text
    assert "scripts/s50_b1_grpo.py" in text
    assert "scripts/run.py" not in text
    assert "for task in ('train', 'eval')" in text


def test_seed_zero_keeps_original_subject_order_and_within_subject_rows():
    from data.subjects import all_subjects
    cfg = load_config(worker.CODE / "configs/050_colab_b05_b1_seed0.yaml")
    rows = [{"subject": subject, "index": index} for subject in all_subjects() for index in (0, 1)]
    ordered = worker.comparator_order(cfg, rows)
    assert [r["subject"] for r in ordered[::2]] == [
        "high_school_psychology", "marketing", "business_ethics", "high_school_geography",
        "philosophy", "us_foreign_policy", "high_school_mathematics", "formal_logic"]
    assert [r["index"] for r in ordered] == [0, 1] * 8
    cfg.train.seed = 1
    assert worker.comparator_order(cfg, rows) == rows
