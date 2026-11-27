import json
import os
from types import SimpleNamespace

import pytest
import yaml

from checkpoint import config_hash
from config import load_config
from obs import A2_VERDICT_FILE, a2_verdict_payload, make_grpo_sentinel
from scripts.s12_eval import ADAPTER_FILES, IntegrityError, atomic_write_json
from scripts.s31_sentinel import (NON_REPRODUCING, commit_non_reproducing,
                                  non_reproducing_state, require_true_verdict,
                                  reuse_parent_gate, stage_checkpoint,
                                  verify_staged_checkpoint,
                                  verdict_state)


class FakeTrainerCallback:
    pass


# obs imports transformers for the callback base, stub just enough to run offline
@pytest.fixture(autouse=True)
def fake_transformers(monkeypatch):
    import sys
    import types

    module = types.ModuleType("transformers")
    module.TrainerCallback = FakeTrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", module)


def history(n, early=0.70, mid=0.69, mid_kl=0.001):
    # this shape is the a2 condition: improvement under 0.02 and mid kl under 0.005,
    # so mid_kl 0.001 reproduces the abort and 0.01 does not
    rows = []
    for step in range(1, n + 1):
        acc = early if step <= 50 else mid
        kl = mid_kl if step > 100 else 0.01
        rows.append({"step": step, "rewards/accuracy_reward/mean": acc,
                     "kl": kl, "learning_rate": 1e-5})
    return rows


def source_tree(tmp_path):
    source_cfg = load_config("configs/030_colab_b7_n4_seed1.yaml")
    target_cfg = load_config("configs/031_colab_b7_n4_a2_record.yaml")
    source_root = tmp_path / "030_grpo"
    checkpoint = source_root / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    required = {
        "adapter_model.safetensors": b"weights",
        "adapter_config.json": b"{}",
        "optimizer.pt": b"optimizer",
        "scheduler.pt": b"scheduler",
        "rng_state.pth": b"rng",
        "training_args.bin": b"args",
    }
    for name, raw in required.items():
        (checkpoint / name).write_bytes(raw)
    json.dump({"global_step": 100, "max_steps": 500, "log_history": history(100)},
              open(checkpoint / "trainer_state.json", "w"))
    # staging keeps the weights and swaps the config identity over to 031
    (source_root / "config_hash.txt").write_text(config_hash(source_cfg))
    archived = tmp_path / "grpo_trainlog.json"
    json.dump(history(100), open(archived, "w"))
    return source_cfg, target_cfg, source_root, checkpoint, archived


def test_031_config_matches_030_except_registered_changes():
    old = yaml.safe_load(open("configs/030_colab_b7_n4_seed1.yaml"))
    new = yaml.safe_load(open("configs/031_colab_b7_n4_a2_record.yaml"))
    assert old.pop("experiment") == "030"
    assert new.pop("experiment") == "031"
    assert new["train"].pop("a2_action") == "record"
    assert old == new


def test_030_hash_stays_at_its_archived_value():
    # 031 chains from the 030 archive, so this hash is the anchor for that history
    cfg = load_config("configs/030_colab_b7_n4_seed1.yaml")
    assert config_hash(cfg) == "9e7284c8d7d1856d3a25f26919c216f0208f3e962a9159fc5b094cb0d2c14729"


def test_checkpoint_staging_roundtrip(tmp_path):
    source_cfg, target_cfg, source_root, checkpoint, archived = source_tree(tmp_path)
    out_dir = tmp_path / "031_grpo"
    record = stage_checkpoint(
        str(source_root), str(checkpoint), str(out_dir), source_cfg, target_cfg,
        str(archived))
    assert record["completed"] is True
    assert (out_dir / "config_hash.txt").read_text() == config_hash(target_cfg)
    assert verify_staged_checkpoint(
        str(source_root), str(checkpoint), str(out_dir), source_cfg, target_cfg,
        str(archived)) == record
    assert not list(out_dir.rglob("*.part"))


def test_checkpoint_staging_rejects_changed_history(tmp_path):
    source_cfg, target_cfg, source_root, checkpoint, archived = source_tree(tmp_path)
    rows = history(100)
    rows[20]["kl"] = 99
    json.dump(rows, open(archived, "w"))
    with pytest.raises(IntegrityError):
        stage_checkpoint(
            str(source_root), str(checkpoint), str(tmp_path / "031_grpo"),
            source_cfg, target_cfg, str(archived))


def test_checkpoint_staging_requires_the_full_trainer_state(tmp_path):
    source_cfg, target_cfg, source_root, checkpoint, archived = source_tree(tmp_path)
    os.remove(checkpoint / "optimizer.pt")
    with pytest.raises(IntegrityError):
        stage_checkpoint(
            str(source_root), str(checkpoint), str(tmp_path / "031_grpo"),
            source_cfg, target_cfg, str(archived))


def test_record_callback_restores_history_and_continues_on_true(tmp_path):
    # record mode: the verdict says it would fire and training carries on anyway
    cfg = load_config("configs/031_colab_b7_n4_a2_record.yaml")
    callback = make_grpo_sentinel(cfg, out_dir=str(tmp_path),
                                  prior_history=history(100))
    control = SimpleNamespace(should_training_stop=False)
    for step in range(101, 151):
        state = SimpleNamespace(global_step=step)
        callback.on_log(None, state, control, logs={
            "rewards/accuracy_reward/mean": 0.69, "kl": 0.001})
    assert control.should_training_stop is False
    record = json.load(open(tmp_path / A2_VERDICT_FILE))
    assert record["would_fire"] is True
    assert not (tmp_path / (A2_VERDICT_FILE + ".part")).exists()


def test_record_callback_stops_normally_on_false(tmp_path):
    # a stop that doesnt reproduce ends the run, nothing is left to continue for
    cfg = load_config("configs/031_colab_b7_n4_a2_record.yaml")
    callback = make_grpo_sentinel(cfg, out_dir=str(tmp_path),
                                  prior_history=history(100))
    control = SimpleNamespace(should_training_stop=False)
    for step in range(101, 151):
        state = SimpleNamespace(global_step=step)
        callback.on_log(None, state, control, logs={
            "rewards/accuracy_reward/mean": 0.69, "kl": 0.01})
    assert control.should_training_stop is True
    assert json.load(open(tmp_path / A2_VERDICT_FILE))["would_fire"] is False
    with pytest.raises(RuntimeError):
        callback.on_log(
            None, SimpleNamespace(global_step=151),
            SimpleNamespace(should_training_stop=False),
            logs={"rewards/accuracy_reward/mean": 0.69, "kl": 0.01})


def test_record_callback_cannot_pass_150_without_the_verdict(tmp_path):
    cfg = load_config("configs/031_colab_b7_n4_a2_record.yaml")
    callback = make_grpo_sentinel(cfg, out_dir=str(tmp_path),
                                  prior_history=history(100))
    control = SimpleNamespace(should_training_stop=False)
    for step in range(101, 151):
        state = SimpleNamespace(global_step=step)
        callback.on_log(None, state, control, logs={
            "rewards/accuracy_reward/mean": 0.69, "kl": 0.001})
    os.remove(tmp_path / A2_VERDICT_FILE)
    with pytest.raises(RuntimeError):
        callback.on_log(None, SimpleNamespace(global_step=151), control, logs={
            "rewards/accuracy_reward/mean": 0.69, "kl": 0.001})


def test_record_callback_reuses_a_true_verdict_after_resume(tmp_path):
    # a committed verdict survives a resume rather than being recomputed
    cfg = load_config("configs/031_colab_b7_n4_a2_record.yaml")
    rows = history(150)
    atomic_write_json(str(tmp_path / A2_VERDICT_FILE),
                      a2_verdict_payload(cfg, rows))
    callback = make_grpo_sentinel(cfg, out_dir=str(tmp_path), prior_history=rows)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_log(None, SimpleNamespace(global_step=151), control, logs={
        "rewards/accuracy_reward/mean": 0.69, "kl": 0.001})
    assert control.should_training_stop is False


def test_record_callback_cannot_start_without_the_prefix(tmp_path):
    # the 100 step prefix has to be complete and unique, the a2 window spans it
    cfg = load_config("configs/031_colab_b7_n4_a2_record.yaml")
    with pytest.raises(RuntimeError):
        make_grpo_sentinel(cfg, out_dir=str(tmp_path), prior_history=[])


def test_record_callback_rejects_missing_prefix_steps(tmp_path):
    cfg = load_config("configs/031_colab_b7_n4_a2_record.yaml")
    broken = history(100)
    del broken[20]
    with pytest.raises(RuntimeError):
        make_grpo_sentinel(cfg, out_dir=str(tmp_path), prior_history=broken)


def test_record_callback_rejects_duplicate_prefix_steps(tmp_path):
    cfg = load_config("configs/031_colab_b7_n4_a2_record.yaml")
    broken = history(100)
    broken.append(dict(broken[-1]))
    with pytest.raises(RuntimeError):
        make_grpo_sentinel(cfg, out_dir=str(tmp_path), prior_history=broken)


def prepare_outcome(tmp_path, would_fire):
    cfg = load_config("configs/031_colab_b7_n4_a2_record.yaml")
    out_dir = tmp_path / "grpo"
    rdir = tmp_path / "results"
    checkpoint = out_dir / "checkpoint-150"
    checkpoint.mkdir(parents=True)
    rdir.mkdir()
    rows = history(150, mid_kl=0.001 if would_fire else 0.01)
    json.dump({"global_step": 150, "log_history": rows},
              open(checkpoint / "trainer_state.json", "w"))
    atomic_write_json(str(out_dir / A2_VERDICT_FILE),
                      a2_verdict_payload(cfg, rows))
    for name in ADAPTER_FILES:
        path = out_dir / name
        if not path.exists():
            path.write_text("{}")
    json.dump(rows, open(rdir / "grpo_trainlog.json", "w"))
    return cfg, rdir, out_dir


def test_false_verdict_commits_a_rerun_safe_conclusion(tmp_path):
    cfg, rdir, out_dir = prepare_outcome(tmp_path, would_fire=False)
    assert verdict_state(str(out_dir), cfg) == "committed_false"
    record = commit_non_reproducing(str(rdir), str(out_dir), cfg)
    assert record["final_step"] == 150
    assert non_reproducing_state(str(rdir), str(out_dir), cfg) == "committed"
    before = open(rdir / NON_REPRODUCING, "rb").read()
    assert commit_non_reproducing(str(rdir), str(out_dir), cfg) == record
    assert open(rdir / NON_REPRODUCING, "rb").read() == before
    assert os.path.exists(rdir / NON_REPRODUCING)


def test_false_verdict_cannot_reach_endpoint(tmp_path):
    cfg, _, out_dir = prepare_outcome(tmp_path, would_fire=False)
    with pytest.raises(IntegrityError):
        require_true_verdict(str(out_dir), cfg)


def test_true_verdict_is_required_for_endpoint(tmp_path):
    cfg, _, out_dir = prepare_outcome(tmp_path, would_fire=True)
    assert verdict_state(str(out_dir), cfg) == "committed_true"
    assert require_true_verdict(str(out_dir), cfg)["would_fire"] is True


def test_parent_gate_reuse_is_hash_bound(tmp_path):
    # reuse the 030 gate instead of paying for it again, bound to the same bytes
    source = tmp_path / "source_gate.json"
    target = tmp_path / "target_gate.json"
    preds = tmp_path / "preds.jsonl"
    preds.write_text("{}\n")
    adapters = {"adapter": "hash"}
    atomic_write_json(str(source), {
        "completed": True, "adapter_sha256": adapters,
        "archived_preds_sha256": __import__("scripts.s12_eval", fromlist=["sha256"]).sha256(str(preds)),
        "new_preds_sha256": "gate-output",
        "stats": {"agreement": 1.0},
    })
    record = reuse_parent_gate(str(source), str(target), adapters, str(preds))
    assert record["completed"] is True
    preds.write_text("changed\n")
    with pytest.raises(IntegrityError):
        reuse_parent_gate(str(source), str(target), adapters, str(preds))


def test_031_notebook_carries_the_registered_runner_guards():
    nb = json.load(open("notebooks/031_colab_b7_n4_a2_record.ipynb"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    assert "configs/031_colab_b7_n4_a2_record.yaml" in source
    assert "source_prefix_facts" in source and "stage_checkpoint" in source
    assert "reuse_parent_gate" in source
    assert "scripts/grpo_worker.py" in source
    assert "require_true_verdict(OUT_DIR, cfg, TRAINLOG_PATH)" in source
    assert "commit_non_reproducing" in source
    assert "031_comparison_summary.json" in source
    assert "cu_before =" in source and "cu_after =" in source
    assert "os.environ['WANDB_PROJECT'] = cfg.log.wandb_project" in source
    assert "setdefault('WANDB_PROJECT'" not in source
    # a literal backslash n in a code cell means the notebook json was mangled
    assert "\\\\n" not in "".join(
        "".join(cell.get("source", [])) for cell in nb["cells"]
        if cell["cell_type"] == "code")
