import base64
import hashlib
import json
import os
from dataclasses import asdict

import pytest

from checkpoint import config_hash
from config import load_config
from scripts.s12_eval import (ARMS, GateFailure, IntegrityError, archive_orphans,
                              complete_summary_nulls, config_hash_status, identity_check,
                              is_committed, paths_for, read_jsonl, recover_pending,
                              run_arm, sha256)

N = 1884
SUBJECTS = ["marketing", "philosophy", "formal_logic"]


def make_rows(pred="A", correct=True):
    return [{"subject": SUBJECTS[i % 3], "gold": "A", "pred": pred,
             "correct": bool(correct)} for i in range(N)]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def arm(tmp_path):
    # use an archived probe arm directory and a separate scratch directory
    rdir = tmp_path / "results" / "007_p2_TEST_seed1"
    sdir = tmp_path / "scratch"
    rdir.mkdir(parents=True)
    p = paths_for(str(rdir))
    write_jsonl(p["probe_preds"], make_rows())
    json.dump({"per_tier_acc": {"easy": 0.5}}, open(p["probe_metrics"], "w"))
    json.dump({"sft_acc": None, "sft_logprob_acc": None, "seed": 1},
              open(p["summary"], "w"))
    return str(rdir), str(sdir)


def fake_evals(sdir, gate_rows=None, gen_rows=None):
    # gate and gen writers that mimic the harness output files
    def gate_eval():
        os.makedirs(sdir, exist_ok=True)
        write_jsonl(os.path.join(sdir, "gate_logprob_preds.jsonl"),
                    gate_rows if gate_rows is not None else make_rows())

    def gen_eval():
        os.makedirs(sdir, exist_ok=True)
        write_jsonl(os.path.join(sdir, "sft_preds.jsonl"),
                    gen_rows if gen_rows is not None else make_rows())
        return {"per_tier_acc": {"easy": 1.0}, "n_items": N}

    return gen_eval, gate_eval


def test_success_commits_and_completes_summary(arm):
    rdir, sdir = arm
    gen_eval, gate_eval = fake_evals(sdir)
    assert run_arm(rdir, sdir, gen_eval, gate_eval, log=lambda *a: None) == "committed"
    p = paths_for(rdir)
    assert is_committed(rdir)
    assert not os.path.exists(p["pending"])
    man = json.load(open(p["manifest"]))
    assert man["sha256_preds"] == sha256(p["preds"])
    sm = json.load(open(p["summary"]))
    assert sm["sft_acc"] == {"easy": 1.0}
    assert sm["sft_logprob_acc"] == {"easy": 0.5}
    assert man["summary_fields_completed"] == ["sft_acc", "sft_logprob_acc"]


def test_committed_arm_skips_and_clears_stale_pending(arm):
    rdir, sdir = arm
    gen_eval, gate_eval = fake_evals(sdir)
    run_arm(rdir, sdir, gen_eval, gate_eval, log=lambda *a: None)
    p = paths_for(rdir)
    json.dump({"summary_raw": "{}", "summary_sha_before": "x"}, open(p["pending"], "w"))
    calls = []
    gen_count = lambda: calls.append(1) or gen_eval()
    assert run_arm(rdir, sdir, gen_count, gate_eval, log=lambda *a: None) == "skipped"
    assert not calls
    assert not os.path.exists(p["pending"])


def test_identity_failure_raises_and_writes_nothing(arm):
    rdir, sdir = arm
    gen_eval, gate_eval = fake_evals(sdir, gate_rows=make_rows(pred="B", correct=False))
    with pytest.raises(GateFailure):
        run_arm(rdir, sdir, gen_eval, gate_eval, log=lambda *a: None)
    p = paths_for(rdir)
    assert not os.path.exists(p["preds"])
    assert not os.path.exists(p["manifest"])
    assert json.load(open(p["summary"]))["sft_acc"] is None


def test_interrupted_after_summary_recovers_from_pending(arm):
    rdir, sdir = arm
    gen_eval, gate_eval = fake_evals(sdir)
    p = paths_for(rdir)

    # simulate a crash after the summary changed but before the manifest wrote
    original_raw = open(p["summary"], "rb").read()
    json.dump({"ts": "x", "summary_b64": base64.b64encode(original_raw).decode("ascii"),
               "summary_sha_before": sha256(p["summary"])}, open(p["pending"], "w"))
    json.dump({"sft_acc": {"easy": 0.9}, "sft_logprob_acc": {"easy": 0.5}, "seed": 1},
              open(p["summary"], "w"))
    write_jsonl(p["preds"], make_rows()[:10])

    assert run_arm(rdir, sdir, gen_eval, gate_eval, log=lambda *a: None) == "committed"
    sm = json.load(open(p["summary"]))
    assert sm["sft_acc"] == {"easy": 1.0}
    orphan_dirs = [d for d in os.listdir(rdir) if d.startswith("orphaned_")]
    assert len(orphan_dirs) == 1
    assert len(read_jsonl(p["preds"])) == N


def test_tampered_commit_halts_and_touches_nothing(arm):
    # a complete manifest whose hashes no longer match is an integrity alarm,
    # never a resume case
    rdir, sdir = arm
    gen_eval, gate_eval = fake_evals(sdir)
    run_arm(rdir, sdir, gen_eval, gate_eval, log=lambda *a: None)
    p = paths_for(rdir)
    with open(p["preds"], "a") as f:
        f.write(json.dumps({"subject": "marketing", "gold": "A", "pred": "A",
                            "correct": True}) + "\n")
    assert not is_committed(rdir)
    tampered_sha = sha256(p["preds"])
    with pytest.raises(IntegrityError):
        run_arm(rdir, sdir, gen_eval, gate_eval, log=lambda *a: None)
    assert not any(d.startswith("orphaned_") for d in os.listdir(rdir))
    assert sha256(p["preds"]) == tampered_sha
    assert os.path.exists(p["manifest"])


def test_tampered_summary_halts_and_touches_nothing(arm):
    rdir, sdir = arm
    gen_eval, gate_eval = fake_evals(sdir)
    run_arm(rdir, sdir, gen_eval, gate_eval, log=lambda *a: None)
    p = paths_for(rdir)
    with open(p["summary"], "a") as f:
        f.write("\n")
    tampered = open(p["summary"], "rb").read()
    with pytest.raises(IntegrityError):
        run_arm(rdir, sdir, gen_eval, gate_eval, log=lambda *a: None)
    assert open(p["summary"], "rb").read() == tampered
    assert not any(d.startswith("orphaned_") for d in os.listdir(rdir))


def test_short_generative_output_aborts_before_commit(arm):
    rdir, sdir = arm
    gen_eval, gate_eval = fake_evals(sdir, gen_rows=make_rows()[:100])
    with pytest.raises(AssertionError):
        run_arm(rdir, sdir, gen_eval, gate_eval, log=lambda *a: None)
    assert not os.path.exists(paths_for(rdir)["preds"])


def test_summary_nulls_only(tmp_path):
    sp = tmp_path / "summary.json"
    json.dump({"sft_acc": {"easy": 0.7}, "sft_logprob_acc": None}, open(sp, "w"))
    changed = complete_summary_nulls(str(sp), {"easy": 0.9}, {"easy": 0.4})
    assert changed == ["sft_logprob_acc"]
    sm = json.load(open(sp))
    assert sm["sft_acc"] == {"easy": 0.7}
    assert sm["sft_logprob_acc"] == {"easy": 0.4}


def test_identity_check_thresholds():
    rows = make_rows()
    assert identity_check(rows, rows)["agreement"] == 1.0
    off = [dict(r) for r in rows]
    for r in off[:30]:
        r["pred"] = "B"
    with pytest.raises(GateFailure):
        identity_check(off, rows)


def test_phase_tables_cover_exactly_the_twelve_arms():
    assert len(ARMS["40"]) == 8 and len(ARMS["80"]) == 4
    assert all(s in (1, 2) for _, s in ARMS["40"] + ARMS["80"])
    assert all("b05" in c or "b3" in c for c, _ in ARMS["40"])
    assert all("b15" in c for c, _ in ARMS["80"])
    assert len({(c, s) for c, s in ARMS["40"] + ARMS["80"]}) == 12


def _era_parent(cfg, omitted):
    # a manifest as an older run would have written it, without the given fields
    core = {
        "experiment": cfg.experiment,
        "env": cfg.env.name, "mode": cfg.mode.name, "scale": cfg.scale.name,
        "condition": cfg.condition.name,
        "train": asdict(cfg.train), "grpo": asdict(cfg.grpo),
        "lora": asdict(cfg.lora), "reward": asdict(cfg.reward),
    }
    if core["train"].get("a2_action") == "abort":
        core["train"].pop("a2_action")
    if core["train"].get("grpo_parameter_mode") == "lora":
        core["train"].pop("grpo_parameter_mode")
    if core["train"].get("grpo_precision_mode") == "legacy":
        core["train"].pop("grpo_precision_mode")
    for f in omitted:
        core["train"].pop(f)
    parent = {
        "experiment": core["experiment"], "env": core["env"],
        "mode": core["mode"], "scale": core["scale"],
        "condition": core["condition"],
        "config": {k: core[k] for k in ("train", "grpo", "lora", "reward")},
    }
    legacy_hash = hashlib.sha256(
        json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()
    return parent, legacy_hash


def test_hash_matrix_allows_exactly_the_era_omissions():
    cfg = load_config("configs/007_p2_colab_b05.yaml")
    cfg.train.seed = 1

    assert config_hash_status(cfg, {}, config_hash(cfg)) == "current"

    # 014 era: only grpo_from_adapter missing from the recorded schema
    parent14, hash14 = _era_parent(cfg, ("grpo_from_adapter",))
    assert config_hash_status(cfg, parent14, hash14) == "legacy_missing_grpo_from_adapter"

    # 007 era: both chain fields missing
    parent07, hash07 = _era_parent(cfg, ("grpo_from_adapter", "sft_from_adapter"))
    assert config_hash_status(cfg, parent07, hash07) == "legacy_missing_both_chain_fields"

    # crossed eras never match
    assert config_hash_status(cfg, parent07, hash14) == "mismatch"
    assert config_hash_status(cfg, parent14, hash07) == "mismatch"

    # a real config difference fails in every era
    changed = load_config("configs/007_p2_colab_b05.yaml")
    changed.train.seed = 1
    changed.train.sft_lr *= 2
    assert config_hash_status(changed, parent07, hash07) == "mismatch"
    assert config_hash_status(changed, parent14, hash14) == "mismatch"

    # non-neutral chain fields disable the allowance entirely
    chained = load_config("configs/007_p2_colab_b05.yaml")
    chained.train.seed = 1
    chained.train.sft_from_adapter = "some_adapter"
    assert config_hash_status(chained, parent07, hash07) == "mismatch"
    composed = load_config("configs/007_p2_colab_b05.yaml")
    composed.train.seed = 1
    composed.train.grpo_from_adapter = "some_adapter"
    assert config_hash_status(composed, parent14, hash14) == "mismatch"
    assert config_hash_status(composed, parent07, hash07) == "mismatch"


def test_orphan_archive_moves_not_deletes(arm):
    rdir, _ = arm
    p = paths_for(rdir)
    write_jsonl(p["preds"], make_rows()[:5])
    json.dump({"x": 1}, open(p["eval"], "w"))
    moved = archive_orphans(rdir)
    assert set(moved) == {"preds", "eval"}
    odir = [d for d in os.listdir(rdir) if d.startswith("orphaned_")][0]
    assert os.path.exists(os.path.join(rdir, odir, "sft_preds.jsonl"))


def test_recover_pending_restores_exact_bytes(arm):
    rdir, _ = arm
    p = paths_for(rdir)
    raw = b'{\r\n  "note": "\xc3\xa9",\r\n  "sft_acc": null\r\n}'
    with open(p["summary"], "wb") as f:
        f.write(raw)
    json.dump({"summary_b64": base64.b64encode(raw).decode("ascii"),
               "summary_sha_before": sha256(p["summary"])},
              open(p["pending"], "w"))
    json.dump({"sft_acc": {"easy": 0.1}}, open(p["summary"], "w"))
    assert recover_pending(rdir)
    assert open(p["summary"], "rb").read() == raw


def test_recover_pending_bad_backup_halts(arm):
    rdir, _ = arm
    p = paths_for(rdir)
    before = open(p["summary"], "rb").read()
    bad = base64.b64encode(b'{"tampered": true}').decode("ascii")
    json.dump({"summary_b64": bad, "summary_sha_before": "0" * 64},
              open(p["pending"], "w"))
    with pytest.raises(IntegrityError):
        recover_pending(rdir)
    assert open(p["summary"], "rb").read() == before


def test_billing_fires_once_per_attempt(arm):
    rdir, sdir = arm
    calls = []
    bill = lambda timings, done: calls.append((sorted(timings), done))

    gen_eval, gate_eval = fake_evals(sdir)
    run_arm(rdir, sdir, gen_eval, gate_eval, log=lambda *a: None, bill=bill)
    assert calls == [(["commit", "gate_logprob", "sft_eval"], True)]

    # a crash inside the generative stage still bills the attempt exactly once
    calls.clear()
    p = paths_for(rdir)
    os.remove(p["manifest"])

    def broken_gen():
        raise RuntimeError("session died")

    with pytest.raises(RuntimeError):
        run_arm(rdir, sdir, broken_gen, gate_eval, log=lambda *a: None, bill=bill)
    assert len(calls) == 1
    assert calls[0][1] is False and "gate_logprob" in calls[0][0]


def test_gate_failure_bills_once(arm):
    rdir, sdir = arm
    calls = []
    gen_eval, gate_eval = fake_evals(sdir, gate_rows=make_rows(pred="B", correct=False))
    with pytest.raises(GateFailure):
        run_arm(rdir, sdir, gen_eval, gate_eval, log=lambda *a: None,
                bill=lambda t, d: calls.append((t, d)))
    assert len(calls) == 1 and calls[0][1] is False
