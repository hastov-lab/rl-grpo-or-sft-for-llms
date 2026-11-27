import json
import os

import pytest
import yaml

from config import load_config
from enums import Condition
from scripts.s12_eval import ADAPTER_FILES, IntegrityError, sha256
from scripts.s28_compose import (ARMS_028, BASE_EVAL_FILES, BINDING_NAME,
                                 ensure_bound_merge, external_reference, merge_for,
                                 parent_results_dir_for, prepopulate_base_evals,
                                 resolve_source_adapter)

N = 1884


def make_adapter(d, salt="a"):
    os.makedirs(d, exist_ok=True)
    for n in ADAPTER_FILES:
        with open(os.path.join(d, n), "w") as f:
            f.write(salt + n)
    return d


def make_merged(d, salt="m"):
    os.makedirs(d, exist_ok=True)
    names = ("config.json", "model-00001-of-00002.safetensors",
             "model-00002-of-00002.safetensors",
             "tokenizer.json", "tokenizer_config.json", "generation_config.json")
    for n in names:
        with open(os.path.join(d, n), "w") as f:
            f.write(salt + n if n != "config.json" else salt + "cfg")
    index = {"weight_map": {"a.weight": "model-00001-of-00002.safetensors",
                            "b.weight": "model-00002-of-00002.safetensors"}}
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)
    return d


def fake_merge(d, salt="m"):
    def merge():
        make_merged(d, salt)
        return d
    return merge


def test_resolver_joins_output_root_and_verifies(tmp_path):
    adir = make_adapter(tmp_path / "checkpoints" / "tag" / "sft")
    abs_path, hashes = resolve_source_adapter("checkpoints/tag/sft", str(tmp_path))
    assert os.path.isabs(abs_path) and abs_path == str(adir)
    assert set(hashes) == set(ADAPTER_FILES)

    os.remove(os.path.join(adir, "config_hash.txt"))
    with pytest.raises(IntegrityError):
        resolve_source_adapter("checkpoints/tag/sft", str(tmp_path))
    with pytest.raises(IntegrityError):
        resolve_source_adapter("", str(tmp_path))


def test_clean_merge_builds_and_binds(tmp_path):
    adir = make_adapter(tmp_path / "adapter")
    merged = str(tmp_path / "sft_merged")
    grpo_dir = str(tmp_path / "grpo")
    out = ensure_bound_merge(str(adir), merged, grpo_dir, fake_merge(merged),
                             log=lambda *a: None)
    assert out == merged
    bind = json.load(open(os.path.join(merged, BINDING_NAME)))
    assert bind["adapter_sha256"]["adapter_config.json"] == sha256(os.path.join(adir, "adapter_config.json"))

    # a second call reuses without merging again
    calls = []
    def counting_merge():
        calls.append(1)
    ensure_bound_merge(str(adir), merged, grpo_dir, counting_merge, log=lambda *a: None)
    assert not calls


def test_adapter_drift_halts(tmp_path):
    adir = make_adapter(tmp_path / "adapter")
    merged = str(tmp_path / "sft_merged")
    ensure_bound_merge(str(adir), merged, str(tmp_path / "grpo"), fake_merge(merged),
                       log=lambda *a: None)
    with open(os.path.join(adir, "adapter_model.safetensors"), "w") as f:
        f.write("different weights")
    with pytest.raises(IntegrityError):
        ensure_bound_merge(str(adir), merged, str(tmp_path / "grpo"), fake_merge(merged),
                           log=lambda *a: None)


def test_merged_drift_halts(tmp_path):
    adir = make_adapter(tmp_path / "adapter")
    merged = str(tmp_path / "sft_merged")
    ensure_bound_merge(str(adir), merged, str(tmp_path / "grpo"), fake_merge(merged),
                       log=lambda *a: None)
    with open(os.path.join(merged, "model.safetensors"), "a") as f:
        f.write("tail")
    with pytest.raises(IntegrityError):
        ensure_bound_merge(str(adir), merged, str(tmp_path / "grpo"), fake_merge(merged),
                           log=lambda *a: None)


def test_unbound_merge_without_checkpoints_is_archived_and_rebuilt(tmp_path):
    adir = make_adapter(tmp_path / "adapter")
    merged = str(tmp_path / "sft_merged")
    make_merged(merged, salt="stale")
    grpo_dir = str(tmp_path / "grpo")
    ensure_bound_merge(str(adir), merged, grpo_dir, fake_merge(merged, salt="fresh"),
                       log=lambda *a: None)
    archived = [d for d in os.listdir(tmp_path) if d.startswith("sft_merged_unbound_")]
    assert len(archived) == 1
    assert open(os.path.join(merged, "config.json")).read() == "freshcfg"
    assert os.path.exists(os.path.join(merged, BINDING_NAME))


def test_unbound_merge_with_checkpoints_halts(tmp_path):
    adir = make_adapter(tmp_path / "adapter")
    merged = str(tmp_path / "sft_merged")
    make_merged(merged, salt="stale")
    grpo_dir = str(tmp_path / "grpo")
    os.makedirs(os.path.join(grpo_dir, "checkpoint-100"))
    with pytest.raises(IntegrityError):
        ensure_bound_merge(str(adir), merged, grpo_dir, fake_merge(merged),
                           log=lambda *a: None)
    # nothing archived, nothing rebuilt
    assert open(os.path.join(merged, "config.json")).read() == "stalecfg"
    assert not any(d.startswith("sft_merged_unbound_") for d in os.listdir(tmp_path))


def _base_eval_src(tmp_path):
    src = tmp_path / "base_src"
    src.mkdir(parents=True)
    tiers = ["easy", "medium", "hard"]
    rows = [{"subject": "s", "tier": tiers[i % 3], "gold": "A", "pred": "A",
             "correct": i % 2 == 0} for i in range(N)]
    by_tier = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r["correct"])
    acc = {k: sum(v) / len(v) for k, v in by_tier.items()}
    for n in ("base_preds.jsonl", "base_logprob_preds.jsonl"):
        with open(src / n, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    for n in ("base_eval.json", "base_logprob.json"):
        json.dump({"n_items": N, "per_tier_acc": acc}, open(src / n, "w"))
    return src


def test_base_eval_prepopulation_copies_all_four_and_skips(tmp_path):
    src = _base_eval_src(tmp_path)
    dst = str(tmp_path / "base_dst")
    rec = prepopulate_base_evals(str(src), dst, log=lambda *a: None)
    assert set(rec["sha256"]) == set(BASE_EVAL_FILES)
    for n in BASE_EVAL_FILES:
        assert sha256(os.path.join(dst, n)) == sha256(os.path.join(str(src), n))

    # both metric files are required to skip the base evaluations
    assert os.path.exists(os.path.join(dst, "base_eval.json"))
    assert os.path.exists(os.path.join(dst, "base_logprob.json"))

    # a second call verifies the record and copies nothing
    before = {n: os.path.getmtime(os.path.join(dst, n)) for n in BASE_EVAL_FILES}
    prepopulate_base_evals(str(src), dst, log=lambda *a: None)
    after = {n: os.path.getmtime(os.path.join(dst, n)) for n in BASE_EVAL_FILES}
    assert before == after


def test_base_eval_prepopulation_halts_on_bad_source(tmp_path):
    src = _base_eval_src(tmp_path)
    os.remove(src / "base_logprob.json")
    with pytest.raises(IntegrityError):
        prepopulate_base_evals(str(src), str(tmp_path / "dst"), log=lambda *a: None)

    src2 = _base_eval_src(tmp_path / "two")
    json.dump({"n_items": 12}, open(src2 / "base_eval.json", "w"))
    with pytest.raises(IntegrityError):
        prepopulate_base_evals(str(src2), str(tmp_path / "dst2"), log=lambda *a: None)


def test_stage_list_for_the_new_condition():
    from scripts.run import plan_stages
    assert plan_stages(Condition.MCSFT_GRPO) == ["base_eval", "grpo", "grpo_eval"]


def test_summary_references_the_external_parent(tmp_path):
    from scripts.run import build_summary
    cfg = load_config("configs/028_colab_b7_mcsft_grpo_seed1.yaml")
    cfg.output_root = str(tmp_path)
    parent = tmp_path / "results" / "014_p8_COLAB_FULL_B7_sft_only_seed1"
    parent.mkdir(parents=True)
    json.dump({"per_tier_acc": {"easy": 0.8, "hard": 0.5}}, open(parent / "sft_eval.json", "w"))
    json.dump({"per_tier_acc": {"easy": 0.7, "hard": 0.55}}, open(parent / "sft_logprob.json", "w"))

    assert parent_results_dir_for(cfg) == str(parent)
    ref = external_reference(str(parent))
    assert ref["tag"].endswith("seed1")

    grpo_eval = {"per_tier_acc": {"easy": 0.85, "hard": 0.6}}
    grpo_lp = {"per_tier_acc": {"easy": 0.7, "hard": 0.55}}
    base_eval = {"per_tier_acc": {"easy": 0.5, "hard": 0.4}}
    s = build_summary(cfg, base_eval, None, grpo_eval, base_eval, None, grpo_lp)
    assert s["reference"] == "mcsft_parent:014_p8_COLAB_FULL_B7_sft_only_seed1"
    assert s["sft_acc"] == {"easy": 0.8, "hard": 0.5}
    assert abs(s["grpo_delta"]["hard"] - 0.1) < 1e-9
    assert abs(s["grpo_delta"]["easy"] - 0.05) < 1e-9


def _sections(path):
    cfg = yaml.safe_load(open(path))
    return cfg


def test_028_configs_match_their_seed_013_sources_exactly():
    allowed_top = {"experiment", "condition", "exploratory"}
    for s in (0, 1, 2):
        a = _sections(f"configs/028_colab_b7_mcsft_grpo_seed{s}.yaml")
        b = _sections(f"configs/013_colab_full_b7_seed{s}.yaml")
        assert a["experiment"] == "028" and a["condition"] == "MCSFT_GRPO"
        assert a["exploratory"] is True
        a_train = dict(a["train"])
        src = a_train.pop("grpo_from_adapter")
        assert src == f"checkpoints/014_p8_COLAB_FULL_B7_sft_only_seed{s}/sft"
        assert a_train == b["train"]
        for key in set(a) | set(b):
            if key in allowed_top or key == "train":
                continue
            assert a.get(key) == b.get(key), f"seed {s}: {key} differs"


def test_028_configs_differ_only_by_seed_and_source():
    parsed = [_sections(c) for c, _, _ in ARMS_028]
    for i in (1, 2):
        a, b = dict(parsed[0]), dict(parsed[i])
        at, bt = dict(a.pop("train")), dict(b.pop("train"))
        assert a == b
        assert at.pop("seed") == 0 and bt.pop("seed") == i
        assert at.pop("grpo_from_adapter").endswith("seed0/sft")
        assert bt.pop("grpo_from_adapter").endswith(f"seed{i}/sft")
        assert at == bt


def test_arm_table_is_seed_matched():
    assert len(ARMS_028) == 3
    for cfg_path, seed, parent in ARMS_028:
        assert f"seed{seed}" in cfg_path and parent.endswith(f"seed{seed}")


def test_legacy_sft_grpo_merge_keeps_the_original_path(tmp_path):
    # an old unbound merged dir with an interrupted grpo checkpoint must still
    # resume when the run does not chain from an external adapter
    class FakeTrain:
        grpo_from_adapter = ""

    class FakeCfg:
        train = FakeTrain()

    merged = str(tmp_path / "sft_merged")
    make_merged(merged, salt="old")
    grpo_dir = str(tmp_path / "grpo")
    os.makedirs(os.path.join(grpo_dir, "checkpoint-250"))

    calls = []
    out = merge_for(FakeCfg(), str(tmp_path / "adapter"), merged, grpo_dir,
                    merge_fn=lambda: calls.append(1) or merged, log=lambda *a: None)
    assert out == merged and calls == [1]
    assert not os.path.exists(os.path.join(merged, BINDING_NAME))
    assert open(os.path.join(merged, "config.json")).read() == "oldcfg"


def test_composition_merge_goes_through_the_binding(tmp_path):
    class FakeTrain:
        grpo_from_adapter = "checkpoints/x/sft"

    class FakeCfg:
        train = FakeTrain()

    adir = make_adapter(tmp_path / "adapter")
    merged = str(tmp_path / "sft_merged")
    out = merge_for(FakeCfg(), str(adir), merged, str(tmp_path / "grpo"),
                    fake_merge(merged), log=lambda *a: None)
    assert out == merged
    assert os.path.exists(os.path.join(merged, BINDING_NAME))


def test_tokenizer_drift_halts(tmp_path):
    adir = make_adapter(tmp_path / "adapter")
    merged = str(tmp_path / "sft_merged")
    ensure_bound_merge(str(adir), merged, str(tmp_path / "grpo"), fake_merge(merged),
                       log=lambda *a: None)
    with open(os.path.join(merged, "tokenizer.json"), "a") as f:
        f.write("drift")
    with pytest.raises(IntegrityError):
        ensure_bound_merge(str(adir), merged, str(tmp_path / "grpo"), fake_merge(merged),
                           log=lambda *a: None)


def test_missing_index_halts(tmp_path):
    adir = make_adapter(tmp_path / "adapter")
    merged = str(tmp_path / "sft_merged")
    ensure_bound_merge(str(adir), merged, str(tmp_path / "grpo"), fake_merge(merged),
                       log=lambda *a: None)
    os.remove(os.path.join(merged, "model.safetensors.index.json"))
    with pytest.raises(IntegrityError):
        ensure_bound_merge(str(adir), merged, str(tmp_path / "grpo"), fake_merge(merged),
                           log=lambda *a: None)


def test_shard_named_by_the_index_must_exist(tmp_path):
    from scripts.s28_compose import check_merged_complete
    merged = make_merged(str(tmp_path / "sft_merged"))
    os.remove(os.path.join(merged, "model-00002-of-00002.safetensors"))
    with pytest.raises(IntegrityError):
        check_merged_complete(merged)


def test_metrics_missing_a_tier_is_rejected(tmp_path):
    src = _base_eval_src(tmp_path)
    m = json.load(open(src / "base_eval.json"))
    del m["per_tier_acc"]["hard"]
    json.dump(m, open(src / "base_eval.json", "w"))
    with pytest.raises(IntegrityError):
        prepopulate_base_evals(str(src), str(tmp_path / "dst"), log=lambda *a: None)


def test_incomplete_fresh_merge_halts(tmp_path):
    adir = make_adapter(tmp_path / "adapter")
    merged = str(tmp_path / "sft_merged")

    def bad_merge():
        os.makedirs(merged, exist_ok=True)
        with open(os.path.join(merged, "config.json"), "w") as f:
            f.write("cfg only, no tokenizer, no weights")
        return merged

    with pytest.raises(IntegrityError):
        ensure_bound_merge(str(adir), merged, str(tmp_path / "grpo"), bad_merge,
                           log=lambda *a: None)


def test_wrong_recorded_accuracy_is_rejected(tmp_path):
    src = _base_eval_src(tmp_path)
    m = json.load(open(src / "base_eval.json"))
    m["per_tier_acc"]["easy"] += 0.05
    json.dump(m, open(src / "base_eval.json", "w"))
    with pytest.raises(IntegrityError):
        prepopulate_base_evals(str(src), str(tmp_path / "dst"), log=lambda *a: None)


def test_prepopulation_makes_the_runner_skip_both_evals(tmp_path):
    # the actual runner skip helper, with evaluators that would blow up if the
    # prepopulated files did not cause a skip
    from scripts.run import load_or_run
    src = _base_eval_src(tmp_path)
    dst = str(tmp_path / "base_dst")
    prepopulate_base_evals(str(src), dst, log=lambda *a: None)

    def must_not_run():
        raise RuntimeError("evaluation ran despite prepopulation")

    ev = load_or_run(os.path.join(dst, "base_eval.json"), "base_eval", must_not_run)
    lp = load_or_run(os.path.join(dst, "base_logprob.json"), "base_logprob_eval", must_not_run)
    assert ev["n_items"] == N and lp["n_items"] == N


def test_verify_binding_passes_and_catches_drift(tmp_path):
    from scripts.s28_compose import verify_binding
    adir = make_adapter(tmp_path / "adapter")
    merged = str(tmp_path / "sft_merged")
    ensure_bound_merge(str(adir), merged, str(tmp_path / "grpo"), fake_merge(merged),
                       log=lambda *a: None)
    assert verify_binding(str(adir), merged)["adapter_sha256"]
    with open(os.path.join(merged, "tokenizer.json"), "a") as f:
        f.write("x")
    with pytest.raises(IntegrityError):
        verify_binding(str(adir), merged)
    with pytest.raises(IntegrityError):
        verify_binding(str(adir), str(tmp_path / "never_merged"))


def _endpoint_scratch(tmp_path):
    from scripts.s28_compose import ENDPOINT_FILES
    sdir = tmp_path / "scratch"
    sdir.mkdir(parents=True, exist_ok=True)
    rows = [{"subject": "s", "tier": "easy", "gold": "A", "pred": "A", "correct": True}] * N
    for n in ("grpo_preds.jsonl", "grpo_logprob_preds.jsonl"):
        with open(sdir / n, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    for n in ("grpo_eval.json", "grpo_logprob.json"):
        json.dump({"n_items": N, "per_tier_acc": {"easy": 1.0}}, open(sdir / n, "w"))
    return str(sdir)


def test_endpoint_commit_and_skip(tmp_path):
    from scripts.s28_compose import commit_endpoint_eval, endpoint_state, ENDPOINT_MANIFEST
    rdir = str(tmp_path / "results" / "arm")
    os.makedirs(rdir)
    sdir = _endpoint_scratch(tmp_path)
    gdir = make_adapter(tmp_path / "grpo_adapter", salt="g")
    assert commit_endpoint_eval(rdir, sdir, adapter_dir=str(gdir),
                                extra_manifest={"seed": 0}, log=lambda *a: None) == "committed"
    assert endpoint_state(rdir, str(gdir)) == "committed"
    man = json.load(open(os.path.join(rdir, ENDPOINT_MANIFEST)))
    assert man["sha256"]["grpo_preds.jsonl"] == sha256(os.path.join(rdir, "grpo_preds.jsonl"))
    assert man["grpo_adapter_sha256"]["adapter_model.safetensors"]
    assert commit_endpoint_eval(rdir, sdir, adapter_dir=str(gdir), log=lambda *a: None) == "skipped"

    # the endpoint is bound to the trained adapter: changing it flips the state
    with open(os.path.join(gdir, "adapter_model.safetensors"), "w") as f:
        f.write("swapped weights")
    assert endpoint_state(rdir, str(gdir)) == "tampered"
    assert endpoint_state(rdir) == "committed"


def test_endpoint_tamper_halts_and_orphans_recommit(tmp_path):
    from scripts.s28_compose import commit_endpoint_eval, endpoint_state, ENDPOINT_MANIFEST
    rdir = str(tmp_path / "results" / "arm")
    os.makedirs(rdir)
    sdir = _endpoint_scratch(tmp_path)
    commit_endpoint_eval(rdir, sdir, log=lambda *a: None)
    with open(os.path.join(rdir, "grpo_preds.jsonl"), "a") as f:
        f.write("\n")
    assert endpoint_state(rdir) == "tampered"
    with pytest.raises(IntegrityError):
        commit_endpoint_eval(rdir, sdir, log=lambda *a: None)

    # without a manifest the same files are orphans and the commit repeats
    os.remove(os.path.join(rdir, ENDPOINT_MANIFEST))
    assert commit_endpoint_eval(rdir, sdir, log=lambda *a: None) == "committed"
    assert any(d.startswith("orphaned_") for d in os.listdir(rdir))


def test_endpoint_rejects_short_scratch(tmp_path):
    from scripts.s28_compose import commit_endpoint_eval
    rdir = str(tmp_path / "results" / "arm")
    os.makedirs(rdir)
    sdir = _endpoint_scratch(tmp_path)
    with open(os.path.join(sdir, "grpo_preds.jsonl"), "w") as f:
        f.write("")
    with pytest.raises(IntegrityError):
        commit_endpoint_eval(rdir, sdir, log=lambda *a: None)
    assert not os.path.exists(os.path.join(rdir, "grpo_eval.json"))


def test_run_py_ledger_can_be_suppressed():
    from scripts import run as run_mod
    run_mod._live.clear()
    run_mod._live["suppress"] = True
    run_mod._live["row"] = {"timings_sec": {}, "eval_items": {}}
    run_mod._live["path"] = "/nonexistent/ledger.jsonl"
    run_mod._flush_ledger()  # would raise on the bad path if it tried to write
    run_mod._live.clear()


def test_gate_state_three_way(tmp_path):
    from scripts.s28_compose import gate_state
    from scripts.s12_eval import atomic_write_json
    archived = tmp_path / "sft_logprob_preds.jsonl"
    archived.write_text("rows")
    gp = str(tmp_path / "parent_gate.json")
    hashes = {"adapter_model.safetensors": "abc"}

    assert gate_state(gp, hashes, str(archived)) == "absent"

    atomic_write_json(gp, {"completed": True, "adapter_sha256": hashes,
                           "stats": {"agreement": 1.0}, "new_preds_sha256": "x",
                           "archived_preds_sha256": sha256(str(archived))})
    assert gate_state(gp, hashes, str(archived)) == "committed"

    # frozen parent predictions changed: halt, never a silent rerun
    archived.write_text("rows changed")
    with pytest.raises(IntegrityError):
        gate_state(gp, hashes, str(archived))
    archived.write_text("rows")

    # a record bound to different adapter bytes: halt as well
    with pytest.raises(IntegrityError):
        gate_state(gp, {"adapter_model.safetensors": "other"}, str(archived))


def test_arm_completion_record_roundtrip(tmp_path):
    from scripts.s28_compose import (arm_state, commit_endpoint_eval,
                                     write_arm_complete)
    rdir = str(tmp_path / "results" / "arm")
    os.makedirs(rdir)
    gdir = make_adapter(tmp_path / "grpo_adapter", salt="g")
    sdir = _endpoint_scratch(tmp_path)
    commit_endpoint_eval(rdir, sdir, adapter_dir=str(gdir), log=lambda *a: None)
    json.dump({"reference": "mcsft_parent:x"}, open(os.path.join(rdir, "summary.json"), "w"))
    json.dump({"completed": True}, open(os.path.join(rdir, "manifest.json"), "w"))

    assert arm_state(rdir, str(gdir)) == "absent"
    write_arm_complete(rdir)
    assert arm_state(rdir, str(gdir)) == "committed"

    # a truncated summary contradicts the record and halts, never skips
    with open(os.path.join(rdir, "summary.json"), "w") as f:
        f.write("{")
    with pytest.raises(IntegrityError):
        arm_state(rdir, str(gdir))


# 029 (w14b) additions: the composition at 1.5b and 3b. The a2 abort becomes
# a concluded outcome the loop skips over, everything else still halts

from scripts.grpo_worker import blocked_resume_reason
from scripts.s28_compose import (A2_MESSAGE, ARMS_029, CONCLUDED_A2, SCALES_029,
                                 POOL_VARIABILITY, commit_pool_variability,
                                 conclude_a2, concluded_a2_state,
                                 pool_variability_state, sentinel_abort_reason,
                                 status_is_a2)

A2_MSG = A2_MESSAGE


def a2_attempt(**over):
    a = {"completed": False, "cuda_oom": False, "resume_blocked": True,
         "error_type": "RuntimeError", "error_message": A2_MSG, "end_step": 100}
    a.update(over)
    return a


def test_exact_a2_abort_is_a_concluded_outcome():
    assert sentinel_abort_reason([a2_attempt()]) == A2_MSG
    # and the arm never counts as resumable work either, 028 semantics intact
    assert blocked_resume_reason([a2_attempt()]) is not None


def test_everything_else_still_halts():
    assert sentinel_abort_reason([]) is None
    assert sentinel_abort_reason([a2_attempt(cuda_oom=True)]) is None
    assert sentinel_abort_reason([a2_attempt(error_message="worker failed with exit code 1")]) is None
    assert sentinel_abort_reason([a2_attempt(error_type="KeyboardInterrupt")]) is None
    assert sentinel_abort_reason([a2_attempt(completed=True)]) is None
    # a different sentinel is not an a2 conclusion
    assert sentinel_abort_reason([a2_attempt(error_message="sentinel abort: a4: projection over budget")]) is None
    # near-misses of the exact registered message do not classify either
    assert sentinel_abort_reason([a2_attempt(error_message=A2_MSG + " extra")]) is None
    assert sentinel_abort_reason([a2_attempt(error_message=A2_MSG[:-5])]) is None
    # only the last attempt decides
    assert sentinel_abort_reason([a2_attempt(), a2_attempt(error_message="oom", cuda_oom=True)]) is None


def test_status_predicate_matches_and_ambiguity_halts():
    assert status_is_a2({"completed": False, "cuda_oom": False,
                         "error_type": "RuntimeError", "error_message": A2_MSG})
    assert not status_is_a2({})
    # a status file missing the completed field does not classify
    assert not status_is_a2({"error_type": "RuntimeError", "error_message": A2_MSG})


def pool_record(hashes):
    return {"parent_tag": "010_p8_COLAB_FULL_B15_sft_only_seed0",
            "adapter_sha256": hashes, "n_prompts": 245, "samples_per_prompt": 8,
            "temperature": 1.0, "split_revision": "cais/mmlu@c30699e",
            "share_single_completion": 0.9,
            "mean_distinct_completions": 1.2, "share_single_answer": 0.95,
            "mean_completion_tokens": 16.4,
            "mean_accuracy_reward": 0.72, "format_rate": 1.0,
            "share_zero_reward_spread": 0.88, "mean_reward_std": 0.03,
            "ts": "2026-08-12T10:00:00"}


def test_pool_variability_roundtrip_and_binding(tmp_path):
    rdir = str(tmp_path / "res"); os.makedirs(rdir)
    hashes = {n: "h" + n for n in ADAPTER_FILES}
    assert pool_variability_state(rdir, hashes) == "absent"
    commit_pool_variability(rdir, pool_record(hashes))
    assert pool_variability_state(rdir, hashes) == "committed"
    with pytest.raises(IntegrityError):
        pool_variability_state(rdir, {n: "other" for n in ADAPTER_FILES})


def test_pool_variability_rejects_junk(tmp_path):
    hashes = {n: "h" + n for n in ADAPTER_FILES}
    for mutate in (lambda r: r.pop("share_single_completion"),
                   lambda r: r.pop("mean_reward_std"),
                   lambda r: r.pop("share_zero_reward_spread"),
                   lambda r: r.update(n_prompts=200),
                   lambda r: r.update(samples_per_prompt=4),
                   lambda r: r.update(temperature=0.7),
                   lambda r: r.update(split_revision="")):
        bad = pool_record(hashes)
        mutate(bad)
        with pytest.raises(IntegrityError):
            commit_pool_variability(str(tmp_path), bad)


def test_029_arm_table_is_seed_and_scale_matched():
    assert len(ARMS_029) == 6
    for cfg_path, seed, parent in ARMS_029:
        assert f"seed{seed}" in cfg_path and parent.endswith(f"seed{seed}")
        scale = "B15" if "_b15_" in cfg_path else "B3"
        assert scale in parent
        assert os.path.exists(cfg_path), cfg_path
        assert os.path.exists(SCALES_029[scale]["parent_config"])


def test_029_configs_match_their_seed_pipeline_sources_exactly():
    allowed_top = {"experiment", "condition", "exploratory"}
    src = {"B15": ("configs/009_colab_full_b15_seed{s}.yaml", "010_p8_COLAB_FULL_B15"),
           "B3": ("configs/003_colab_full_b3_seed{s}.yaml", "007_p8_COLAB_FULL_B3")}
    for cfg_path, s, parent in ARMS_029:
        scale = "B15" if "_b15_" in cfg_path else "B3"
        a = yaml.safe_load(open(cfg_path))
        b = yaml.safe_load(open(src[scale][0].format(s=s)))
        assert a["experiment"] == "029" and a["condition"] == "MCSFT_GRPO"
        assert a["exploratory"] is True
        a_train = dict(a["train"])
        assert a_train.pop("grpo_from_adapter") == f"checkpoints/{parent}/sft"
        assert a_train == b["train"]
        # the oom ladder stays off under the current worker regime, the one
        # registered exec difference against the pipeline sources
        a_exec = dict(a["exec"])
        assert a_exec.pop("allow_oom_fallback") is False
        assert a_exec == b["exec"]
        for key in set(a) | set(b):
            if key in allowed_top or key in ("train", "exec"):
                continue
            assert a.get(key) == b.get(key), f"{cfg_path}: {key} differs"


def test_029_configs_differ_only_by_seed_scale_and_source():
    # within a scale the three configs differ by seed and source path alone
    for low, scale_tag in (("b15", "B15"), ("b3", "B3")):
        parsed = [yaml.safe_load(open(f"configs/029_colab_{low}_mcsft_grpo_seed{s}.yaml"))
                  for s in (0, 1, 2)]
        for i in (1, 2):
            a, b = dict(parsed[0]), dict(parsed[i])
            at, bt = dict(a.pop("train")), dict(b.pop("train"))
            assert a == b
            assert at.pop("seed") == 0 and bt.pop("seed") == i
            assert at.pop("grpo_from_adapter").endswith("seed0/sft")
            assert bt.pop("grpo_from_adapter").endswith(f"seed{i}/sft")
            assert at == bt


def test_029_configs_load_and_plan_the_mcsft_stages():
    from scripts.run import plan_stages
    for cfg_path, seed, parent in ARMS_029:
        cfg = load_config(cfg_path)
        assert cfg.condition == Condition.MCSFT_GRPO
        assert cfg.train.seed == seed
        assert cfg.exec.allow_oom_fallback is False
        assert plan_stages(cfg.condition) == ["base_eval", "grpo", "grpo_eval"]


def test_a2_conclusion_record_roundtrip(tmp_path):
    rdir = str(tmp_path / "res"); os.makedirs(rdir)
    att = a2_attempt(adapter_sha256={"adapter_model.safetensors": "x"},
                     elapsed_sec=600.0, peak_vram_gib=12.3)
    assert concluded_a2_state(rdir, [att]) == "absent"
    rec = conclude_a2(rdir, att, evidence={"share_zero_reward_spread": 0.9})
    assert rec["outcome"] == "concluded_a2" and rec["end_step"] == 100
    assert concluded_a2_state(rdir, [att]) == "committed"


def test_a2_conclusion_refuses_non_a2_and_endpoint_leftovers(tmp_path):
    rdir = str(tmp_path / "res"); os.makedirs(rdir)
    with pytest.raises(IntegrityError):
        conclude_a2(rdir, a2_attempt(error_message="worker crash"))
    # a concluded arm must carry no endpoint artifacts at all
    with open(os.path.join(rdir, "summary.json"), "w") as f:
        f.write("{}")
    with pytest.raises(IntegrityError):
        conclude_a2(rdir, a2_attempt())


def test_a2_conclusion_record_must_match_the_manifest(tmp_path):
    rdir = str(tmp_path / "res"); os.makedirs(rdir)
    conclude_a2(rdir, a2_attempt())
    # the record without a matching a2 attempt is an integrity halt
    with pytest.raises(IntegrityError):
        concluded_a2_state(rdir, [a2_attempt(completed=True, error_type=None,
                                             error_message=None)])
    tampered = json.load(open(os.path.join(rdir, CONCLUDED_A2)))
    tampered["message"] = "something else"
    with open(os.path.join(rdir, CONCLUDED_A2), "w") as f:
        json.dump(tampered, f)
    with pytest.raises(IntegrityError):
        concluded_a2_state(rdir, [a2_attempt()])


def test_endpoint_manifest_carries_the_package_label(tmp_path):
    from scripts.s28_compose import ENDPOINT_FILES, ENDPOINT_MANIFEST, commit_endpoint_eval
    sdir = str(tmp_path / "scratch"); os.makedirs(sdir)
    rdir = str(tmp_path / "res"); os.makedirs(rdir)
    row = json.dumps({"subject": "s", "gold": "A", "correct": True})
    for n in ENDPOINT_FILES:
        with open(os.path.join(sdir, n), "w") as f:
            f.write('{"n_items": 1884}' if n.endswith(".json") else (row + "\n") * N)
    commit_endpoint_eval(rdir, sdir, package="029_w14b")
    man = json.load(open(os.path.join(rdir, ENDPOINT_MANIFEST)))
    assert man["package"] == "029_w14b"
    # and the default label stays the historical one for the 028 caller
    rdir2 = str(tmp_path / "res2"); os.makedirs(rdir2)
    commit_endpoint_eval(rdir2, sdir)
    assert json.load(open(os.path.join(rdir2, ENDPOINT_MANIFEST)))["package"] == "028_w14"


def test_a2_conclusion_rejects_later_endpoint_artifacts(tmp_path):
    rdir = str(tmp_path / "res"); os.makedirs(rdir)
    att = a2_attempt()
    conclude_a2(rdir, att)
    assert concluded_a2_state(rdir, [att]) == "committed"
    # an endpoint file appearing after the conclusion is an integrity halt
    with open(os.path.join(rdir, "grpo_eval.json"), "w") as f:
        f.write("{}")
    with pytest.raises(IntegrityError):
        concluded_a2_state(rdir, [att])


def test_a2_conclusion_fields_must_describe_the_manifest_attempt(tmp_path):
    rdir = str(tmp_path / "res"); os.makedirs(rdir)
    att = a2_attempt(adapter_sha256={"adapter_model.safetensors": "x"},
                     elapsed_sec=600.0, peak_vram_gib=12.3)
    conclude_a2(rdir, att)
    # same abort shape but a different attempt underneath: every drifted
    # field halts on its own
    base = dict(adapter_sha256={"adapter_model.safetensors": "x"},
                elapsed_sec=600.0, peak_vram_gib=12.3)
    for field, other in (("end_step", 150), ("elapsed_sec", 601.0),
                         ("peak_vram_gib", 50.0),
                         ("adapter_sha256", {"adapter_model.safetensors": "y"})):
        over = dict(base)
        over[field] = other
        with pytest.raises(IntegrityError):
            concluded_a2_state(rdir, [a2_attempt(**over)])
    assert concluded_a2_state(rdir, [att]) == "committed"
