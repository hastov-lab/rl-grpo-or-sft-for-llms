# Every record here binds to the merged start the branches actually load, so a
# baseline taken from a different representation cannot pass unnoticed.
import json
import os

import pytest

from config import RunConfig
from data.synthetic_facts import build_bundle
from scripts.s12_eval import atomic_write_json
from scripts.s28_compose import ADAPTER_FILES, IntegrityError
from scripts.w2_probe import (IDENTITY_FIELDS, align_fact_rows, bind_start, commit_eval,
                              eval_state, load_bundle, smoke_qualification, write_bundle)


def _write(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _start_tree(tmp_path):
    root = str(tmp_path)
    tag = "009_COLAB_FULL_B15_sft_grpo_seed0"
    adapter = os.path.join(root, "checkpoints", tag, "sft")
    merged = os.path.join(root, "checkpoints", tag, "sft_merged")
    for name in ADAPTER_FILES:
        _write(os.path.join(adapter, name), name)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors"):
        _write(os.path.join(merged, name), name)
    manifest = os.path.join(root, "results", tag, "manifest.json")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    atomic_write_json(manifest, {"run_tag": tag, "seed": 0})
    return tag, adapter, merged, manifest


def _prediction_rows(bundle, correct=False, format_ok=True):
    rows = []
    for source in bundle["eval_rows"]:
        row = {field: source[field] for field in IDENTITY_FIELDS}
        row.update({"gold": source["gold"], "pred": source["gold"] if correct else None,
                    "correct": correct, "format_ok": format_ok})
        rows.append(row)
    return rows


def _write_eval_scratch(path, condition, bundle, correct=False):
    os.makedirs(path, exist_ok=True)
    rows = _prediction_rows(bundle, correct=correct)
    for suffix in ("preds.jsonl", "logprob_preds.jsonl"):
        with open(os.path.join(path, f"{condition}_{suffix}"), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    for suffix in ("eval.json", "logprob.json"):
        atomic_write_json(os.path.join(path, f"{condition}_{suffix}"),
                          {"n_items": len(rows)})
    return rows


def test_bundle_and_start_binding_roundtrip(monkeypatch, tmp_path):
    import train.sft

    tag, adapter, merged, manifest = _start_tree(tmp_path)
    bundle_path = os.path.join(tmp_path, "bundle.json")
    bundle = build_bundle("smoke", 34001, 16, 16)
    write_bundle(bundle_path, bundle)

    cfg = RunConfig()
    cfg.output_root = str(tmp_path)
    cfg.train.sft_from_adapter = os.path.relpath(adapter, str(tmp_path))
    monkeypatch.setattr(train.sft, "resolve_sft_start", lambda cfg: (
        merged, "local", os.path.relpath(merged, str(tmp_path))))
    record_path = os.path.join(tmp_path, "start.json")
    first = bind_start(cfg, bundle_path, manifest, tag, record_path)
    second = bind_start(cfg, bundle_path, manifest, tag, record_path)
    assert first == second
    assert first["dataset_sha256"] == load_bundle(bundle_path)["sha256"]

    _write(os.path.join(merged, "model.safetensors"), "changed")
    with pytest.raises(IntegrityError, match="no longer matches"):
        bind_start(cfg, bundle_path, manifest, tag, record_path)


def test_fact_alignment_rejects_changed_identity():
    bundle = build_bundle("smoke", 34001, 16, 16)
    rows = _prediction_rows(bundle)
    assert len(align_fact_rows(bundle["eval_rows"], rows)) == 96
    rows[0]["choice_order"] = list(reversed(rows[0]["choice_order"]))
    with pytest.raises(IntegrityError, match="choice_order"):
        align_fact_rows(bundle["eval_rows"], rows)


def test_endpoint_commit_is_bound_to_dataset_start_and_adapter(tmp_path):
    bundle = build_bundle("smoke", 34001, 16, 16)
    bundle_path = os.path.join(tmp_path, "bundle.json")
    write_bundle(bundle_path, bundle)
    start_path = os.path.join(tmp_path, "start.json")
    atomic_write_json(start_path, {"merged_sha256": {"model.safetensors": "abc"}})
    adapter = os.path.join(tmp_path, "adapter")
    for name in ADAPTER_FILES:
        _write(os.path.join(adapter, name), name)
    scratch = os.path.join(tmp_path, "scratch")
    result = os.path.join(tmp_path, "result")
    _write_eval_scratch(scratch, "sft", bundle, correct=True)

    assert commit_eval(result, scratch, "sft", bundle_path, start_path, adapter) == "committed"
    assert eval_state(result, "sft", adapter, bundle_path, start_path) == "committed"
    assert commit_eval(result, scratch, "sft", bundle_path, start_path, adapter) == "skipped"
    _write(os.path.join(adapter, "adapter_model.safetensors"), "changed")
    assert eval_state(result, "sft", adapter, bundle_path, start_path) == "tampered"


def test_endpoint_commit_detects_dataset_and_start_drift(tmp_path):
    bundle = build_bundle("smoke", 34001, 16, 16)
    bundle_path = os.path.join(tmp_path, "bundle.json")
    write_bundle(bundle_path, bundle)
    start_path = os.path.join(tmp_path, "start.json")
    atomic_write_json(start_path, {"merged_sha256": {"model.safetensors": "abc"}})
    scratch = os.path.join(tmp_path, "scratch")
    result = os.path.join(tmp_path, "result")
    _write_eval_scratch(scratch, "baseline", bundle, correct=True)

    commit_eval(result, scratch, "baseline", bundle_path, start_path)
    assert eval_state(result, "baseline", None, bundle_path, start_path) == "committed"
    atomic_write_json(start_path, {"merged_sha256": {"model.safetensors": "changed"}})
    assert eval_state(result, "baseline", None, bundle_path, start_path) == "tampered"


def test_smoke_qualification_uses_fixed_thresholds():
    bundle = build_bundle("smoke", 34001, 16, 16)
    base_gen = _prediction_rows(bundle, correct=False)
    base_probe = _prediction_rows(bundle, correct=False)
    sft_gen = _prediction_rows(bundle, correct=False)
    sft_probe = _prediction_rows(bundle, correct=False)
    injected_keys = [(row["fact_id"], row["variant"]) for row in bundle["eval_rows"]
                     if row["group"] == "injected" and row["variant"] == "canonical"]
    for rows in (sft_gen, sft_probe):
        by_key = {(row["fact_id"], row["variant"]): row for row in rows}
        for key in injected_keys[:12]:
            by_key[key]["correct"] = True
            by_key[key]["pred"] = by_key[key]["gold"]
    result = smoke_qualification(base_gen, base_probe, sft_gen, sft_probe)
    assert result["passed"] is True
    assert result["probe_gain_injected"] == 0.75


def test_smoke_configs_are_fixed_and_differ_only_by_condition():
    import yaml
    from config import load_config
    from enums import Condition, Scale

    paths = ("configs/034_smoke_sft_colab_b15.yaml",
             "configs/034_smoke_grpo_colab_b15.yaml")
    cfgs = [load_config(path) for path in paths]
    assert [cfg.condition for cfg in cfgs] == [Condition.SFT_ONLY, Condition.EXTERNAL_SFT_GRPO]
    for cfg in cfgs:
        assert cfg.experiment == "034_smoke" and cfg.exploratory is True
        assert cfg.scale == Scale.B15 and cfg.train.seed == 0
        assert cfg.train.sft_data == "synthetic_facts"
        assert cfg.train.max_steps == 50 and cfg.train.sft_val_examples == 0
        assert cfg.train.num_generations == 8 and cfg.grpo.max_completion_length == 512
        assert cfg.train.sft_from_adapter.endswith(
            "009_COLAB_FULL_B15_sft_grpo_seed0/sft")
        assert cfg.exec.allow_oom_fallback is False

    raw = [yaml.safe_load(open(path)) for path in paths]
    assert raw[0].pop("condition") == "SFT_ONLY"
    assert raw[1].pop("condition") == "EXTERNAL_SFT_GRPO"
    assert raw[0] == raw[1]
