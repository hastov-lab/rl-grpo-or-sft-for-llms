# dataset, start and endpoint records for the W2 smoke run
import json
import os
import shutil
import time
from datetime import datetime

from data.synthetic_facts import bundle_sha256, validate_bundle
from file_utils import read_json
from scripts.s12_eval import atomic_write_json, read_jsonl, sha256, stage_into
from scripts.s28_compose import (IntegrityError, adapter_hashes_of,
                                 check_merged_complete, merged_hashes)


IDENTITY_FIELDS = ("fact_id", "relation", "group", "variant", "choice_order")
EVAL_PARTS = ("preds.jsonl", "eval.json", "logprob_preds.jsonl", "logprob.json")


def write_bundle(path, bundle):
    validate_bundle(bundle)
    if bundle.get("sha256") != bundle_sha256(bundle):
        raise IntegrityError("synthetic bundle hash does not match its contents")
    if os.path.exists(path):
        if load_bundle(path) != bundle:
            raise IntegrityError(f"{path}: synthetic bundle already exists with different contents")
        return bundle
    atomic_write_json(path, bundle)
    return load_bundle(path)


def load_bundle(path):
    try:
        bundle = read_json(path)
    except Exception as exc:
        raise IntegrityError(f"{path}: synthetic bundle does not parse") from exc
    validate_bundle(bundle)
    if bundle.get("sha256") != bundle_sha256(bundle):
        raise IntegrityError(f"{path}: synthetic bundle hash mismatch")
    return bundle


def _start_payload(cfg, bundle_path, parent_manifest_path, expected_parent):
    from train.sft import resolve_sft_start

    adapter_rel = cfg.train.sft_from_adapter
    if not adapter_rel:
        raise IntegrityError("W2 start config has no source adapter")
    adapter_dir = adapter_rel if os.path.isabs(adapter_rel) else os.path.join(
        cfg.output_root, adapter_rel)
    expected_merged = os.path.join(os.path.dirname(adapter_dir), "sft_merged")
    if not os.path.exists(os.path.join(expected_merged, "config.json")):
        raise IntegrityError(f"{expected_merged}: archived merged start is missing")

    # bind the record to the exact merged path training will load
    model_src, base_kind, base_val = resolve_sft_start(cfg)
    if base_kind != "local" or os.path.abspath(model_src) != os.path.abspath(expected_merged):
        raise IntegrityError("W2 resolver did not return the archived parent merge")
    check_merged_complete(model_src)

    try:
        parent = read_json(parent_manifest_path)
    except Exception as exc:
        raise IntegrityError(f"{parent_manifest_path}: parent manifest does not parse") from exc
    if parent.get("run_tag") != expected_parent or parent.get("seed") != cfg.train.seed:
        raise IntegrityError("W2 parent manifest has the wrong run tag or seed")

    bundle = load_bundle(bundle_path)
    return {
        "schema": 1,
        "completed": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "parent_tag": expected_parent,
        "seed": cfg.train.seed,
        "source_adapter": os.path.abspath(adapter_dir),
        "source_adapter_sha256": adapter_hashes_of(adapter_dir),
        "merged_dir": os.path.abspath(model_src),
        "merged_sha256": merged_hashes(model_src),
        "base_value": base_val,
        "parent_manifest": os.path.abspath(parent_manifest_path),
        "parent_manifest_sha256": sha256(parent_manifest_path),
        "dataset": os.path.abspath(bundle_path),
        "dataset_file_sha256": sha256(bundle_path),
        "dataset_sha256": bundle["sha256"],
    }


def bind_start(cfg, bundle_path, parent_manifest_path, expected_parent, record_path):
    current = _start_payload(cfg, bundle_path, parent_manifest_path, expected_parent)
    if os.path.exists(record_path):
        recorded = read_json(record_path)
        comparable = dict(current)
        comparable["timestamp"] = recorded.get("timestamp")
        if recorded != comparable:
            raise IntegrityError(f"{record_path}: start binding no longer matches its sources")
        return recorded
    atomic_write_json(record_path, current)
    return read_json(record_path)


def verify_start(cfg, record_path):
    try:
        record = read_json(record_path)
    except Exception as exc:
        raise IntegrityError(f"{record_path}: start binding does not parse") from exc
    current = _start_payload(
        cfg, record["dataset"], record["parent_manifest"], record["parent_tag"])
    current["timestamp"] = record.get("timestamp")
    if current != record:
        raise IntegrityError(f"{record_path}: start binding no longer matches its sources")
    return record


def align_fact_rows(expected, actual):
    expected_by_key = {(row["fact_id"], row["variant"]): row for row in expected}
    actual_by_key = {(row["fact_id"], row["variant"]): row for row in actual}
    if len(expected_by_key) != len(expected) or len(actual_by_key) != len(actual):
        raise IntegrityError("W2 evaluation rows contain duplicate fact variants")
    if set(expected_by_key) != set(actual_by_key):
        raise IntegrityError("W2 evaluation rows do not cover the fixed fact variants")
    aligned = []
    for key in sorted(expected_by_key):
        left, right = expected_by_key[key], actual_by_key[key]
        for field in ("gold",) + IDENTITY_FIELDS:
            if left[field] != right[field]:
                raise IntegrityError(f"W2 evaluation row {key} changed field {field}")
        aligned.append(right)
    return aligned


def eval_names(condition):
    return {
        "preds.jsonl": f"{condition}_preds.jsonl",
        "eval.json": f"{condition}_eval.json",
        "logprob_preds.jsonl": f"{condition}_logprob_preds.jsonl",
        "logprob.json": f"{condition}_logprob.json",
    }


def eval_state(result_dir, condition, adapter_dir=None, bundle_path=None,
               start_record_path=None):
    manifest_path = os.path.join(result_dir, f"{condition}_eval_manifest.json")
    if not os.path.exists(manifest_path):
        return "absent"
    try:
        manifest = read_json(manifest_path)
    except Exception:
        return "tampered"
    if manifest.get("completed") is not True:
        return "tampered"
    for name in eval_names(condition).values():
        path = os.path.join(result_dir, name)
        if not os.path.exists(path) or sha256(path) != manifest.get("sha256", {}).get(name):
            return "tampered"
    if adapter_dir is not None:
        try:
            current = adapter_hashes_of(adapter_dir)
        except IntegrityError:
            return "tampered"
        if manifest.get("adapter_sha256") != current:
            return "tampered"
    if bundle_path is not None:
        try:
            bundle = load_bundle(bundle_path)
        except IntegrityError:
            return "tampered"
        if manifest.get("dataset_sha256") != bundle["sha256"]:
            return "tampered"
    if start_record_path is not None:
        try:
            start = read_json(start_record_path)
        except Exception:
            return "tampered"
        if manifest.get("start_record_sha256") != sha256(start_record_path):
            return "tampered"
        if manifest.get("merged_sha256") != start.get("merged_sha256"):
            return "tampered"
    return "committed"


def _archive_eval_orphans(result_dir, condition):
    names = list(eval_names(condition).values()) + [f"{condition}_eval_manifest.json"]
    found = [name for name in names if os.path.exists(os.path.join(result_dir, name))]
    if not found:
        return []
    archive = os.path.join(result_dir, "orphaned_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(archive, exist_ok=True)
    for name in found:
        shutil.move(os.path.join(result_dir, name), os.path.join(archive, name))
    return found


def commit_eval(result_dir, scratch_dir, condition, bundle_path, start_record_path,
                adapter_dir=None, extra=None):
    state = eval_state(result_dir, condition, adapter_dir, bundle_path,
                       start_record_path)
    if state == "tampered":
        raise IntegrityError(f"{result_dir}: committed {condition} evaluation is inconsistent")
    if state == "committed":
        return "skipped"

    bundle = load_bundle(bundle_path)
    start = read_json(start_record_path)
    names = eval_names(condition)
    for source_name in names.values():
        if not os.path.exists(os.path.join(scratch_dir, source_name)):
            raise IntegrityError(f"{scratch_dir}: missing {source_name}")
    gen_rows = read_jsonl(os.path.join(scratch_dir, names["preds.jsonl"]))
    probe_rows = read_jsonl(os.path.join(scratch_dir, names["logprob_preds.jsonl"]))
    align_fact_rows(bundle["eval_rows"], gen_rows)
    align_fact_rows(bundle["eval_rows"], probe_rows)
    expected_n = len(bundle["eval_rows"])
    if read_json(os.path.join(scratch_dir, names["eval.json"])).get("n_items") != expected_n:
        raise IntegrityError("W2 generated metrics have the wrong row count")
    if read_json(os.path.join(scratch_dir, names["logprob.json"])).get("n_items") != expected_n:
        raise IntegrityError("W2 probe metrics have the wrong row count")

    os.makedirs(result_dir, exist_ok=True)
    _archive_eval_orphans(result_dir, condition)
    hashes = {}
    for name in names.values():
        final = stage_into(result_dir, os.path.join(scratch_dir, name), name)
        hashes[name] = sha256(final)
    manifest = {
        "schema": 1,
        "completed": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "condition": condition,
        "n_items": expected_n,
        "dataset_sha256": bundle["sha256"],
        "start_record_sha256": sha256(start_record_path),
        "merged_sha256": start["merged_sha256"],
        "adapter_sha256": adapter_hashes_of(adapter_dir) if adapter_dir else None,
        "sha256": hashes,
    }
    if extra:
        manifest.update(extra)
    # write the manifest last so incomplete files cannot look committed
    atomic_write_json(os.path.join(result_dir, f"{condition}_eval_manifest.json"), manifest)
    if eval_state(result_dir, condition, adapter_dir, bundle_path,
                  start_record_path) != "committed":
        raise IntegrityError(f"{result_dir}: {condition} evaluation did not commit")
    return "committed"


def _accuracy(rows, group):
    selected = [row for row in rows if row["group"] == group and row["variant"] == "canonical"]
    return sum(bool(row["correct"]) for row in selected) / len(selected)


def smoke_qualification(baseline_gen, baseline_probe, sft_gen, sft_probe):
    values = {}
    for label, rows in (("base_gen", baseline_gen), ("base_probe", baseline_probe),
                        ("sft_gen", sft_gen), ("sft_probe", sft_probe)):
        values[label] = {group: _accuracy(rows, group) for group in ("injected", "control")}
    probe_gain = values["sft_probe"]["injected"] - values["base_probe"]["injected"]
    probe_control = values["sft_probe"]["control"] - values["base_probe"]["control"]
    gen_control = values["sft_gen"]["control"] - values["base_gen"]["control"]
    canonical_sft = [row for row in sft_gen if row["variant"] == "canonical"]
    format_rate = sum(bool(row["format_ok"]) for row in canonical_sft) / len(canonical_sft)
    thresholds = {
        "min_generated_injected": 0.75,
        "min_probe_gain": 0.25,
        "max_abs_control_change": 0.125,
        "min_format_rate": 0.90,
    }
    passed = (
        values["sft_gen"]["injected"] >= thresholds["min_generated_injected"]
        and probe_gain >= thresholds["min_probe_gain"]
        and abs(probe_control) <= thresholds["max_abs_control_change"]
        and abs(gen_control) <= thresholds["max_abs_control_change"]
        and format_rate >= thresholds["min_format_rate"]
    )
    return {
        "passed": passed,
        "values": values,
        "probe_gain_injected": probe_gain,
        "probe_change_control": probe_control,
        "generated_change_control": gen_control,
        "format_rate": format_rate,
        "thresholds": thresholds,
    }
