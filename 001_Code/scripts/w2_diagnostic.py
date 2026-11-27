# declarative diagnosis for the failed W2 smoke arm
import json
import os
import re
import shutil
import time
from datetime import datetime

from data.synthetic_facts import (diagnostic_sha256, validate_declarative_diagnostic,
                                  validate_bundle)
from file_utils import read_json
from scripts.s12_eval import atomic_write_json, read_jsonl, sha256, stage_into
from scripts.s28_compose import IntegrityError, adapter_hashes_of


DIAGNOSTIC_FILES = (
    "generation_preds.jsonl",
    "generation.json",
    "continuation_preds.jsonl",
    "continuation.json",
)
IDENTITY_FIELDS = ("fact_id", "relation", "group", "variant", "prompt",
                   "subject", "object", "candidates")
THRESHOLDS = {
    "min_endpoint_accuracy": 0.75,
    "min_injected_gain": 0.25,
    "max_abs_control_change": 0.125,
}


def write_diagnostic(path, diagnostic, source_bundle):
    validate_declarative_diagnostic(diagnostic, source_bundle)
    if diagnostic.get("sha256") != diagnostic_sha256(diagnostic):
        raise IntegrityError("declarative diagnostic hash does not match its contents")
    if os.path.exists(path):
        if load_diagnostic(path, source_bundle) != diagnostic:
            raise IntegrityError(f"{path}: declarative diagnostic already differs")
        return diagnostic
    atomic_write_json(path, diagnostic)
    return load_diagnostic(path, source_bundle)


def load_diagnostic(path, source_bundle=None):
    try:
        diagnostic = read_json(path)
    except Exception as exc:
        raise IntegrityError(f"{path}: declarative diagnostic does not parse") from exc
    try:
        validate_declarative_diagnostic(diagnostic, source_bundle)
    except Exception as exc:
        raise IntegrityError(f"{path}: declarative diagnostic is invalid") from exc
    if diagnostic.get("sha256") != diagnostic_sha256(diagnostic):
        raise IntegrityError(f"{path}: declarative diagnostic hash mismatch")
    return diagnostic


def first_synthetic_name(text, names):
    lookup = {name.lower() for name in names}
    for match in re.finditer(r"[A-Za-z]+", text):
        value = match.group(0).lower()
        if value in lookup:
            return value
    return None


def generation_rows(diagnostic, outputs):
    if len(outputs) != len(diagnostic["rows"]):
        raise IntegrityError("declarative generation output count is wrong")
    names = {row["subject"] for row in diagnostic["rows"]}
    names.update(row["object"] for row in diagnostic["rows"])
    rows = []
    for source, output in zip(diagnostic["rows"], outputs):
        row = {field: source[field] for field in IDENTITY_FIELDS}
        first = first_synthetic_name(output, names)
        row.update({"output": output, "first_name": first,
                    "correct": first == source["object"].lower()})
        rows.append(row)
    return rows


def continuation_rows(diagnostic, scores):
    if len(scores) != len(diagnostic["rows"]):
        raise IntegrityError("declarative continuation score count is wrong")
    rows = []
    for source, values in zip(diagnostic["rows"], scores):
        if set(values) != set(source["candidates"]):
            raise IntegrityError("declarative continuation candidates changed")
        pred = max(source["candidates"], key=lambda value: values[value])
        correct = source["object"]
        strongest_wrong = max(values[value] for value in source["candidates"]
                              if value != correct)
        row = {field: source[field] for field in IDENTITY_FIELDS}
        row.update({"candidate_scores": values, "pred_object": pred,
                    "correct": pred == correct,
                    "correct_margin": values[correct] - strongest_wrong})
        rows.append(row)
    return rows


def metric_summary(condition, metric, rows):
    values = {}
    for group in ("injected", "control"):
        values[group] = {}
        for variant in ("exact", "declarative_1", "declarative_2"):
            selected = [row for row in rows
                        if row["group"] == group and row["variant"] == variant]
            values[group][variant] = sum(bool(row["correct"]) for row in selected) / len(selected)
    return {"condition": condition, "metric": metric, "n_items": len(rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "by_group_variant": values}


def _aligned_rows(diagnostic, actual, metric):
    expected = {(row["fact_id"], row["variant"]): row for row in diagnostic["rows"]}
    found = {(row["fact_id"], row["variant"]): row for row in actual}
    if len(found) != len(actual) or set(found) != set(expected):
        raise IntegrityError(f"declarative {metric} rows do not cover the fixed prompts")
    aligned = []
    for key in sorted(expected):
        source, row = expected[key], found[key]
        for field in IDENTITY_FIELDS:
            if row.get(field) != source[field]:
                raise IntegrityError(f"declarative {metric} row {key} changed {field}")
        if metric == "generation":
            if row.get("correct") != (row.get("first_name") == source["object"].lower()):
                raise IntegrityError(f"declarative generation row {key} has wrong scoring")
        else:
            scores = row.get("candidate_scores", {})
            if set(scores) != set(source["candidates"]):
                raise IntegrityError(f"declarative continuation row {key} changed candidates")
            pred = max(source["candidates"], key=lambda value: scores[value])
            if row.get("pred_object") != pred or row.get("correct") != (pred == source["object"]):
                raise IntegrityError(f"declarative continuation row {key} has wrong scoring")
        aligned.append(row)
    return aligned


def diagnostic_state(result_dir, condition, diagnostic_path, start_record_path,
                     adapter_dir=None):
    manifest_path = os.path.join(result_dir, "diagnostic_eval_manifest.json")
    if not os.path.exists(manifest_path):
        return "absent"
    try:
        manifest = read_json(manifest_path)
        diagnostic = load_diagnostic(diagnostic_path)
        start = read_json(start_record_path)
    except Exception:
        return "tampered"
    if manifest.get("completed") is not True or manifest.get("condition") != condition:
        return "tampered"
    for name in DIAGNOSTIC_FILES:
        path = os.path.join(result_dir, name)
        if not os.path.exists(path) or sha256(path) != manifest.get("sha256", {}).get(name):
            return "tampered"
    if manifest.get("diagnostic_sha256") != diagnostic["sha256"]:
        return "tampered"
    if manifest.get("diagnostic_file_sha256") != sha256(diagnostic_path):
        return "tampered"
    if manifest.get("start_record_sha256") != sha256(start_record_path):
        return "tampered"
    if manifest.get("merged_sha256") != start.get("merged_sha256"):
        return "tampered"
    source_sha = diagnostic.get("source_dataset_sha256")
    if source_sha != start.get("dataset_sha256"):
        return "tampered"
    if manifest.get("source_dataset_sha256") != source_sha:
        return "tampered"
    try:
        expected_adapter = adapter_hashes_of(adapter_dir) if adapter_dir else None
    except IntegrityError:
        return "tampered"
    if manifest.get("adapter_sha256") != expected_adapter:
        return "tampered"
    return "committed"


def _archive_orphans(result_dir):
    names = list(DIAGNOSTIC_FILES) + ["diagnostic_eval_manifest.json"]
    found = [name for name in names if os.path.exists(os.path.join(result_dir, name))]
    if not found:
        return
    archive = os.path.join(result_dir, "orphaned_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(archive, exist_ok=True)
    for name in found:
        shutil.move(os.path.join(result_dir, name), os.path.join(archive, name))


def commit_diagnostic_eval(result_dir, scratch_dir, condition, diagnostic_path,
                           start_record_path, adapter_dir=None, extra=None):
    state = diagnostic_state(result_dir, condition, diagnostic_path,
                             start_record_path, adapter_dir)
    if state == "tampered":
        raise IntegrityError(f"{result_dir}: committed declarative diagnostic is inconsistent")
    if state == "committed":
        return "skipped"

    diagnostic = load_diagnostic(diagnostic_path)
    generation = read_jsonl(os.path.join(scratch_dir, "generation_preds.jsonl"))
    continuation = read_jsonl(os.path.join(scratch_dir, "continuation_preds.jsonl"))
    generation = _aligned_rows(diagnostic, generation, "generation")
    continuation = _aligned_rows(diagnostic, continuation, "continuation")
    expected_metrics = {
        "generation.json": metric_summary(
            condition, "direct_generation", generation),
        "continuation.json": metric_summary(
            condition, "continuation_score", continuation),
    }
    for name, expected in expected_metrics.items():
        if read_json(os.path.join(scratch_dir, name)) != expected:
            raise IntegrityError(f"{scratch_dir}: {name} does not match its rows")

    os.makedirs(result_dir, exist_ok=True)
    _archive_orphans(result_dir)
    hashes = {}
    for name in DIAGNOSTIC_FILES:
        final = stage_into(result_dir, os.path.join(scratch_dir, name), name)
        hashes[name] = sha256(final)
    start = read_json(start_record_path)
    manifest = {
        "schema": 1,
        "completed": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "condition": condition,
        "n_items": len(diagnostic["rows"]),
        "diagnostic_sha256": diagnostic["sha256"],
        "diagnostic_file_sha256": sha256(diagnostic_path),
        "source_dataset_sha256": diagnostic["source_dataset_sha256"],
        "start_record_sha256": sha256(start_record_path),
        "merged_sha256": start["merged_sha256"],
        "adapter_sha256": adapter_hashes_of(adapter_dir) if adapter_dir else None,
        "sha256": hashes,
    }
    if extra:
        manifest.update(extra)
    # commit the manifest after all four result files
    atomic_write_json(os.path.join(result_dir, "diagnostic_eval_manifest.json"), manifest)
    if diagnostic_state(result_dir, condition, diagnostic_path,
                        start_record_path, adapter_dir) != "committed":
        raise IntegrityError(f"{result_dir}: declarative diagnostic did not commit")
    return "committed"


def _accuracy(rows, group, variants):
    selected = [row for row in rows
                if row["group"] == group and row["variant"] in variants]
    return sum(bool(row["correct"]) for row in selected) / len(selected)


def _measure_values(baseline, endpoint):
    out = {}
    for label, variants in (("exact", ("exact",)),
                            ("paraphrase", ("declarative_1", "declarative_2"))):
        base_injected = _accuracy(baseline, "injected", variants)
        end_injected = _accuracy(endpoint, "injected", variants)
        base_control = _accuracy(baseline, "control", variants)
        end_control = _accuracy(endpoint, "control", variants)
        out[label] = {
            "baseline_injected": base_injected,
            "endpoint_injected": end_injected,
            "injected_gain": end_injected - base_injected,
            "baseline_control": base_control,
            "endpoint_control": end_control,
            "control_change": end_control - base_control,
            "passes_acquisition": (
                end_injected >= THRESHOLDS["min_endpoint_accuracy"]
                and end_injected - base_injected >= THRESHOLDS["min_injected_gain"]),
        }
    return out


def classify_diagnostic(baseline_generation, baseline_continuation,
                        endpoint_generation, endpoint_continuation):
    values = {
        "generation": _measure_values(baseline_generation, endpoint_generation),
        "continuation": _measure_values(baseline_continuation, endpoint_continuation),
    }
    controls_stable = all(
        abs(values[metric][variant]["control_change"])
        <= THRESHOLDS["max_abs_control_change"]
        for metric in values for variant in ("exact", "paraphrase"))
    exact = [values[metric]["exact"]["passes_acquisition"] for metric in values]
    paraphrase = [values[metric]["paraphrase"]["passes_acquisition"] for metric in values]
    if not controls_stable:
        route = "interference"
    elif len(set(exact)) > 1 or len(set(paraphrase)) > 1:
        route = "measure_disagreement"
    elif exact[0] and paraphrase[0]:
        route = "declarative_acquired"
    elif exact[0] and not paraphrase[0]:
        route = "exact_only"
    elif not exact[0] and not paraphrase[0]:
        route = "declarative_unqualified"
    else:
        route = "unexpected_order"
    return {
        "completed": True,
        "post_hoc": True,
        "registered_w2_evidence": False,
        "independent_validation": False,
        "c1_claim_allowed": False,
        "repair_authorised": False,
        "thresholds": THRESHOLDS,
        "values": values,
        "controls_stable": controls_stable,
        "route": route,
    }


def write_diagnostic_summary(path, summary):
    if os.path.exists(path):
        if read_json(path) != summary:
            raise IntegrityError(f"{path}: declarative diagnostic summary changed")
        return summary
    atomic_write_json(path, summary)
    return read_json(path)
