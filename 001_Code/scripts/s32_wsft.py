import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime

from checkpoint import config_hash
from file_utils import read_json, read_text
from scripts.s12_eval import (IntegrityError, atomic_write_json,
                              complete_summary_nulls, identity_check, read_jsonl,
                              sha256, stage_into)
from scripts.s28_compose import (adapter_hashes_of, check_merged_complete,
                                 ensure_bound_merge, merged_hashes,
                                 prepopulate_base_evals)


# these memory gates use recorded SFT peaks with headroom

SCALES_032 = {
    "B05": {"phase": "40", "slug": "b05", "pipeline": "006_COLAB_FULL_B05_sft_grpo_seed{s}",
            "pipeline_config": "configs/006_colab_full_b05_seed{s}.yaml",
            "historical": "008_wsft_COLAB_FULL_B05_sft_only_seed{s}",
            "plain": "007_p8_COLAB_FULL_B05_sft_only_seed{s}",
            "base_eval": "results/base_evals/006_COLAB_FULL_B05", "vram_gib": 10.0},
    "B3": {"phase": "40", "slug": "b3", "pipeline": "003_COLAB_FULL_B3_sft_grpo_seed{s}",
           "pipeline_config": "configs/003_colab_full_b3_seed{s}.yaml",
           "historical": "008_wsft_COLAB_FULL_B3_sft_only_seed{s}",
           "plain": "007_p8_COLAB_FULL_B3_sft_only_seed{s}",
           "base_eval": "results/base_evals/003_COLAB_FULL_B3", "vram_gib": 22.0},
    "B15": {"phase": "80", "slug": "b15", "pipeline": "009_COLAB_FULL_B15_sft_grpo_seed{s}",
            "pipeline_config": "configs/009_colab_full_b15_seed{s}.yaml",
            "historical": "011_wsft_COLAB_FULL_B15_sft_only_seed{s}",
            "plain": "010_p8_COLAB_FULL_B15_sft_only_seed{s}",
            "base_eval": "results/base_evals/009_COLAB_FULL_B15", "vram_gib": 16.0},
    "B7": {"phase": "80", "slug": "b7", "pipeline": "013_COLAB_FULL_B7_sft_grpo_seed{s}",
           "pipeline_config": "configs/013_colab_full_b7_seed{s}.yaml",
           "historical": "015_wsft_COLAB_FULL_B7_sft_only_seed{s}",
           "plain": "014_p8_COLAB_FULL_B7_sft_only_seed{s}",
           "base_eval": "results/base_evals/013_COLAB_FULL_B7", "vram_gib": 36.0},
}

ARMS_032 = {
    "40": [(scale, seed) for scale in ("B05", "B3") for seed in (1, 2)],
    "80": [(scale, seed) for scale in ("B15", "B7") for seed in (1, 2)],
}

START_BINDING = "source_start_binding.json"
LINEAGE_RECORD = "source_lineage_v6.json"
GATE_FILES = ("source_adapter_gate_logprob.json",
              "source_adapter_gate_logprob_preds.jsonl")
GATE_MANIFEST = "source_adapter_gate_manifest.json"
ENDPOINT_FILES = ("sft_eval.json", "sft_preds.jsonl")
ENDPOINT_MANIFEST = "sft_eval_manifest.json"
ARM_COMPLETE = "032_arm_complete.json"
FAILED_GATE_CORRECTION = "032_b05_seed1_gate_20260814_132511"


def config_path(scale, seed):
    return f"configs/032_wsft_colab_{SCALES_032[scale]['slug']}_seed{seed}.yaml"


def arm_tag(scale, seed):
    return f"032_wsft_COLAB_FULL_{scale}_sft_only_seed{seed}"


def pipeline_tag(scale, seed):
    return SCALES_032[scale]["pipeline"].format(s=seed)


def pipeline_config_path(scale, seed):
    return SCALES_032[scale]["pipeline_config"].format(s=seed)


def historical_tag(scale, seed):
    return SCALES_032[scale]["historical"].format(s=seed)


def plain_tag(scale, seed):
    return SCALES_032[scale]["plain"].format(s=seed)


def _fallback_dir(cfg, adapter_dir):
    source_tag = os.path.basename(os.path.dirname(adapter_dir))
    return os.path.join(cfg.output_root, "checkpoints",
                        f"{cfg.experiment}_{cfg.scale.name}_{source_tag}_merged_base")


def prepare_sft_start(cfg, result_dir, sft_out_dir, merge_fn=None, log=print):
    from train.model_utils import merge_adapter
    from train.sft import resolve_sft_start

    if not cfg.train.sft_from_adapter:
        raise IntegrityError("032 requires train.sft_from_adapter")
    adapter_dir = cfg.train.sft_from_adapter
    if not os.path.isabs(adapter_dir):
        adapter_dir = os.path.join(cfg.output_root, adapter_dir)
    adapter_hashes = adapter_hashes_of(adapter_dir)

    # reuse the parent merge
    parent_merged = os.path.join(os.path.dirname(adapter_dir), "sft_merged")
    if os.path.exists(os.path.join(parent_merged, "config.json")):
        check_merged_complete(parent_merged)
        start_dir = parent_merged
        start_kind = "parent_merged"
    # build a fallback
    else:
        if os.path.isdir(parent_merged) and os.listdir(parent_merged):
            raise IntegrityError(f"{parent_merged}: parent merge is incomplete")
        start_dir = _fallback_dir(cfg, adapter_dir)
        build = merge_fn or (lambda: merge_adapter(cfg, adapter_dir, start_dir))
        ensure_bound_merge(adapter_dir, start_dir, sft_out_dir, build, log=log)
        start_kind = "fallback_merged"

    check_merged_complete(start_dir)
    # check the stored start against the path the training code resolves
    resolved, base_kind, base_value = resolve_sft_start(cfg)
    if os.path.abspath(resolved) != os.path.abspath(start_dir):
        raise IntegrityError(f"resolved SFT start changed: {resolved} vs {start_dir}")
    if base_kind != "local":
        raise IntegrityError(f"032 resolved start must be local, got {base_kind}")

    binding = {
        "adapter_dir": os.path.abspath(adapter_dir),
        "adapter_sha256": adapter_hashes,
        "resolved_start": os.path.abspath(start_dir),
        "resolved_start_sha256": merged_hashes(start_dir),
        "resolved_base_kind": base_kind,
        "resolved_base_value": base_value,
        "start_kind": start_kind,
    }
    os.makedirs(result_dir, exist_ok=True)
    binding_path = os.path.join(result_dir, START_BINDING)
    if os.path.exists(binding_path):
        if read_json(binding_path) != binding:
            raise IntegrityError(f"{binding_path}: resolved start binding changed")
    else:
        atomic_write_json(binding_path, binding)
    return binding


def verify_sft_start(cfg, result_dir, sft_out_dir, log=lambda *_: None):
    binding = prepare_sft_start(cfg, result_dir, sft_out_dir, log=log)
    binding_path = os.path.join(result_dir, START_BINDING)
    if read_json(binding_path) != binding:
        raise IntegrityError(f"{binding_path}: binding no longer matches the resolved start")
    return binding


def verify_source_lineage(cfg, result_dir, sft_out_dir, parent_manifest_path,
                          log=lambda *_: None):
    source_adapter = cfg.train.sft_from_adapter
    if not os.path.isabs(source_adapter):
        source_adapter = os.path.join(cfg.output_root, source_adapter)
    parent_merged = os.path.join(os.path.dirname(source_adapter), "sft_merged")
    if not os.path.exists(os.path.join(parent_merged, "config.json")):
        raise IntegrityError(f"{parent_merged}: historical parent merge is missing")
    binding = verify_sft_start(cfg, result_dir, sft_out_dir, log=log)
    if binding.get("start_kind") != "parent_merged":
        raise IntegrityError("032 v6 requires the historical parent merge")

    parent_manifest = read_json(parent_manifest_path)
    parent_tag = os.path.basename(os.path.dirname(parent_manifest_path))
    if parent_manifest.get("run_tag") != parent_tag:
        raise IntegrityError(f"{parent_manifest_path}: parent run tag differs")
    if parent_manifest.get("seed") != cfg.train.seed:
        raise IntegrityError(f"{parent_manifest_path}: parent seed differs")

    parent_root = os.path.dirname(binding["adapter_dir"])
    grpo_base_path = os.path.join(parent_root, "grpo", "adapter_base.json")
    if not os.path.exists(grpo_base_path):
        raise IntegrityError(f"{grpo_base_path}: parent GRPO base record is missing")
    grpo_base = read_json(grpo_base_path)
    expected_base = {"kind": "local", "value": binding["resolved_base_value"]}
    if grpo_base != expected_base:
        raise IntegrityError(f"{grpo_base_path}: parent GRPO did not use this merged start")

    lineage = {
        **binding,
        "version": 6,
        "binding_level": "path_and_provenance",
        "byte_reconstruction": False,
        "parent_manifest": os.path.abspath(parent_manifest_path),
        "parent_manifest_sha256": sha256(parent_manifest_path),
        "parent_grpo_adapter_base": os.path.abspath(grpo_base_path),
        "parent_grpo_adapter_base_sha256": sha256(grpo_base_path),
        "parent_grpo_base": grpo_base,
    }
    record_path = os.path.join(result_dir, LINEAGE_RECORD)
    if os.path.exists(record_path):
        if read_json(record_path) != lineage:
            raise IntegrityError(f"{record_path}: source lineage changed")
    else:
        atomic_write_json(record_path, lineage)
    return lineage


def append_failed_gate_contact_correction(ledger_path):
    rows = read_jsonl(ledger_path) if os.path.exists(ledger_path) else []
    if any(row.get("correction_for") == FAILED_GATE_CORRECTION for row in rows):
        return False
    correction = {
        "run_tag": arm_tag("B05", 1),
        "condition": "source_identity_gate_contact_correction",
        "timings_sec": {},
        "train_sec": 0,
        "train_steps": 0,
        "eval_items": {"source_gate": 1884},
        "evaluated": True,
        "committed": False,
        "completed": False,
        "correction_for": FAILED_GATE_CORRECTION,
        "note": "contact correction only; compute remains in the original failed row",
    }
    os.makedirs(os.path.dirname(ledger_path) or ".", exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(correction) + "\n")
    return True


def source_gate_state(result_dir, binding, archived_preds_path):
    manifest_path = os.path.join(result_dir, GATE_MANIFEST)
    if not os.path.exists(manifest_path):
        return "absent"
    manifest = read_json(manifest_path)
    if manifest.get("completed") is not True:
        raise IntegrityError(f"{manifest_path}: gate manifest is incomplete")
    if manifest.get("start_binding") != binding:
        raise IntegrityError(f"{manifest_path}: gate is bound to a different training start")
    if not os.path.exists(archived_preds_path):
        raise IntegrityError(f"{archived_preds_path}: archived predictions are missing")
    if manifest.get("archived_preds_sha256") != sha256(archived_preds_path):
        raise IntegrityError(f"{archived_preds_path}: archived predictions changed")
    for name in GATE_FILES:
        path = os.path.join(result_dir, name)
        if not os.path.exists(path) or manifest.get("sha256", {}).get(name) != sha256(path):
            raise IntegrityError(f"{manifest_path}: committed gate file changed: {name}")
    new_rows = read_jsonl(os.path.join(result_dir, GATE_FILES[1]))
    archived_rows = read_jsonl(archived_preds_path)
    stats = identity_check(new_rows, archived_rows)
    if manifest.get("stats") != stats or manifest.get("row_alignment") is not True:
        raise IntegrityError(f"{manifest_path}: gate statistics changed")
    if manifest.get("review_below_exact_agreement") != (stats["agreement"] < 1.0):
        raise IntegrityError(f"{manifest_path}: exact-agreement review flag changed")
    return "committed"


def _archive_files(result_dir, names, label):
    found = [name for name in names if os.path.exists(os.path.join(result_dir, name))]
    if not found:
        return []
    archive = os.path.join(result_dir, f"{label}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(archive, exist_ok=True)
    for name in found:
        shutil.move(os.path.join(result_dir, name), os.path.join(archive, name))
    return found


def commit_source_gate(result_dir, scratch_dir, binding, archived_preds_path, extra=None,
                       log=print):
    state = source_gate_state(result_dir, binding, archived_preds_path)
    if state == "committed":
        log("  source gate already committed")
        return read_json(os.path.join(result_dir, GATE_MANIFEST))

    metrics_path = os.path.join(scratch_dir, GATE_FILES[0])
    preds_path = os.path.join(scratch_dir, GATE_FILES[1])
    if not os.path.exists(metrics_path) or not os.path.exists(preds_path):
        raise IntegrityError(f"{scratch_dir}: source gate output is incomplete")
    metrics = read_json(metrics_path)
    new_rows = read_jsonl(preds_path)
    archived_rows = read_jsonl(archived_preds_path)
    if metrics.get("n_items") != 1884 or len(new_rows) != 1884 or len(archived_rows) != 1884:
        raise IntegrityError("source gate must contain 1884 aligned rows")
    stats = identity_check(new_rows, archived_rows)

    os.makedirs(result_dir, exist_ok=True)
    orphans = _archive_files(result_dir, GATE_FILES + (GATE_MANIFEST,), "orphaned_source_gate")
    if orphans:
        log(f"  archived source gate orphans: {orphans}")
    atomic_write_json(os.path.join(result_dir, GATE_FILES[0]), metrics)
    stage_into(result_dir, preds_path, GATE_FILES[1])
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "package": "032_wsft",
        "completed": True,
        "start_binding": binding,
        "archived_preds": os.path.abspath(archived_preds_path),
        "archived_preds_sha256": sha256(archived_preds_path),
        "row_alignment": True,
        "thresholds": {"min_agreement": 0.99, "max_acc_diff": 0.001},
        "stats": stats,
        "review_below_exact_agreement": stats["agreement"] < 1.0,
        "sha256": {name: sha256(os.path.join(result_dir, name)) for name in GATE_FILES},
    }
    manifest.update(extra or {})
    # write the manifest last so only a complete gate can be reused
    atomic_write_json(os.path.join(result_dir, GATE_MANIFEST), manifest)
    source_gate_state(result_dir, binding, archived_preds_path)
    log(f"  source gate committed: {stats}")
    return manifest


def training_facts(cfg, result_dir, adapter_dir, binding):
    adapter_hashes = adapter_hashes_of(adapter_dir)
    if read_text(os.path.join(adapter_dir, "config_hash.txt")).strip() != config_hash(cfg):
        raise IntegrityError(f"{adapter_dir}: training config hash differs")
    expected_base = os.path.relpath(binding["resolved_start"], cfg.output_root)
    adapter_base = read_json(os.path.join(adapter_dir, "adapter_base.json"))
    if adapter_base != {"kind": "local", "value": expected_base}:
        raise IntegrityError(f"{adapter_dir}: adapter base does not match the gated start")

    # require one complete fresh schedule before accepting the endpoint
    history_path = os.path.join(result_dir, "sft_trainlog.json")
    history = read_json(history_path)
    rows = [row for row in history
            if row.get("step") is not None and row.get("learning_rate") is not None
            and row.get("loss") is not None]
    counts = Counter(int(row["step"]) for row in rows)
    expected_steps = set(range(1, cfg.train.max_steps + 1))
    if set(counts) != expected_steps or any(count != 1 for count in counts.values()):
        raise IntegrityError(f"{history_path}: SFT steps are missing or duplicated")
    learning_rates = [float(row["learning_rate"])
                      for row in sorted(rows, key=lambda row: int(row["step"]))]
    first_floor = cfg.train.sft_lr * (cfg.train.max_steps - 1) / cfg.train.max_steps
    if not first_floor - 1e-12 <= learning_rates[0] <= cfg.train.sft_lr + 1e-12:
        raise IntegrityError(f"{history_path}: learning rate did not start near the peak")
    if any(b > a + 1e-12 for a, b in zip(learning_rates, learning_rates[1:])):
        raise IntegrityError(f"{history_path}: learning rate increased")
    if learning_rates[-1] > cfg.train.sft_lr / cfg.train.max_steps + 1e-12:
        raise IntegrityError(f"{history_path}: final learning rate is not near zero")

    probe_metrics = read_json(os.path.join(result_dir, "sft_logprob.json"))
    probe_rows = read_jsonl(os.path.join(result_dir, "sft_logprob_preds.jsonl"))
    if probe_metrics.get("n_items") != 1884 or len(probe_rows) != 1884:
        raise IntegrityError(f"{result_dir}: endpoint probe is incomplete")
    manifest = read_json(os.path.join(result_dir, "manifest.json"))
    if manifest.get("seed") != cfg.train.seed:
        raise IntegrityError(f"{result_dir}: manifest seed differs")
    return {
        "steps": cfg.train.max_steps,
        "adapter_sha256": adapter_hashes,
        "adapter_base": adapter_base,
        "trainlog_sha256": sha256(history_path),
        "probe_sha256": {
            "sft_logprob.json": sha256(os.path.join(result_dir, "sft_logprob.json")),
            "sft_logprob_preds.jsonl": sha256(os.path.join(result_dir, "sft_logprob_preds.jsonl")),
        },
        "first_learning_rate": learning_rates[0],
        "final_learning_rate": learning_rates[-1],
    }


def endpoint_state(result_dir, adapter_dir):
    manifest_path = os.path.join(result_dir, ENDPOINT_MANIFEST)
    if not os.path.exists(manifest_path):
        return "absent"
    manifest = read_json(manifest_path)
    if manifest.get("completed") is not True:
        raise IntegrityError(f"{manifest_path}: endpoint manifest is incomplete")
    for name in ENDPOINT_FILES:
        path = os.path.join(result_dir, name)
        if not os.path.exists(path) or manifest.get("sha256", {}).get(name) != sha256(path):
            raise IntegrityError(f"{manifest_path}: committed endpoint file changed: {name}")
    if manifest.get("adapter_sha256") != adapter_hashes_of(adapter_dir):
        raise IntegrityError(f"{manifest_path}: endpoint is bound to another adapter")
    for name in ("sft_logprob.json", "sft_logprob_preds.jsonl"):
        path = os.path.join(result_dir, name)
        if not os.path.exists(path) or manifest.get("probe_sha256", {}).get(name) != sha256(path):
            raise IntegrityError(f"{manifest_path}: endpoint probe changed: {name}")
    summary_path = os.path.join(result_dir, "summary.json")
    if manifest.get("summary_sha256") != sha256(summary_path):
        raise IntegrityError(f"{manifest_path}: summary changed after endpoint commit")
    return "committed"


def commit_sft_endpoint(result_dir, scratch_dir, adapter_dir, extra=None, log=print):
    if endpoint_state(result_dir, adapter_dir) == "committed":
        log("  SFT endpoint already committed")
        return read_json(os.path.join(result_dir, ENDPOINT_MANIFEST))

    metrics_path = os.path.join(scratch_dir, "sft_eval.json")
    preds_path = os.path.join(scratch_dir, "sft_preds.jsonl")
    if not os.path.exists(metrics_path) or not os.path.exists(preds_path):
        raise IntegrityError(f"{scratch_dir}: generative endpoint output is incomplete")
    metrics = read_json(metrics_path)
    gen_rows = read_jsonl(preds_path)
    probe_rows = read_jsonl(os.path.join(result_dir, "sft_logprob_preds.jsonl"))
    if metrics.get("n_items") != 1884 or len(gen_rows) != 1884:
        raise IntegrityError("generative endpoint must contain 1884 rows")
    if len(probe_rows) != 1884 or any(
            a["subject"] != b["subject"] or a["gold"] != b["gold"]
            for a, b in zip(gen_rows, probe_rows)):
        raise IntegrityError("generative and probe endpoint rows do not align")

    os.makedirs(result_dir, exist_ok=True)
    orphans = _archive_files(result_dir, ENDPOINT_FILES + (ENDPOINT_MANIFEST,),
                             "orphaned_sft_endpoint")
    if orphans:
        log(f"  archived endpoint orphans: {orphans}")
    atomic_write_json(os.path.join(result_dir, "sft_eval.json"), metrics)
    stage_into(result_dir, preds_path, "sft_preds.jsonl")
    summary_path = os.path.join(result_dir, "summary.json")
    probe_metrics = read_json(os.path.join(result_dir, "sft_logprob.json"))
    completed_fields = complete_summary_nulls(summary_path, metrics["per_tier_acc"],
                                               probe_metrics["per_tier_acc"])
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "package": "032_wsft",
        "condition": "sft",
        "completed": True,
        "adapter_sha256": adapter_hashes_of(adapter_dir),
        "sha256": {name: sha256(os.path.join(result_dir, name)) for name in ENDPOINT_FILES},
        "probe_sha256": {
            name: sha256(os.path.join(result_dir, name))
            for name in ("sft_logprob.json", "sft_logprob_preds.jsonl")
        },
        "summary_sha256": sha256(summary_path),
        "summary_fields_completed": completed_fields,
    }
    manifest.update(extra or {})
    atomic_write_json(os.path.join(result_dir, ENDPOINT_MANIFEST), manifest)
    endpoint_state(result_dir, adapter_dir)
    log("  SFT endpoint committed")
    return manifest


def arm_state(cfg, result_dir, adapter_dir, binding, archived_preds_path):
    record_path = os.path.join(result_dir, ARM_COMPLETE)
    if not os.path.exists(record_path):
        return "absent"
    record = read_json(record_path)
    if record.get("completed") is not True:
        raise IntegrityError(f"{record_path}: arm record is incomplete")
    source_gate_state(result_dir, binding, archived_preds_path)
    endpoint_state(result_dir, adapter_dir)
    facts = training_facts(cfg, result_dir, adapter_dir, binding)
    if record.get("training") != facts:
        raise IntegrityError(f"{record_path}: completed training artifacts changed")
    for name in ("summary.json", "manifest.json", GATE_MANIFEST, ENDPOINT_MANIFEST):
        path = os.path.join(result_dir, name)
        if record.get("sha256", {}).get(name) != sha256(path):
            raise IntegrityError(f"{record_path}: completed arm file changed: {name}")
    return "committed"


def write_arm_complete(cfg, result_dir, adapter_dir, binding, archived_preds_path):
    source_gate_state(result_dir, binding, archived_preds_path)
    endpoint_state(result_dir, adapter_dir)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "package": "032_wsft",
        "completed": True,
        "training": training_facts(cfg, result_dir, adapter_dir, binding),
        "sha256": {
            name: sha256(os.path.join(result_dir, name))
            for name in ("summary.json", "manifest.json", GATE_MANIFEST, ENDPOINT_MANIFEST)
        },
    }
    atomic_write_json(os.path.join(result_dir, ARM_COMPLETE), record)
    arm_state(cfg, result_dir, adapter_dir, binding, archived_preds_path)
    return record


def prepopulate_scale_base(output_root, scale, destination):
    source = os.path.join(output_root, SCALES_032[scale]["base_eval"])
    return prepopulate_base_evals(source, destination)
