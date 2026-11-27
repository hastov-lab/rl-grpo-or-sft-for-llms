# experiment 035 bindings, pilot gates and artifact transactions
import os
import shutil
from collections import Counter
from contextlib import nullcontext
from datetime import datetime

from checkpoint import config_hash
from file_utils import read_json, read_text
from scripts.s12_eval import (ADAPTER_FILES, IntegrityError, atomic_write_json,
                              read_jsonl, sha256, stage_into)
from scripts.s28_compose import (adapter_hashes_of, check_merged_complete,
                                 merged_hashes)


SCALES_035 = {
    "B3": {"slug": "b3", "source": "003_COLAB_FULL_B3_sft_grpo_seed{s}",
           "source_config": "configs/003_colab_full_b3_seed{s}.yaml"},
    "B15": {"slug": "b15", "source": "009_COLAB_FULL_B15_sft_grpo_seed{s}",
            "source_config": "configs/009_colab_full_b15_seed{s}.yaml"},
}

RUNGS_035 = {
    "A": {"batch": 8, "accum": 1, "checkpointing": False},
    "B": {"batch": 4, "accum": 2, "checkpointing": False},
    "C": {"batch": 4, "accum": 2, "checkpointing": True},
    "D": {"batch": 2, "accum": 4, "checkpointing": True},
}
PILOT_RUNGS_035 = {
    "A_GC": {"batch": 8, "accum": 1, "checkpointing": True},
}
SCIENCE_RUNGS_035 = tuple(RUNGS_035)

PARITY_TOLERANCE = 1.0e-3
MAX_AUDIT_ELEMENTS = 1_000_000
TRAINING_COMPLETE = "035_training_complete.json"
EXECUTION_BINDING = "035_execution_binding.json"
ENDPOINT_MANIFEST = "035_eval_manifest.json"
EXECUTION_OVERRIDE = "035_execution_override.json"
ALLOCATOR_RETRY = "035_allocator_retry.json"
ALLOCATOR_RETRY_PENDING = "035_allocator_retry_pending.json"
AGC_SCIENCE = "035_agc_science.json"
AGC_SCIENCE_PENDING = "035_agc_science_pending.json"
AGC_LORA_CONTROL = "035_agc_lora_control.json"
B15_SCALE_CHECK = "035_b15_scale_check.json"
B15_PAIR_SETTLED = "035_b15_pair_settled.json"
B15_WANDB_RUNS = "035_wandb_runs"
B15_DRIVE_RESUME = "035_b15_lora_drive_resume.json"
B15_DRIVE_STATUS = "035_b15_lora_drive_disconnect_status.json"
B15_DRIVE_ERROR = "Transport endpoint is not connected"
B15_RESUME_STATE = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
)
ALLOCATOR_CONFIG = "expandable_segments:True"
AGC_FULL_A1_ERROR = (
    "sentinel abort: a1: over 30% of rollouts truncated in the first 50 steps. "
    "raise the completion cap or mask truncated completions"
)
ENDPOINT_FILES = ("grpo_preds.jsonl", "grpo_eval.json",
                  "grpo_logprob_preds.jsonl", "grpo_logprob.json")


def config_path(scale, seed, parameter_mode):
    slug = SCALES_035[scale]["slug"]
    return f"configs/035_colab_{slug}_{parameter_mode}_seed{seed}.yaml"


def source_tag(scale, seed):
    return SCALES_035[scale]["source"].format(s=seed)


def arm_tag(scale, seed, parameter_mode):
    condition = "matched_lora_grpo" if parameter_mode == "lora" else "matched_full_grpo"
    return f"035_COLAB_FULL_{scale}_{condition}_seed{seed}"


def source_start(output_root, scale, seed):
    return os.path.join(output_root, "checkpoints", source_tag(scale, seed), "sft_merged")


def execution_settings(rung):
    if rung in RUNGS_035:
        return RUNGS_035[rung]
    if rung in PILOT_RUNGS_035:
        return PILOT_RUNGS_035[rung]
    raise ValueError(f"bad experiment 035 rung: {rung}")


def apply_rung(cfg, rung):
    values = execution_settings(rung)
    # batch times accumulation stays at N, the split is only a memory knob
    if values["batch"] * values["accum"] != cfg.train.num_generations:
        raise IntegrityError("experiment 035 rung changes the effective generation group")
    cfg.exec.grpo_batch_size = values["batch"]
    cfg.exec.grpo_grad_accum = values["accum"]
    cfg.exec.gradient_checkpointing = values["checkpointing"]
    cfg.exec.allow_oom_fallback = False
    return values


def bind_execution(result_dir, cfg, start_record, rung, task, environment):
    # record the execution fields the config hash leaves out, before any resume
    payload = {
        "experiment": "035",
        "task": task,
        "scale": cfg.scale.name,
        "seed": cfg.train.seed,
        "parameter_mode": cfg.train.grpo_parameter_mode,
        "config_hash": config_hash(cfg),
        "start_record": start_record,
        "rung": rung,
        "execution": execution_settings(rung),
        "environment": environment,
    }
    os.makedirs(result_dir, exist_ok=True)
    return _write_once(os.path.join(result_dir, EXECUTION_BINDING), payload)


def _write_once(path, payload):
    if os.path.exists(path):
        if read_json(path) != payload:
            raise IntegrityError(f"{path}: committed record changed")
        return read_json(path)
    atomic_write_json(path, payload)
    return payload


def capture_wandb_run(record_dir, context, step):
    import wandb

    active = wandb.run
    if active is None:
        raise IntegrityError("experiment 035b W&B run is not active")
    run_id = str(getattr(active, "id", "") or "")
    project = str(getattr(active, "project", "") or "")
    run_name = str(getattr(active, "name", "") or "")
    entity = str(getattr(active, "entity", "") or "") or None
    # a resumed session can silently attach to a different run
    if (not run_id or not run_id.replace("-", "").replace("_", "").isalnum()
            or project != context["project"] or run_name != context["run_name"]):
        raise IntegrityError("experiment 035b W&B identity differs from the worker")
    os.makedirs(record_dir, exist_ok=True)
    path = os.path.join(record_dir, f"{run_id}.json")
    stable = {
        "completed": True,
        "experiment": "035b",
        "scale": context["scale"],
        "seed": context["seed"],
        "parameter_mode": context["parameter_mode"],
        "rung": context["rung"],
        "project": project,
        "run_id": run_id,
        "run_name": run_name,
        "entity": entity,
        "config_hash": context["config_hash"],
        "execution_binding_sha256": context["execution_binding_sha256"],
    }
    if os.path.exists(path):
        record = read_json(path)
        if any(record.get(key) != value for key, value in stable.items()):
            raise IntegrityError("experiment 035b W&B run record changed")
        return record
    stable["first_recorded_step"] = int(step)
    return _write_once(path, stable)


def bind_start(cfg, record_path, require_032=True):
    from config import load_config

    # bind the merged model that the historical GRPO stage actually used
    scale = cfg.scale.name
    seed = cfg.train.seed
    if scale not in SCALES_035:
        raise IntegrityError(f"experiment 035 does not register scale {scale}")
    start = source_start(cfg.output_root, scale, seed)
    check_merged_complete(start)

    parent_tag = source_tag(scale, seed)
    parent_cfg = load_config(SCALES_035[scale]["source_config"].format(s=seed))
    parent_manifest_path = os.path.join(
        cfg.output_root, "results", parent_tag, "manifest.json")
    parent_manifest = read_json(parent_manifest_path)
    if (parent_cfg.scale != cfg.scale or parent_cfg.train.seed != seed
            or parent_manifest.get("run_tag") != parent_tag
            or parent_manifest.get("seed") != seed):
        raise IntegrityError(f"{parent_tag}: source identity differs from experiment 035")

    base_path = os.path.join(
        cfg.output_root, "checkpoints", parent_tag, "grpo", "adapter_base.json")
    base = read_json(base_path)
    expected_value = os.path.relpath(start, cfg.output_root)
    if base != {"kind": "local", "value": expected_value}:
        raise IntegrityError(f"{base_path}: historical GRPO used a different merged start")

    current_hashes = merged_hashes(start)
    lineage = None
    if seed in (1, 2) and require_032:
        lineage_path = os.path.join(
            cfg.output_root, "results",
            f"032_wsft_COLAB_FULL_{scale}_sft_only_seed{seed}",
            "source_lineage_v6.json")
        lineage = read_json(lineage_path)
        if (os.path.abspath(lineage.get("resolved_start", "")) != os.path.abspath(start)
                or lineage.get("resolved_start_sha256") != current_hashes):
            raise IntegrityError(f"{lineage_path}: source hashes differ from the current archive")

    record = {
        "experiment": "035",
        "scale": scale,
        "seed": seed,
        "source_tag": parent_tag,
        "source_config": SCALES_035[scale]["source_config"].format(s=seed),
        "source_manifest": os.path.abspath(parent_manifest_path),
        "source_manifest_sha256": sha256(parent_manifest_path),
        "historical_adapter_base": os.path.abspath(base_path),
        "historical_adapter_base_sha256": sha256(base_path),
        "start": os.path.abspath(start),
        "start_sha256": current_hashes,
        "binding_level": ("current_hashes_and_032_lineage" if seed in (1, 2)
                          else "path_provenance_and_current_hashes"),
        "lineage_sha256": sha256(lineage_path) if lineage is not None else None,
    }
    os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
    return _write_once(record_path, record)


def pilot_passes(report):
    # unchanged weights fail a rung even when it fits in memory
    audit = report.get("precision_audit", {})
    changes = audit.get("module_changes", {})
    return bool(
        report.get("completed_updates") == 30
        and report.get("finite") is True
        and report.get("storage_dtype") == "float32"
        and report.get("compute_dtype") == "bfloat16"
        and report.get("gradient_dtype") == "float32"
        and report.get("optimizer_state_dtypes") == ["float32"]
        and audit.get("accelerator_mixed_precision") == "bf16"
        and report.get("checkpoint_bytes", 0) > 0
        and audit.get("reference_gate", {}).get("verified") is True
        and changes
        and all(row.get("changed_elements", 0) > 0 for row in changes.values())
        and report.get("peak_reserved_fraction", 1.0) <= 0.95
    )


def commit_pilot_report(path, report):
    payload = dict(report)
    payload["passed"] = pilot_passes(payload)
    payload["scientific_metrics_used"] = False
    return _write_once(path, payload)


def commit_scale_selection(output_root):
    # select a scale from technical pilot verdicts only
    root = os.path.join(output_root, "results", "035_technical")
    selection_path = os.path.join(root, "035_scale_selection.json")
    if os.path.exists(selection_path):
        return read_json(selection_path)

    def reports(scale):
        out = []
        for rung in SCIENCE_RUNGS_035:
            path = os.path.join(root, "pilots", scale, rung, "pilot_report.json")
            if os.path.exists(path):
                out.append((rung, read_json(path)))
        return out

    b3 = reports("B3")
    passed = [(rung, row) for rung, row in b3 if row.get("passed") is True]
    if passed:
        scale, rung = "B3", passed[0][0]
    else:
        if len(b3) != len(SCIENCE_RUNGS_035) or any(
                row.get("passed") is not False for _, row in b3):
            raise IntegrityError("3B pilot ladder is not complete")
        b15 = reports("B15")
        passed = [(rung, row) for rung, row in b15 if row.get("passed") is True]
        if not passed:
            if len(b15) == len(SCIENCE_RUNGS_035):
                raise IntegrityError("both experiment 035 pilot ladders failed")
            raise IntegrityError("1.5B fallback pilot ladder is not complete")
        scale, rung = "B15", passed[0][0]

    payload = {"completed": True, "selected_scale": scale,
               "pilot_rung": rung, "selection_basis": "technical_pilot_only"}
    os.makedirs(root, exist_ok=True)
    return _write_once(selection_path, payload)


def commit_execution_override(output_root, selection, requested_rung):
    if (selection.get("selected_scale") != "B3"
            or selection.get("pilot_rung") != "C"
            or requested_rung != "A"):
        raise IntegrityError("experiment 035 execution override differs from plan v7")
    root = os.path.join(output_root, "results", "035_technical")
    path = os.path.join(root, EXECUTION_OVERRIDE)
    a_report = os.path.join(root, "pilots", "B3", "A", "pilot_report.json")
    c_report = os.path.join(root, "pilots", "B3", "C", "pilot_report.json")
    a_row = read_json(a_report)
    c_row = read_json(c_report)
    if (a_row.get("passed") is not False or c_row.get("passed") is not True
            or a_row.get("peak_reserved_fraction", 0) < 0.99
            or c_row.get("peak_reserved_fraction", 1) > 0.95):
        raise IntegrityError("experiment 035 pilot record does not support the override")
    payload = {
        "completed": True,
        "scale": "B3",
        "pilot_selected_rung": "C",
        "requested_rung": "A",
        "automatic_fallback": False,
        "selection_basis": "technical_override_before_scientific_training",
        "registered_date": "2026-08-17",
        "pilot_reports_sha256": {"A": sha256(a_report), "C": sha256(c_report)},
    }
    if os.path.exists(path):
        return _write_once(path, payload)
    if os.path.exists(os.path.join(root, "execution_lock.json")):
        raise IntegrityError("execution override must precede the scientific lock")
    return _write_once(path, payload)


def _move_once(source, destination, required=True):
    source_exists = os.path.exists(source)
    destination_exists = os.path.exists(destination)
    if source_exists and destination_exists:
        raise IntegrityError(f"both archive paths exist: {source} and {destination}")
    if source_exists:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(source, destination)
    elif required and not destination_exists:
        raise IntegrityError(f"archive source is missing: {source}")


def prepare_allocator_retry(output_root, selection, override):
    if (selection.get("selected_scale") != "B3"
            or selection.get("pilot_rung") != "C"
            or override.get("requested_rung") != "A"
            or override.get("automatic_fallback") is not False):
        raise IntegrityError("experiment 035 allocator retry differs from plan v8")
    root = os.path.join(output_root, "results", "035_technical")
    retry_path = os.path.join(root, ALLOCATOR_RETRY)
    pending_path = os.path.join(root, ALLOCATOR_RETRY_PENDING)
    tag = arm_tag("B3", 0, "full")
    result_dir = os.path.join(output_root, "results", tag)
    checkpoint_dir = os.path.join(output_root, "checkpoints", tag)
    parity_dir = os.path.join(root, "parity", "B3", "seed0", "A")
    archive = os.path.join(
        output_root, "failed_attempts", "035", "B3_seed0_full_A_default_allocator")
    destinations = {
        "results": os.path.join(archive, "results"),
        "checkpoints": os.path.join(archive, "checkpoints"),
        "parity": os.path.join(archive, "parity"),
    }

    if os.path.exists(retry_path):
        record = read_json(retry_path)
        if (record.get("completed") is not True
                or record.get("scale") != "B3" or record.get("seed") != 0
                or record.get("parameter_mode") != "full" or record.get("rung") != "A"
                or record.get("allocator_config") != ALLOCATOR_CONFIG
                or record.get("clean_restart") is not True
                or record.get("automatic_fallback") is not False
                or record.get("archive") != os.path.abspath(archive)):
            raise IntegrityError("experiment 035 allocator retry record changed")
        if os.path.exists(pending_path):
            if read_json(pending_path) != record:
                raise IntegrityError("experiment 035 allocator retry pending record changed")
            os.remove(pending_path)
        if (not os.path.isdir(destinations["results"])
                or not os.path.isdir(destinations["parity"])):
            raise IntegrityError("experiment 035 allocator retry archive is incomplete")
        if (sha256(os.path.join(destinations["results"], "worker_status.json"))
                != record.get("failed_status_sha256")
                or sha256(os.path.join(destinations["results"], EXECUTION_BINDING))
                != record.get("failed_binding_sha256")):
            raise IntegrityError("experiment 035 allocator retry archive changed")
        active_complete = os.path.join(result_dir, TRAINING_COMPLETE)
        active_status = os.path.join(result_dir, "worker_status.json")
        if os.path.exists(active_status) and not os.path.exists(active_complete):
            status = read_json(active_status)
            if (status.get("cuda_oom") is True
                    and status.get("allocator_config") == ALLOCATOR_CONFIG):
                raise IntegrityError("experiment 035 allocator retry already OOMed")
        return record

    if os.path.exists(os.path.join(root, "execution_lock.json")):
        raise IntegrityError("allocator retry must precede the scientific lock")
    if os.path.exists(pending_path):
        payload = read_json(pending_path)
    else:
        status_path = os.path.join(result_dir, "worker_status.json")
        binding_path = os.path.join(result_dir, EXECUTION_BINDING)
        status = read_json(status_path)
        binding = read_json(binding_path)
        if (status.get("task") != "train" or status.get("rung") != "A"
                or status.get("cuda_oom") is not True
                or status.get("allocator_config") not in (None, "")):
            raise IntegrityError("experiment 035 retry requires the recorded rung-A OOM")
        if (binding.get("rung") != "A"
                or binding.get("environment", {}).get("allocator_config") not in (None, "")):
            raise IntegrityError("experiment 035 failed binding differs from rung A")
        # only the recorded OOM may be archived, never a completed arm
        if (os.path.exists(os.path.join(result_dir, TRAINING_COMPLETE))
                or os.path.exists(os.path.join(result_dir, ENDPOINT_MANIFEST))):
            raise IntegrityError("a completed experiment 035 arm cannot be archived")
        if os.path.exists(archive):
            raise IntegrityError(f"failed-attempt archive already exists: {archive}")
        payload = {
            "completed": True,
            "scale": "B3",
            "seed": 0,
            "parameter_mode": "full",
            "rung": "A",
            "allocator_config": ALLOCATOR_CONFIG,
            "clean_restart": True,
            "automatic_fallback": False,
            "registered_date": "2026-08-17",
            "failed_status_sha256": sha256(status_path),
            "failed_binding_sha256": sha256(binding_path),
            "archive": os.path.abspath(archive),
        }
        # keep a pending record so the move can repeat after a disconnect
        atomic_write_json(pending_path, payload)

    _move_once(result_dir, destinations["results"])
    _move_once(checkpoint_dir, destinations["checkpoints"], required=False)
    _move_once(parity_dir, destinations["parity"])
    if (sha256(os.path.join(destinations["results"], "worker_status.json"))
            != payload["failed_status_sha256"]
            or sha256(os.path.join(destinations["results"], EXECUTION_BINDING))
            != payload["failed_binding_sha256"]):
        raise IntegrityError("experiment 035 failed archive hashes differ")
    _write_once(retry_path, payload)
    os.remove(pending_path)
    return payload


def _agc_paths(output_root):
    root = os.path.join(output_root, "results", "035_technical")
    tag = arm_tag("B3", 0, "full")
    archive = os.path.join(
        output_root, "failed_attempts", "035", "B3_seed0_full_A_expandable_allocator")
    return {
        "root": root,
        "record": os.path.join(root, AGC_SCIENCE),
        "pending": os.path.join(root, AGC_SCIENCE_PENDING),
        "pilot": os.path.join(root, "pilots", "B3", "A_GC", "pilot_report.json"),
        "result": os.path.join(output_root, "results", tag),
        "checkpoint": os.path.join(output_root, "checkpoints", tag),
        "parity": os.path.join(root, "parity", "B3", "seed0", "A"),
        "archive": archive,
        "archive_results": os.path.join(archive, "results"),
        "archive_checkpoints": os.path.join(archive, "checkpoints"),
        "archive_parity": os.path.join(archive, "parity"),
    }


def _check_agc_pilot(path):
    pilot = read_json(path)
    # memory and precision decide this, never the reward
    if (pilot.get("passed") is not True or pilot.get("completed_updates") != 30
            or pilot.get("execution") != execution_settings("A_GC")
            or pilot.get("allocator_config") != ALLOCATOR_CONFIG
            or pilot.get("pilot_weights_retained") is not False
            or pilot.get("scientific_metrics_used") is not False
            or pilot.get("peak_reserved_fraction", 1.0) > 0.95):
        raise IntegrityError("experiment 035 A_GC pilot did not pass the registered gates")
    return pilot


def _check_agc_record(record, paths):
    if (record.get("completed") is not True or record.get("scale") != "B3"
            or record.get("seed") != 0 or record.get("rung") != "A_GC"
            or record.get("execution") != execution_settings("A_GC")
            or record.get("allocator_config") != ALLOCATOR_CONFIG
            or record.get("pilot_report_sha256") != sha256(paths["pilot"])
            or record.get("clean_restart") is not True
            or record.get("automatic_fallback") is not False
            or record.get("authorised_parity") != ["lora_seed0", "full_seed0"]
            or record.get("authorised_training") != ["full_seed0"]
            or record.get("authorised_evaluation") != ["full_seed0"]
            or record.get("archive") != os.path.abspath(paths["archive"])):
        raise IntegrityError("experiment 035 A_GC science record changed")


def _check_agc_archive(record, paths):
    if (not os.path.isdir(paths["archive_results"])
            or not os.path.isdir(paths["archive_parity"])
            or sha256(os.path.join(paths["archive_results"], "worker_status.json"))
                != record.get("failed_status_sha256")
            or sha256(os.path.join(paths["archive_results"], EXECUTION_BINDING))
                != record.get("failed_binding_sha256")):
        raise IntegrityError("experiment 035 A_GC failed-attempt archive changed")


def _agc_lora_paths(output_root):
    paths = _agc_paths(output_root)
    paths.update({
        "control": os.path.join(paths["root"], AGC_LORA_CONTROL),
        "full_status": os.path.join(paths["result"], "worker_status.json"),
        "full_binding": os.path.join(paths["result"], EXECUTION_BINDING),
        "pair_parity": os.path.join(
            paths["root"], "parity", "B3", "seed0", "A_GC", "pair_parity.json"),
        "execution_lock": os.path.join(paths["root"], "execution_lock.json"),
    })
    return paths


def check_pair_parity(path, scale):
    record = read_json(path)
    checks = record.get("checks", {})
    expected = {"lora_policy_vs_reference", "full_policy_vs_reference",
                "policy_between_modes", "reference_between_modes"}
    if (record.get("completed") is not True
            or record.get("tolerance") != PARITY_TOLERANCE
            or record.get("lora", {}).get("parameter_mode") != "lora"
            or record.get("full", {}).get("parameter_mode") != "full"
            or record.get("lora", {}).get("scale") != scale
            or record.get("full", {}).get("scale") != scale
            or record.get("lora", {}).get("seed") != 0
            or record.get("full", {}).get("seed") != 0
            or record.get("lora", {}).get("rung") != "A_GC"
            or record.get("full", {}).get("rung") != "A_GC"
            or record.get("lora", {}).get("execution") != execution_settings("A_GC")
            or record.get("full", {}).get("execution") != execution_settings("A_GC")
            or record.get("lora", {}).get("allocator_config") != ALLOCATOR_CONFIG
            or record.get("full", {}).get("allocator_config") != ALLOCATOR_CONFIG
            or set(checks) != expected
            or any(value > PARITY_TOLERANCE for value in checks.values())):
        raise IntegrityError("experiment 035 A_GC pair parity is incomplete")
    return record


def _check_agc_pair_parity(path):
    return check_pair_parity(path, "B3")


def _check_agc_lora_record(record, paths):
    from config import load_config

    lora_hash = config_hash(load_config(config_path("B3", 0, "lora")))
    if (record.get("completed") is not True or record.get("scale") != "B3"
            or record.get("seed") != 0 or record.get("rung") != "A_GC"
            or record.get("execution") != execution_settings("A_GC")
            or record.get("allocator_config") != ALLOCATOR_CONFIG
            or record.get("parameter_mode") != "lora"
            or record.get("lora_config_hash") != lora_hash
            or record.get("source_agc_science_sha256") != sha256(paths["record"])
            or record.get("failed_full_status_sha256") != sha256(paths["full_status"])
            or record.get("failed_full_binding_sha256") != sha256(paths["full_binding"])
            or record.get("pair_parity_sha256") != sha256(paths["pair_parity"])
            or record.get("authorised_training") != ["lora_seed0"]
            or record.get("authorised_evaluation") != ["lora_seed0"]):
        raise IntegrityError("experiment 035 A_GC LoRA control record changed")


def prepare_agc_lora_control(output_root):
    from config import load_config

    paths = _agc_lora_paths(output_root)
    science = read_json(paths["record"])
    _check_agc_pilot(paths["pilot"])
    _check_agc_record(science, paths)
    _check_agc_archive(science, paths)
    parity = _check_agc_pair_parity(paths["pair_parity"])

    status = read_json(paths["full_status"])
    binding = read_json(paths["full_binding"])
    # the control exists only while the full arm keeps its exact A1 stop
    if (status.get("completed") is not False or status.get("task") != "train"
            or status.get("rung") != "A_GC" or status.get("cuda_oom") is not False
            or status.get("error_type") != "RuntimeError"
            or status.get("error_message") != AGC_FULL_A1_ERROR
            or status.get("execution") != execution_settings("A_GC")
            or status.get("allocator_config") != ALLOCATOR_CONFIG):
        raise IntegrityError("experiment 035 LoRA control requires the full A1 stop")
    full_hash = config_hash(load_config(config_path("B3", 0, "full")))
    if (binding.get("parameter_mode") != "full" or binding.get("task") != "train"
            or binding.get("scale") != "B3" or binding.get("seed") != 0
            or binding.get("rung") != "A_GC"
            or binding.get("execution") != execution_settings("A_GC")
            or binding.get("config_hash") != full_hash
            or binding.get("start_record") != parity["full"].get("start_record")
            or binding.get("environment", {}).get("allocator_config")
                != ALLOCATOR_CONFIG):
        raise IntegrityError("experiment 035 full A_GC binding changed")
    if (os.path.exists(os.path.join(paths["result"], TRAINING_COMPLETE))
            or os.path.exists(os.path.join(paths["result"], ENDPOINT_MANIFEST))
            or os.path.exists(paths["execution_lock"])):
        raise IntegrityError("experiment 035 full A_GC arm has endpoint artifacts")

    payload = {
        "completed": True,
        "scale": "B3",
        "seed": 0,
        "rung": "A_GC",
        "execution": execution_settings("A_GC"),
        "allocator_config": ALLOCATOR_CONFIG,
        "parameter_mode": "lora",
        "lora_config_hash": config_hash(load_config(config_path("B3", 0, "lora"))),
        "source_agc_science_sha256": sha256(paths["record"]),
        "failed_full_status_sha256": sha256(paths["full_status"]),
        "failed_full_binding_sha256": sha256(paths["full_binding"]),
        "pair_parity_sha256": sha256(paths["pair_parity"]),
        "authorised_training": ["lora_seed0"],
        "authorised_evaluation": ["lora_seed0"],
        "registered_date": "2026-08-18",
    }
    record = _write_once(paths["control"], payload)
    _check_agc_lora_record(record, paths)
    return record


def prepare_agc_science(output_root, selection, override, retry):
    if (selection.get("selected_scale") != "B3"
            or selection.get("pilot_rung") != "C"
            or override.get("requested_rung") != "A"
            or retry.get("rung") != "A"
            or retry.get("allocator_config") != ALLOCATOR_CONFIG):
        raise IntegrityError("experiment 035 A_GC run differs from plan v10")
    paths = _agc_paths(output_root)
    _check_agc_pilot(paths["pilot"])

    if os.path.exists(paths["record"]):
        record = read_json(paths["record"])
        _check_agc_record(record, paths)
        if os.path.exists(paths["pending"]):
            if read_json(paths["pending"]) != record:
                raise IntegrityError("experiment 035 A_GC pending record changed")
            os.remove(paths["pending"])
        _check_agc_archive(record, paths)
        if os.path.exists(os.path.join(paths["result"], EXECUTION_BINDING)):
            binding = read_json(os.path.join(paths["result"], EXECUTION_BINDING))
            if (binding.get("rung") != "A_GC"
                    or binding.get("environment", {}).get("allocator_config")
                    != ALLOCATOR_CONFIG):
                raise IntegrityError("experiment 035 active arm differs from A_GC")
        return record

    if os.path.exists(os.path.join(paths["root"], "execution_lock.json")):
        raise IntegrityError("A_GC authorisation must precede the scientific lock")
    if os.path.exists(paths["pending"]):
        payload = read_json(paths["pending"])
        _check_agc_record(payload, paths)
    else:
        status_path = os.path.join(paths["result"], "worker_status.json")
        binding_path = os.path.join(paths["result"], EXECUTION_BINDING)
        status = read_json(status_path)
        binding = read_json(binding_path)
        if (status.get("completed") is not False or status.get("task") != "train"
                or status.get("rung") != "A" or status.get("cuda_oom") is not True
                or status.get("allocator_config") != ALLOCATOR_CONFIG):
            raise IntegrityError("A_GC run requires the allocator retry OOM")
        if (binding.get("rung") != "A"
                or binding.get("environment", {}).get("allocator_config")
                != ALLOCATOR_CONFIG):
            raise IntegrityError("experiment 035 failed binding differs from the retry")
        if (os.path.exists(os.path.join(paths["result"], TRAINING_COMPLETE))
                or os.path.exists(os.path.join(paths["result"], ENDPOINT_MANIFEST))):
            raise IntegrityError("a completed experiment 035 arm cannot be archived")
        if os.path.exists(paths["archive"]):
            raise IntegrityError(f"failed-attempt archive already exists: {paths['archive']}")
        payload = {
            "completed": True,
            "scale": "B3",
            "seed": 0,
            "rung": "A_GC",
            "execution": execution_settings("A_GC"),
            "allocator_config": ALLOCATOR_CONFIG,
            "clean_restart": True,
            "automatic_fallback": False,
            "authorised_parity": ["lora_seed0", "full_seed0"],
            "authorised_training": ["full_seed0"],
            "authorised_evaluation": ["full_seed0"],
            "registered_date": "2026-08-18",
            "pilot_report_sha256": sha256(paths["pilot"]),
            "failed_status_sha256": sha256(status_path),
            "failed_binding_sha256": sha256(binding_path),
            "archive": os.path.abspath(paths["archive"]),
        }
        atomic_write_json(paths["pending"], payload)

    _move_once(paths["result"], paths["archive_results"])
    _move_once(paths["checkpoint"], paths["archive_checkpoints"], required=False)
    _move_once(paths["parity"], paths["archive_parity"])
    _check_agc_archive(payload, paths)
    _write_once(paths["record"], payload)
    os.remove(paths["pending"])
    return payload


def check_agc_authorisation(cfg, rung, task):
    b15_path = os.path.join(
        cfg.output_root, "results", "035_technical", B15_SCALE_CHECK)
    if cfg.scale.name == "B15":
        if not os.path.exists(b15_path):
            raise IntegrityError("experiment 035b scale-check record is missing")
        if rung != "A_GC":
            raise IntegrityError("experiment 035b only authorises A_GC")
        return _check_b15_task(cfg, task)
    if os.path.exists(b15_path):
        record = read_json(b15_path)
        _check_b15_scale_record(record, _b15_paths(cfg.output_root))
        raise IntegrityError("experiment 035 B3 package is closed")
    if rung != "A_GC" or task == "pilot":
        return None
    paths = _agc_paths(cfg.output_root)
    record = read_json(paths["record"])
    _check_agc_pilot(paths["pilot"])
    _check_agc_record(record, paths)
    if cfg.scale.name != "B3" or cfg.train.seed != 0:
        raise IntegrityError("experiment 035 A_GC task is not authorised")
    _check_agc_archive(record, paths)
    mode = cfg.train.grpo_parameter_mode
    control_path = os.path.join(paths["root"], AGC_LORA_CONTROL)
    if os.path.exists(control_path) and task in ("train", "eval"):
        control_paths = _agc_lora_paths(cfg.output_root)
        control = read_json(control_path)
        _check_agc_lora_record(control, control_paths)
        _check_agc_pair_parity(control_paths["pair_parity"])
        if mode != "lora":
            raise IntegrityError("experiment 035 full A_GC arm is closed after A1")
        return control
    allowed = task == "parity" and mode in ("lora", "full")
    allowed = allowed or task in ("train", "eval") and mode == "full"
    if not allowed:
        raise IntegrityError("experiment 035 A_GC task is outside the staged run")
    return record


def _b15_paths(output_root):
    root = os.path.join(output_root, "results", "035_technical")
    full_tag = arm_tag("B15", 0, "full")
    lora_tag = arm_tag("B15", 0, "lora")
    b3_lora_tag = arm_tag("B3", 0, "lora")
    return {
        "root": root,
        "record": os.path.join(root, B15_SCALE_CHECK),
        "selection": os.path.join(root, "035_scale_selection.json"),
        "b3_science": os.path.join(root, AGC_SCIENCE),
        "b3_control": os.path.join(root, AGC_LORA_CONTROL),
        "b3_full_status": os.path.join(
            output_root, "results", arm_tag("B3", 0, "full"), "worker_status.json"),
        "b3_full_binding": os.path.join(
            output_root, "results", arm_tag("B3", 0, "full"), EXECUTION_BINDING),
        "b3_lora_result": os.path.join(output_root, "results", b3_lora_tag),
        "b3_lora_training": os.path.join(
            output_root, "results", b3_lora_tag, TRAINING_COMPLETE),
        "b3_lora_endpoint": os.path.join(
            output_root, "results", b3_lora_tag, ENDPOINT_MANIFEST),
        "start": os.path.join(root, "starts", "B15", "seed0.json"),
        "pilot": os.path.join(
            root, "pilots", "B15", "A_GC", "pilot_report.json"),
        "parity": os.path.join(
            root, "parity", "B15", "seed0", "A_GC", "pair_parity.json"),
        "settled": os.path.join(root, B15_PAIR_SETTLED),
        "full_result": os.path.join(output_root, "results", full_tag),
        "full_output": os.path.join(output_root, "checkpoints", full_tag, "grpo"),
        "lora_result": os.path.join(output_root, "results", lora_tag),
        "lora_output": os.path.join(output_root, "checkpoints", lora_tag, "grpo"),
        "lora_drive_resume": os.path.join(root, B15_DRIVE_RESUME),
        "lora_drive_status": os.path.join(root, B15_DRIVE_STATUS),
    }


def _check_closed_b3(paths):
    science_paths = _agc_paths(os.path.dirname(os.path.dirname(paths["root"])))
    science = read_json(paths["b3_science"])
    _check_agc_pilot(science_paths["pilot"])
    _check_agc_record(science, science_paths)
    _check_agc_archive(science, science_paths)

    control_paths = _agc_lora_paths(os.path.dirname(os.path.dirname(paths["root"])))
    control = read_json(paths["b3_control"])
    _check_agc_lora_record(control, control_paths)
    _check_agc_pair_parity(control_paths["pair_parity"])

    status = read_json(paths["b3_full_status"])
    if (status.get("completed") is not False or status.get("task") != "train"
            or status.get("rung") != "A_GC" or status.get("cuda_oom") is not False
            or status.get("error_type") != "RuntimeError"
            or status.get("error_message") != AGC_FULL_A1_ERROR):
        raise IntegrityError("experiment 035 full arm no longer has the exact A1 stop")
    if (os.path.exists(os.path.join(os.path.dirname(paths["b3_full_status"]),
                                    TRAINING_COMPLETE))
            or os.path.exists(os.path.join(os.path.dirname(paths["b3_full_status"]),
                                           ENDPOINT_MANIFEST))):
        raise IntegrityError("experiment 035 full A1 arm gained endpoint artifacts")

    training = read_json(paths["b3_lora_training"])
    if endpoint_state(paths["b3_lora_result"], training) != "committed":
        raise IntegrityError("experiment 035 LoRA endpoint is no longer committed")
    return science, control, training


def _check_b15_scale_record(record, paths):
    from config import load_config

    selection = read_json(paths["selection"])
    if (selection.get("selected_scale") != "B3"
            or selection.get("pilot_rung") != "C"):
        raise IntegrityError("experiment 035 scale selection changed")
    _, _, b3_training = _check_closed_b3(paths)
    start = read_json(paths["start"])
    expected_configs = {
        mode: config_hash(load_config(config_path("B15", 0, mode)))
        for mode in ("lora", "full")
    }
    if (record.get("completed") is not True or record.get("experiment") != "035b"
            or record.get("scale") != "B15" or record.get("seed") != 0
            or record.get("rung") != "A_GC"
            or record.get("execution") != execution_settings("A_GC")
            or record.get("allocator_config") != ALLOCATOR_CONFIG
            or record.get("fixed_roster") != ["full_seed0", "lora_seed0"]
            or record.get("pilot_authorised") != ["full_seed0"]
            or record.get("scientific_requires_passing_pilot") is not True
            or record.get("scale_selection_sha256") != sha256(paths["selection"])
            or record.get("b3_science_sha256") != sha256(paths["b3_science"])
            or record.get("b3_control_sha256") != sha256(paths["b3_control"])
            or record.get("b3_full_status_sha256") != sha256(paths["b3_full_status"])
            or record.get("b3_full_binding_sha256") != sha256(paths["b3_full_binding"])
            or record.get("b3_lora_training_sha256") != sha256(paths["b3_lora_training"])
            or record.get("b3_lora_endpoint_sha256") != sha256(paths["b3_lora_endpoint"])
            or record.get("b15_config_hashes") != expected_configs
            or record.get("b15_start_record") != start
            or record.get("b15_start_record_sha256") != sha256(paths["start"])
            or record.get("b3_lora_parameter_mode") != "lora"
            or b3_training.get("parameter_mode") != "lora"):
        raise IntegrityError("experiment 035b scale-check record changed")
    return record


def prepare_b15_scale_check(output_root, start_record_path):
    from config import load_config

    paths = _b15_paths(output_root)
    if os.path.abspath(start_record_path) != os.path.abspath(paths["start"]):
        raise IntegrityError("experiment 035b start record uses the wrong path")
    if os.path.exists(paths["record"]):
        return _check_b15_scale_record(read_json(paths["record"]), paths)

    selection = read_json(paths["selection"])
    if (selection.get("selected_scale") != "B3"
            or selection.get("pilot_rung") != "C"):
        raise IntegrityError("experiment 035b cannot replace the scale selection")
    _, _, b3_training = _check_closed_b3(paths)
    start = read_json(paths["start"])
    if (start.get("scale") != "B15" or start.get("seed") != 0
            or start.get("source_tag") != source_tag("B15", 0)):
        raise IntegrityError("experiment 035b start record is not B15 seed 0")
    for key in ("full_result", "full_output", "lora_result", "lora_output"):
        if os.path.exists(paths[key]):
            raise IntegrityError("experiment 035b requires empty active arm directories")

    payload = {
        "completed": True,
        "experiment": "035b",
        "scale": "B15",
        "seed": 0,
        "rung": "A_GC",
        "execution": execution_settings("A_GC"),
        "allocator_config": ALLOCATOR_CONFIG,
        "fixed_roster": ["full_seed0", "lora_seed0"],
        "pilot_authorised": ["full_seed0"],
        "scientific_requires_passing_pilot": True,
        "scale_selection_sha256": sha256(paths["selection"]),
        "b3_science_sha256": sha256(paths["b3_science"]),
        "b3_control_sha256": sha256(paths["b3_control"]),
        "b3_full_status_sha256": sha256(paths["b3_full_status"]),
        "b3_full_binding_sha256": sha256(paths["b3_full_binding"]),
        "b3_lora_training_sha256": sha256(paths["b3_lora_training"]),
        "b3_lora_endpoint_sha256": sha256(paths["b3_lora_endpoint"]),
        "b15_config_hashes": {
            mode: config_hash(load_config(config_path("B15", 0, mode)))
            for mode in ("lora", "full")
        },
        "b15_start_record": start,
        "b15_start_record_sha256": sha256(paths["start"]),
        "b3_lora_parameter_mode": b3_training.get("parameter_mode"),
        "registered_date": "2026-08-19",
    }
    os.makedirs(paths["root"], exist_ok=True)
    record = _write_once(paths["record"], payload)
    return _check_b15_scale_record(record, paths)


def _check_b15_pilot(paths):
    if not os.path.exists(paths["pilot"]):
        raise IntegrityError("experiment 035b pilot report is missing")
    pilot = _check_agc_pilot(paths["pilot"])
    # the pilot proves memory and precision, not that training will hold
    if (pilot.get("scale") != "B15" or pilot.get("seed") != 0
            or pilot.get("rung") != "A_GC"):
        raise IntegrityError("experiment 035b pilot identity changed")
    return pilot


def _b15_a1_path(paths, mode):
    return os.path.join(paths["root"], f"035_b15_{mode}_a1.json")


def _b15_arm_paths(paths, mode):
    return paths[f"{mode}_result"], paths[f"{mode}_output"]


def _check_b15_binding(paths, mode):
    from config import load_config

    result_dir, _ = _b15_arm_paths(paths, mode)
    path = os.path.join(result_dir, EXECUTION_BINDING)
    binding = read_json(path)
    expected_hash = config_hash(load_config(config_path("B15", 0, mode)))
    if (binding.get("task") != "train" or binding.get("scale") != "B15"
            or binding.get("seed") != 0 or binding.get("parameter_mode") != mode
            or binding.get("config_hash") != expected_hash
            or binding.get("start_record") != read_json(paths["start"])
            or binding.get("rung") != "A_GC"
            or binding.get("execution") != execution_settings("A_GC")
            or binding.get("environment", {}).get("allocator_config")
                != ALLOCATOR_CONFIG):
        raise IntegrityError(f"experiment 035b {mode} execution binding changed")
    return binding


def _check_b15_wandb_runs(paths, mode):
    from config import load_config

    result_dir, _ = _b15_arm_paths(paths, mode)
    record_dir = os.path.join(result_dir, B15_WANDB_RUNS)
    if not os.path.isdir(record_dir):
        raise IntegrityError(f"experiment 035b {mode} W&B run record is missing")
    names = sorted(name for name in os.listdir(record_dir) if name.endswith(".json"))
    if not names or set(names) != set(os.listdir(record_dir)):
        raise IntegrityError(f"experiment 035b {mode} W&B run records are incomplete")
    cfg = load_config(config_path("B15", 0, mode))
    binding_path = os.path.join(result_dir, EXECUTION_BINDING)
    expected_name = f"{os.path.basename(result_dir)}_grpo"
    rows = []
    for name in names:
        path = os.path.join(record_dir, name)
        record = read_json(path)
        if (record.get("completed") is not True
                or record.get("experiment") != "035b"
                or record.get("scale") != "B15" or record.get("seed") != 0
                or record.get("parameter_mode") != mode
                or record.get("rung") != "A_GC"
                or record.get("project") != cfg.log.wandb_project
                or record.get("run_id") != name[:-5]
                or record.get("run_name") != expected_name
                or record.get("config_hash") != config_hash(cfg)
                or record.get("execution_binding_sha256") != sha256(binding_path)
                or not 1 <= record.get("first_recorded_step", 0) <= cfg.train.max_steps):
            raise IntegrityError(f"experiment 035b {mode} W&B run record changed")
        rows.append({"path": os.path.abspath(path), "sha256": sha256(path),
                     "project": record["project"], "run_id": record["run_id"],
                     "run_name": record["run_name"]})
    return rows


def _check_b15_a1(record, paths, mode):
    result_dir, _ = _b15_arm_paths(paths, mode)
    status_path = os.path.join(result_dir, "worker_status.json")
    binding_path = os.path.join(result_dir, EXECUTION_BINDING)
    status = read_json(status_path)
    binding = _check_b15_binding(paths, mode)
    wandb_runs = _check_b15_wandb_runs(paths, mode)
    # compare the exact sentinel message, any other failure means something else
    if (record.get("completed") is not True or record.get("outcome") != "a1"
            or record.get("scale") != "B15" or record.get("seed") != 0
            or record.get("parameter_mode") != mode or record.get("rung") != "A_GC"
            or record.get("execution") != execution_settings("A_GC")
            or record.get("allocator_config") != ALLOCATOR_CONFIG
            or record.get("status_sha256") != sha256(status_path)
            or record.get("binding_sha256") != sha256(binding_path)
            or record.get("wandb_runs") != wandb_runs
            or record.get("authorisation_sha256") != sha256(paths["record"])
            or status.get("completed") is not False or status.get("task") != "train"
            or status.get("rung") != "A_GC"
            or status.get("execution") != execution_settings("A_GC")
            or status.get("allocator_config") != ALLOCATOR_CONFIG
            or status.get("cuda_oom") is not False
            or status.get("error_type") != "RuntimeError"
            or status.get("error_message") != AGC_FULL_A1_ERROR
            or binding.get("parameter_mode") != mode):
        raise IntegrityError(f"experiment 035b {mode} A1 record changed")
    if (os.path.exists(os.path.join(result_dir, TRAINING_COMPLETE))
            or os.path.exists(os.path.join(result_dir, ENDPOINT_MANIFEST))):
        raise IntegrityError(f"experiment 035b {mode} A1 arm gained endpoint artifacts")
    return record


def commit_b15_a1_conclusion(output_root, mode):
    if mode not in ("lora", "full"):
        raise ValueError(f"bad experiment 035b parameter mode: {mode}")
    paths = _b15_paths(output_root)
    _check_b15_scale_record(read_json(paths["record"]), paths)
    _check_b15_pilot(paths)
    result_dir, _ = _b15_arm_paths(paths, mode)
    status_path = os.path.join(result_dir, "worker_status.json")
    binding_path = os.path.join(result_dir, EXECUTION_BINDING)
    payload = {
        "completed": True,
        "outcome": "a1",
        "scale": "B15",
        "seed": 0,
        "parameter_mode": mode,
        "rung": "A_GC",
        "execution": execution_settings("A_GC"),
        "allocator_config": ALLOCATOR_CONFIG,
        "status_sha256": sha256(status_path),
        "binding_sha256": sha256(binding_path),
        "wandb_runs": _check_b15_wandb_runs(paths, mode),
        "authorisation_sha256": sha256(paths["record"]),
    }
    record = _write_once(_b15_a1_path(paths, mode), payload)
    return _check_b15_a1(record, paths, mode)


def _is_b15_drive_disconnect(status, paths):
    message = status.get("error_message", "")
    run_fragment = f"/checkpoints/{arm_tag('B15', 0, 'lora')}/grpo/runs/"
    # Drive dropping mid write looks nothing like training failure
    return (
        status.get("completed") is False
        and status.get("task") == "train"
        and status.get("rung") == "A_GC"
        and status.get("execution") == execution_settings("A_GC")
        and status.get("allocator_config") == ALLOCATOR_CONFIG
        and status.get("cuda_oom") is False
        and status.get("error_type") == "FailedPreconditionError"
        and message.endswith(f"; {B15_DRIVE_ERROR}")
        and run_fragment in message
        and status.get("start_step") == 0
        and status.get("end_step") == 100
        and status.get("completed_updates") == 100
    )


def _b15_checkpoint_facts(paths):
    from config import load_config

    output_dir = paths["lora_output"]
    # resume only from a chekpoint holding the whole 100 step state
    checkpoints = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if name.startswith("checkpoint-"):
                try:
                    checkpoints.append((int(name.rsplit("-", 1)[1]), name))
                except ValueError:
                    raise IntegrityError(f"{output_dir}: malformed checkpoint directory")
    if not checkpoints or max(checkpoints)[0] != 100:
        raise IntegrityError("experiment 035b LoRA resume requires checkpoint 100")
    checkpoint_dir = os.path.join(output_dir, max(checkpoints)[1])
    missing = [name for name in B15_RESUME_STATE
               if not os.path.isfile(os.path.join(checkpoint_dir, name))]
    if missing:
        raise IntegrityError(f"{checkpoint_dir}: continuation state is missing {missing}")
    # partial files mean the write was still going when the mount went
    if any(name.endswith(".part") for _, _, names in os.walk(checkpoint_dir)
           for name in names):
        raise IntegrityError(f"{checkpoint_dir}: partial checkpoint file found")

    state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    state = read_json(state_path)
    reward_steps = {
        int(row["step"]) for row in state.get("log_history", [])
        if row.get("step") is not None
        and row.get("rewards/accuracy_reward/mean") is not None
    }
    if (state.get("global_step") != 100 or state.get("max_steps") != 500
            or reward_steps != set(range(1, 101))):
        raise IntegrityError(f"{checkpoint_dir}: expected the complete step-100 state")

    cfg = load_config(config_path("B15", 0, "lora"))
    hash_path = os.path.join(output_dir, "config_hash.txt")
    if not os.path.isfile(hash_path) or read_text(hash_path).strip() != config_hash(cfg):
        raise IntegrityError("experiment 035b LoRA checkpoint config hash changed")
    hashes = {}
    for root, _, names in os.walk(checkpoint_dir):
        for name in sorted(names):
            path = os.path.join(root, name)
            hashes[os.path.relpath(path, checkpoint_dir)] = sha256(path)
    return {
        "step": 100,
        "path": os.path.abspath(checkpoint_dir),
        "sha256": dict(sorted(hashes.items())),
        "config_hash_sha256": sha256(hash_path),
        "trainer_state_sha256": sha256(state_path),
    }


def _b15_drive_attempts(paths):
    attempt_path = os.path.join(paths["lora_result"], "035b_attempts.jsonl")
    if not os.path.isfile(attempt_path):
        raise IntegrityError("experiment 035b LoRA attempt ledger is missing")
    rows = read_jsonl(attempt_path)
    matches = [row for row in rows if _is_b15_drive_disconnect(row, paths)]
    if len(matches) != 1:
        raise IntegrityError("experiment 035b permits one LoRA Drive interruption")
    return attempt_path, rows, matches[0]


def _check_b15_drive_resume(record, paths, verify_checkpoint):
    status = read_json(paths["lora_drive_status"])
    if not _is_b15_drive_disconnect(status, paths):
        raise IntegrityError("experiment 035b archived Drive failure changed")
    binding = _check_b15_binding(paths, "lora")
    attempt_path, rows, failed_attempt = _b15_drive_attempts(paths)
    if rows[0] != failed_attempt:
        raise IntegrityError("experiment 035b Drive failure is not the first LoRA attempt")
    for row in record.get("wandb_runs", []):
        if not os.path.isfile(row.get("path", "")) or sha256(row["path"]) != row.get("sha256"):
            raise IntegrityError("experiment 035b pre-resume W&B record changed")
    expected = {
        "completed": True,
        "experiment": "035b",
        "outcome": "drive_interruption_resume",
        "scale": "B15",
        "seed": 0,
        "parameter_mode": "lora",
        "rung": "A_GC",
        "execution": execution_settings("A_GC"),
        "config_hash": binding["config_hash"],
        "authorisation_sha256": sha256(paths["record"]),
        "binding_sha256": sha256(os.path.join(paths["lora_result"], EXECUTION_BINDING)),
        "failed_status_path": os.path.abspath(paths["lora_drive_status"]),
        "failed_status_sha256": sha256(paths["lora_drive_status"]),
        "failed_attempt": failed_attempt,
        "attempt_ledger_path": os.path.abspath(attempt_path),
        "source_checkpoint": record.get("source_checkpoint"),
        "report_to": ["wandb"],
        "tensorboard_disabled_on_resume": True,
        "maximum_drive_resumes": 1,
        "wandb_runs": record.get("wandb_runs"),
    }
    if record != expected:
        raise IntegrityError("experiment 035b LoRA Drive-resume record changed")
    if verify_checkpoint and _b15_checkpoint_facts(paths) != record["source_checkpoint"]:
        raise IntegrityError("experiment 035b LoRA checkpoint changed before resume")
    return record


def commit_b15_drive_resume(output_root):
    paths = _b15_paths(output_root)
    _check_b15_scale_record(read_json(paths["record"]), paths)
    _check_b15_pilot(paths)
    _check_b15_pair_parity(paths)
    if b15_arm_state(output_root, "full")["state"] not in ("completed", "a1"):
        raise IntegrityError("experiment 035b full outcome must settle before LoRA resume")
    if os.path.exists(os.path.join(paths["lora_result"], TRAINING_COMPLETE)):
        raise IntegrityError("experiment 035b LoRA training is already complete")
    live_status_path = os.path.join(paths["lora_result"], "worker_status.json")
    status = read_json(live_status_path)
    if not _is_b15_drive_disconnect(status, paths):
        raise IntegrityError("experiment 035b LoRA failure is not the registered Drive interruption")
    if not os.path.exists(paths["lora_drive_status"]):
        atomic_write_json(paths["lora_drive_status"], status)
    elif read_json(paths["lora_drive_status"]) != status:
        raise IntegrityError("experiment 035b archived Drive status differs from the live status")
    _, _, failed_attempt = _b15_drive_attempts(paths)
    payload = {
        "completed": True,
        "experiment": "035b",
        "outcome": "drive_interruption_resume",
        "scale": "B15",
        "seed": 0,
        "parameter_mode": "lora",
        "rung": "A_GC",
        "execution": execution_settings("A_GC"),
        "config_hash": _check_b15_binding(paths, "lora")["config_hash"],
        "authorisation_sha256": sha256(paths["record"]),
        "binding_sha256": sha256(os.path.join(paths["lora_result"], EXECUTION_BINDING)),
        "failed_status_path": os.path.abspath(paths["lora_drive_status"]),
        "failed_status_sha256": sha256(paths["lora_drive_status"]),
        "failed_attempt": failed_attempt,
        "attempt_ledger_path": os.path.abspath(
            os.path.join(paths["lora_result"], "035b_attempts.jsonl")),
        "source_checkpoint": _b15_checkpoint_facts(paths),
        "report_to": ["wandb"],
        "tensorboard_disabled_on_resume": True,
        "maximum_drive_resumes": 1,
        "wandb_runs": _check_b15_wandb_runs(paths, "lora"),
    }
    record = _write_once(paths["lora_drive_resume"], payload)
    return _check_b15_drive_resume(record, paths, verify_checkpoint=True)


def b15_arm_state(output_root, mode):
    from config import load_config

    if mode not in ("lora", "full"):
        raise ValueError(f"bad experiment 035b parameter mode: {mode}")
    paths = _b15_paths(output_root)
    result_dir, output_dir = _b15_arm_paths(paths, mode)
    a1_path = _b15_a1_path(paths, mode)
    training_path = os.path.join(result_dir, TRAINING_COMPLETE)
    endpoint_path = os.path.join(result_dir, ENDPOINT_MANIFEST)
    if os.path.exists(a1_path):
        record = _check_b15_a1(read_json(a1_path), paths, mode)
        return {"state": "a1", "record": record, "record_path": a1_path}
    if os.path.exists(training_path):
        _check_b15_binding(paths, mode)
        cfg = load_config(config_path("B15", 0, mode))
        cfg.output_root = output_root
        training = commit_training_complete(
            cfg, result_dir, output_dir, read_json(paths["start"]), "A_GC",
            os.path.join(result_dir, "precision_report.json"))
        return {"state": "completed", "record": training,
                "record_path": training_path, "wandb_runs": training["wandb_runs"]}
    if os.path.exists(endpoint_path):
        raise IntegrityError(f"experiment 035b {mode} endpoint has no training record")
    status_path = os.path.join(result_dir, "worker_status.json")
    if not os.path.exists(status_path):
        if mode == "lora" and os.path.exists(paths["lora_drive_resume"]):
            raise IntegrityError("experiment 035b LoRA resume attempt left no worker status")
        return {"state": "absent"}
    _check_b15_binding(paths, mode)
    status = read_json(status_path)
    if status.get("completed") is True:
        raise IntegrityError(f"experiment 035b {mode} worker has no training record")
    # a1_pending is not a1 until the conclusion record exists
    if (status.get("cuda_oom") is False and status.get("error_type") == "RuntimeError"
            and status.get("error_message") == AGC_FULL_A1_ERROR):
        return {"state": "a1_pending", "status": status}
    if status.get("error_type") == "KeyboardInterrupt":
        return {"state": "interrupted", "status": status}
    if mode == "lora" and _is_b15_drive_disconnect(status, paths):
        if not os.path.exists(paths["lora_drive_resume"]):
            return {"state": "drive_interrupted_pending", "status": status}
        record = _check_b15_drive_resume(
            read_json(paths["lora_drive_resume"]), paths, verify_checkpoint=True)
        return {"state": "interrupted", "status": status,
                "drive_resume": record}
    return {"state": "failed", "status": status}


def commit_b15_pair_settled(output_root):
    paths = _b15_paths(output_root)
    _check_b15_scale_record(read_json(paths["record"]), paths)
    _check_b15_pilot(paths)
    if not os.path.exists(paths["parity"]):
        raise IntegrityError("experiment 035b pair parity is missing")
    parity = _check_b15_pair_parity(paths)
    # both outcomes settle first, so no endpoint can steer the other arm
    outcomes = {mode: b15_arm_state(output_root, mode)
                for mode in ("full", "lora")}
    if any(row["state"] not in ("completed", "a1") for row in outcomes.values()):
        raise IntegrityError("experiment 035b training outcomes are not settled")
    payload = {
        "completed": True,
        "scale": "B15",
        "seed": 0,
        "rung": "A_GC",
        "authorisation_sha256": sha256(paths["record"]),
        "pair_parity_sha256": sha256(paths["parity"]),
        "outcomes": {
            mode: {"state": row["state"], "record_path": row["record_path"],
                   "record_sha256": sha256(row["record_path"]),
                   "wandb_runs": row["record"]["wandb_runs"]}
            for mode, row in outcomes.items()
        },
    }
    return _write_once(paths["settled"], payload)


def _check_b15_pair_settled(paths):
    record = read_json(paths["settled"])
    if (record.get("completed") is not True or record.get("scale") != "B15"
            or record.get("seed") != 0 or record.get("rung") != "A_GC"
            or record.get("authorisation_sha256") != sha256(paths["record"])
            or record.get("pair_parity_sha256") != sha256(paths["parity"])):
        raise IntegrityError("experiment 035b settled-pair record changed")
    current = {mode: b15_arm_state(os.path.dirname(os.path.dirname(paths["root"])), mode)
               for mode in ("full", "lora")}
    for mode, row in current.items():
        bound = record.get("outcomes", {}).get(mode, {})
        if (row["state"] not in ("completed", "a1")
                or row["state"] != bound.get("state")
                or row.get("record_path") != bound.get("record_path")
                or row["record"].get("wandb_runs") != bound.get("wandb_runs")
                or sha256(row["record_path"]) != bound.get("record_sha256")):
            raise IntegrityError("experiment 035b settled outcome changed")
    return record


def _check_b15_pair_parity(paths):
    from config import load_config

    record = check_pair_parity(paths["parity"], "B15")
    start = read_json(paths["start"])
    for mode in ("lora", "full"):
        row = record[mode]
        if (row.get("config_hash")
                != config_hash(load_config(config_path("B15", 0, mode)))
                or row.get("start_record") != start):
            raise IntegrityError("experiment 035b pair parity identity changed")
    return record


def _check_b15_task(cfg, task):
    paths = _b15_paths(cfg.output_root)
    record = _check_b15_scale_record(read_json(paths["record"]), paths)
    if cfg.train.seed != 0 or cfg.train.grpo_parameter_mode not in ("lora", "full"):
        raise IntegrityError("experiment 035b task is outside the fixed roster")
    mode = cfg.train.grpo_parameter_mode
    if config_hash(cfg) != record["b15_config_hashes"][mode]:
        raise IntegrityError("experiment 035b worker config changed")
    if task == "pilot":
        if mode != "full":
            raise IntegrityError("experiment 035b pilot is full seed 0 only")
        if os.path.exists(paths["pilot"]):
            raise IntegrityError("experiment 035b pilot is already concluded")
        return record

    _check_b15_pilot(paths)
    if task == "parity":
        states = [b15_arm_state(cfg.output_root, value)["state"]
                  for value in ("full", "lora")]
        if any(state != "absent" for state in states):
            raise IntegrityError("experiment 035b parity must precede training")
        return record

    if not os.path.exists(paths["parity"]):
        raise IntegrityError("experiment 035b pair parity is missing")
    _check_b15_pair_parity(paths)
    current = b15_arm_state(cfg.output_root, mode)
    full = b15_arm_state(cfg.output_root, "full")
    if task == "train":
        if current["state"] not in ("absent", "interrupted"):
            raise IntegrityError(f"experiment 035b {mode} training is already concluded")
        if mode == "lora" and full["state"] not in ("completed", "a1"):
            raise IntegrityError("experiment 035b full outcome must settle before LoRA")
        return record
    if task == "eval":
        _check_b15_pair_settled(paths)
        if current["state"] != "completed":
            raise IntegrityError(f"experiment 035b {mode} arm has no endpoint to evaluate")
        return record
    raise IntegrityError(f"experiment 035b does not authorise task {task}")


def commit_execution_lock(output_root, scale, rung, full_seed0_record):
    if full_seed0_record.get("completed") is not True:
        raise IntegrityError("execution lock requires a completed full seed-0 arm")
    if (full_seed0_record.get("scale") != scale or full_seed0_record.get("seed") != 0
            or full_seed0_record.get("parameter_mode") != "full"
            or full_seed0_record.get("rung") != rung
            or full_seed0_record.get("execution") != execution_settings(rung)):
        raise IntegrityError("execution lock received the wrong full seed-0 record")
    path = os.path.join(output_root, "results", "035_technical", "execution_lock.json")
    payload = {"completed": True, "scale": scale, "rung": rung,
               "execution": execution_settings(rung),
               "full_seed0_training_sha256": sha256(full_seed0_record["record_path"])}
    return _write_once(path, payload)


def module_family(name):
    if ("embed_tokens" in name or ".wte." in name
            or "tok_emb" in name or "pos_emb" in name):
        return "embeddings"
    if "self_attn" in name or ".attn." in name or ".att." in name:
        return "attention"
    if ".mlp." in name or ".ff." in name:
        return "mlp"
    if "norm" in name:
        return "normalization"
    if "lm_head" in name or "out_head" in name:
        return "output_head"
    return None


def verify_reference_structure(trainer):
    # LoRA reaches its reference by disabling the adapter. Full mode holds a
    # frozen copy of the same start.
    import torch

    model = trainer.accelerator.unwrap_model(trainer.model)
    if trainer.ref_model is None:
        if not hasattr(model, "disable_adapter"):
            raise IntegrityError("LoRA reference cannot disable its adapter")
        return {"verified": True, "reference_kind": "adapter_disabled"}
    reference = trainer.accelerator.unwrap_model(trainer.ref_model)
    policy_named = dict(model.named_parameters())
    reference_named = dict(reference.named_parameters())
    if set(policy_named) != set(reference_named):
        raise IntegrityError("full policy and reference parameter names differ")
    for name, param in policy_named.items():
        ref = reference_named[name]
        if param.shape != ref.shape or param.dtype != ref.dtype:
            raise IntegrityError(f"full reference tensor differs: {name}")
        if ref.requires_grad or not torch.equal(param.detach(), ref.detach()):
            raise IntegrityError(f"full reference is not a frozen copy: {name}")
    return {"verified": True, "reference_kind": "separate_frozen_model"}


class PrecisionAudit:
    def __init__(self, stop_after=None, wandb_record_dir=None, wandb_context=None,
                 allow_initial_zero_gradient=False):
        self.stop_after = stop_after
        if bool(wandb_record_dir) != bool(wandb_context):
            raise ValueError("W&B record directory and context must be provided together")
        self.wandb_record_dir = wandb_record_dir
        self.wandb_context = wandb_context
        self.allow_initial_zero_gradient = allow_initial_zero_gradient
        self.wandb_record = None
        self.trainer = None
        self.before = {}
        self.gradient = {
            "dtypes": set(), "finite": True, "nonzero": 0,
            "updates_checked": 0, "first_nonzero_update": None,
        }
        self.optimizer_dtypes = set()
        self.reference_gate = None
        self.report = {}

    def _capture_gradient_step(self, parameters, update):
        import torch

        if (self.gradient["dtypes"]
                and (not self.allow_initial_zero_gradient
                     or self.gradient["nonzero"] > 0)):
            return
        found = False
        nonzero = 0
        for param in parameters:
            if param.requires_grad and param.grad is not None:
                found = True
                self.gradient["dtypes"].add(
                    str(param.grad.dtype).replace("torch.", ""))
                self.gradient["finite"] &= bool(
                    torch.isfinite(param.grad).all().item())
                nonzero += int(torch.count_nonzero(param.grad).item())
        if not found:
            return
        self.gradient["updates_checked"] += 1
        self.gradient["nonzero"] += nonzero
        if nonzero and self.gradient["first_nonzero_update"] is None:
            self.gradient["first_nonzero_update"] = int(update)

    def attach(self, trainer):
        import torch
        from transformers import TrainerCallback

        self.trainer = trainer
        self.reference_gate = verify_reference_structure(trainer)
        trainable = [(name, param) for name, param in trainer.model.named_parameters()
                     if param.requires_grad]
        if not trainable or any(param.dtype != torch.float32 for _, param in trainable):
            raise IntegrityError("matched GRPO trainable parameters must use float32 storage")
        # keep one fixed CPU sample for each present module family
        for name, param in trainable:
            family = module_family(name)
            if family and family not in self.before:
                count = min(param.numel(), MAX_AUDIT_ELEMENTS)
                indices = torch.linspace(0, param.numel() - 1, count, dtype=torch.long)
                values = param.detach().reshape(-1)[indices.to(param.device)].cpu().clone()
                self.before[family] = {
                    "name": name, "indices": indices, "value": values,
                    "tensor_elements": param.numel(),
                }
        if not self.before:
            raise IntegrityError("precision audit found no registered module family")
        audit = self

        class callback(TrainerCallback):
            def on_log(self, args, state, control, **kwargs):
                if (audit.wandb_context and audit.wandb_record is None
                        and state.global_step > 0):
                    audit.wandb_record = capture_wandb_run(
                        audit.wandb_record_dir, audit.wandb_context,
                        state.global_step)
                return control

            def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
                active_model = model or audit.trainer.model
                audit._capture_gradient_step(
                    active_model.parameters(), int(state.global_step) + 1)
                return control

            def on_step_end(self, args, state, control, optimizer=None, **kwargs):
                active_optimizer = optimizer or audit.trainer.optimizer
                if not audit.optimizer_dtypes and active_optimizer is not None:
                    for values in active_optimizer.state.values():
                        for key in ("exp_avg", "exp_avg_sq"):
                            value = values.get(key)
                            if value is not None:
                                audit.optimizer_dtypes.add(
                                    str(value.dtype).replace("torch.", ""))
                if audit.stop_after and state.global_step >= audit.stop_after:
                    control.should_training_stop = True
                return control

        trainer.add_callback(callback())

    def finish(self, trainer):
        import torch

        named = dict(trainer.model.named_parameters())
        parameters_finite = all(
            bool(torch.isfinite(param).all().item())
            for param in named.values() if param.requires_grad)
        optimizer_states_finite = True
        for values in trainer.optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                value = values.get(key)
                if value is not None:
                    optimizer_states_finite &= bool(torch.isfinite(value).all().item())
        changes = {}
        for family, row in self.before.items():
            param = named[row["name"]]
            current = param.detach().reshape(-1)[row["indices"].to(param.device)].cpu()
            before = row["value"]
            diff = (current - before).float()
            changed = int(torch.count_nonzero(diff).item())
            changes[family] = {
                "tensor": row["name"],
                "tensor_elements": row["tensor_elements"],
                "elements": diff.numel(),
                "changed_elements": changed,
                "changed_fraction": changed / diff.numel(),
                "mean_abs_change": float(diff.abs().mean().item()),
                "max_abs_change": float(diff.abs().max().item()),
                "change_norm_ratio": float(
                    diff.norm().item() / max(before.float().norm().item(), 1.0e-30)),
            }
        self.report = {
            "trainable_parameter_dtypes": ["float32"],
            "gradient_dtypes": sorted(self.gradient["dtypes"]),
            "gradients_finite": self.gradient["finite"],
            "parameters_finite": parameters_finite,
            "optimizer_states_finite": optimizer_states_finite,
            "nonzero_gradient_elements": self.gradient["nonzero"],
            "gradient_updates_checked": self.gradient["updates_checked"],
            "first_nonzero_gradient_update": self.gradient["first_nonzero_update"],
            "optimizer_state_dtypes": sorted(self.optimizer_dtypes),
            "bf16_enabled": bool(getattr(trainer.args, "bf16", False)),
            "accelerator_mixed_precision": getattr(trainer.accelerator, "mixed_precision", None),
            "module_changes": changes,
            "reference_gate": self.reference_gate,
            "wandb_run_record": self.wandb_record,
        }
        if self.wandb_context and self.wandb_record is None:
            raise IntegrityError("experiment 035b W&B run was not recorded")
        if self.report["gradient_dtypes"] != ["float32"]:
            raise IntegrityError("matched GRPO gradients must use float32 storage")
        if self.report["optimizer_state_dtypes"] != ["float32"]:
            raise IntegrityError("matched GRPO Adam states must use float32 storage")
        if (not self.report["gradients_finite"] or not parameters_finite
                or not optimizer_states_finite or self.report["nonzero_gradient_elements"] == 0):
            raise IntegrityError("matched GRPO first-step gradients are invalid")
        if (not self.report["bf16_enabled"]
                or self.report["accelerator_mixed_precision"] != "bf16"):
            raise IntegrityError("matched GRPO must use bfloat16 autocast")
        if any(row["changed_elements"] == 0 for row in changes.values()):
            raise IntegrityError("a registered module family did not change")
        return self.report


def parity_fingerprint(trainer, tokenizer, prompts, start_record, cfg, rung):
    import torch

    model = trainer.accelerator.unwrap_model(trainer.model)
    reference_gate = verify_reference_structure(trainer)
    model.eval()
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    # compare a fixed slice of the same next-token distributions
    def values(active_model, context=nullcontext()):
        active_model.eval()
        with context, torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = active_model(**encoded).logits[:, -1, :].float()
        ids = [(i * 997) % logits.shape[-1] for i in range(32)]
        return torch.log_softmax(logits, dim=-1)[:, ids].cpu().tolist()

    policy = values(model)
    if trainer.ref_model is not None:
        reference_model = trainer.accelerator.unwrap_model(trainer.ref_model)
        reference = values(reference_model)
    else:
        reference = values(model, model.disable_adapter())

    return {
        "completed": True,
        "scale": cfg.scale.name,
        "seed": cfg.train.seed,
        "parameter_mode": cfg.train.grpo_parameter_mode,
        "rung": rung,
        "execution": execution_settings(rung),
        "beta": cfg.grpo.beta,
        "config_hash": config_hash(cfg),
        "start_record": start_record,
        "storage_dtype": "float32",
        "compute_dtype": "bfloat16",
        "reference_kind": reference_gate["reference_kind"],
        "reference_structure": reference_gate,
        "policy_logprobs": policy,
        "reference_logprobs": reference,
    }


def _max_abs(a, b):
    flat_a = [value for row in a for value in row]
    flat_b = [value for row in b for value in row]
    if len(flat_a) != len(flat_b):
        raise IntegrityError("parity fingerprints have different lengths")
    return max(abs(x - y) for x, y in zip(flat_a, flat_b))


def commit_pair_parity(path, lora_record, full_record):
    # the pair commits only when every field outside the treatment matches
    for record, mode in ((lora_record, "lora"), (full_record, "full")):
        if record.get("completed") is not True or record.get("parameter_mode") != mode:
            raise IntegrityError(f"experiment 035 {mode} parity record is incomplete")
        if (record.get("storage_dtype") != "float32"
                or record.get("compute_dtype") != "bfloat16"
                or record.get("reference_structure", {}).get("verified") is not True):
            raise IntegrityError(f"experiment 035 {mode} parity precision differs")
    same = (lora_record["scale"] == full_record["scale"]
            and lora_record["seed"] == full_record["seed"]
            and lora_record["rung"] == full_record["rung"]
            and lora_record["beta"] == full_record["beta"]
            and lora_record["start_record"] == full_record["start_record"]
            and lora_record.get("versions") == full_record.get("versions")
            and lora_record.get("gpu", {}).get("device")
                == full_record.get("gpu", {}).get("device")
            and lora_record.get("allocator_config") == full_record.get("allocator_config"))
    if not same:
        raise IntegrityError("matched parity records do not describe one pair")
    checks = {
        "lora_policy_vs_reference": _max_abs(
            lora_record["policy_logprobs"], lora_record["reference_logprobs"]),
        "full_policy_vs_reference": _max_abs(
            full_record["policy_logprobs"], full_record["reference_logprobs"]),
        "policy_between_modes": _max_abs(
            lora_record["policy_logprobs"], full_record["policy_logprobs"]),
        "reference_between_modes": _max_abs(
            lora_record["reference_logprobs"], full_record["reference_logprobs"]),
    }
    if any(value > PARITY_TOLERANCE for value in checks.values()):
        raise IntegrityError(f"experiment 035 parity tolerance failed: {checks}")
    payload = {"completed": True, "tolerance": PARITY_TOLERANCE,
               "checks": checks, "lora": lora_record, "full": full_record}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return _write_once(path, payload)


def _training_rows(result_dir, max_steps):
    # require one complete schedule before accepting the endpoint
    path = os.path.join(result_dir, "grpo_trainlog.json")
    history = read_json(path)
    rows = [row for row in history if row.get("step") is not None
            and row.get("rewards/accuracy_reward/mean") is not None]
    counts = Counter(int(row["step"]) for row in rows)
    expected = set(range(1, max_steps + 1))
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        raise IntegrityError(f"{path}: training steps are missing or duplicated")
    lr_rows = sorted((int(row["step"]), float(row["learning_rate"]))
                     for row in rows if row.get("learning_rate") is not None)
    if len(lr_rows) != max_steps or {step for step, _ in lr_rows} != expected:
        raise IntegrityError(f"{path}: learning-rate coverage is incomplete")
    values = [value for _, value in lr_rows]
    if any(b > a + 1.0e-12 for a, b in zip(values, values[1:])):
        raise IntegrityError(f"{path}: learning rate increased")
    return rows


def commit_training_complete(cfg, result_dir, output_dir, start_record, rung,
                             precision_report_path):
    _training_rows(result_dir, cfg.train.max_steps)
    precision = read_json(precision_report_path)
    if (precision.get("gradient_dtypes") != ["float32"]
            or precision.get("optimizer_state_dtypes") != ["float32"]
            or precision.get("bf16_enabled") is not True
            or precision.get("accelerator_mixed_precision") != "bf16"):
        raise IntegrityError("experiment 035 precision report differs from the matched design")
    if precision.get("reference_gate", {}).get("verified") is not True:
        raise IntegrityError("experiment 035 reference gate is missing")
    if cfg.train.grpo_parameter_mode == "lora":
        artifact = {"kind": "lora", "sha256": adapter_hashes_of(output_dir)}
        if any(name.startswith("model-") and name.endswith(".safetensors")
               for name in os.listdir(output_dir)):
            raise IntegrityError("matched LoRA output contains full-model shards")
    else:
        check_merged_complete(output_dir)
        if any(os.path.exists(os.path.join(output_dir, name)) for name in ADAPTER_FILES[:2]):
            raise IntegrityError("matched full output contains LoRA adapter files")
        artifact = {"kind": "full", "sha256": merged_hashes(output_dir)}
    verdict_path = os.path.join(output_dir, "a2_verdict.json")
    if not os.path.exists(verdict_path):
        raise IntegrityError("experiment 035 observe-only a2 verdict is missing")
    payload = {
        "completed": True,
        "scale": cfg.scale.name,
        "seed": cfg.train.seed,
        "parameter_mode": cfg.train.grpo_parameter_mode,
        "rung": rung,
        "execution": execution_settings(rung),
        "config_hash": config_hash(cfg),
        "start_record": start_record,
        "precision_report_sha256": sha256(precision_report_path),
        "a2_verdict_sha256": sha256(verdict_path),
        "artifact": artifact,
    }
    if cfg.scale.name == "B15":
        paths = _b15_paths(cfg.output_root)
        payload["wandb_runs"] = _check_b15_wandb_runs(
            paths, cfg.train.grpo_parameter_mode)
        if (cfg.train.grpo_parameter_mode == "lora"
                and os.path.exists(paths["lora_drive_resume"])):
            resume = _check_b15_drive_resume(
                read_json(paths["lora_drive_resume"]), paths, verify_checkpoint=False)
            payload["infrastructure_resume"] = {
                "path": os.path.abspath(paths["lora_drive_resume"]),
                "sha256": sha256(paths["lora_drive_resume"]),
                "report_to": resume["report_to"],
            }
    path = os.path.join(result_dir, TRAINING_COMPLETE)
    payload["record_path"] = os.path.abspath(path)
    return _write_once(path, payload)


def endpoint_state(result_dir, training_record):
    # tell unfinished work apart from committed files that changed later
    path = os.path.join(result_dir, ENDPOINT_MANIFEST)
    if not os.path.exists(path):
        return "absent"
    manifest = read_json(path)
    if manifest.get("completed") is not True:
        return "absent"
    if manifest.get("training_record") != training_record:
        return "tampered"
    for name in ENDPOINT_FILES:
        artifact = os.path.join(result_dir, name)
        if not os.path.exists(artifact) or manifest.get("sha256", {}).get(name) != sha256(artifact):
            return "tampered"
    return "committed"


def commit_endpoint(result_dir, scratch_dir, training_record, package="035_w1"):
    state = endpoint_state(result_dir, training_record)
    if state == "tampered":
        raise IntegrityError(f"{result_dir}: committed endpoint changed")
    if state == "committed":
        return read_json(os.path.join(result_dir, ENDPOINT_MANIFEST))
    for name in ENDPOINT_FILES:
        path = os.path.join(scratch_dir, name)
        if not os.path.exists(path):
            raise IntegrityError(f"{scratch_dir}: endpoint is missing {name}")
    for name in ("grpo_preds.jsonl", "grpo_logprob_preds.jsonl"):
        if len(read_jsonl(os.path.join(scratch_dir, name))) != 1884:
            raise IntegrityError(f"{scratch_dir}: {name} does not contain 1884 rows")
    os.makedirs(result_dir, exist_ok=True)
    hashes = {}
    for name in ENDPOINT_FILES:
        stage_into(result_dir, os.path.join(scratch_dir, name), name)
        hashes[name] = sha256(os.path.join(result_dir, name))
    payload = {"completed": True, "package": package, "sha256": hashes,
               "training_record": training_record,
               "timestamp": datetime.now().isoformat(timespec="seconds")}
    # write the manifest last so partial files cannot look complete
    atomic_write_json(os.path.join(result_dir, ENDPOINT_MANIFEST), payload)
    return payload


def commit_start_probe(result_dir, scratch_dir, start_record):
    # the shared baseline probe stays bound to the exact starting model
    files = ("start_logprob_preds.jsonl", "start_logprob.json")
    manifest_path = os.path.join(result_dir, "035_start_probe_manifest.json")
    if os.path.exists(manifest_path):
        manifest = read_json(manifest_path)
        if manifest.get("start_record") != start_record:
            raise IntegrityError(f"{manifest_path}: start probe binding changed")
        for name in files:
            if sha256(os.path.join(result_dir, name)) != manifest["sha256"][name]:
                raise IntegrityError(f"{manifest_path}: start probe file changed")
        return manifest
    if len(read_jsonl(os.path.join(scratch_dir, files[0]))) != 1884:
        raise IntegrityError("start probe does not contain 1884 rows")
    if read_json(os.path.join(scratch_dir, files[1])).get("n_items") != 1884:
        raise IntegrityError("start probe metrics do not contain 1884 items")
    os.makedirs(result_dir, exist_ok=True)
    hashes = {}
    for name in files:
        stage_into(result_dir, os.path.join(scratch_dir, name), name)
        hashes[name] = sha256(os.path.join(result_dir, name))
    payload = {"completed": True, "start_record": start_record, "sha256": hashes}
    atomic_write_json(manifest_path, payload)
    return payload


def safe_close_checkpoints(result_dir, output_dir):
    # remove resume checkpoints only after the endpoint is committed
    training = read_json(os.path.join(result_dir, TRAINING_COMPLETE))
    if endpoint_state(result_dir, training) != "committed":
        raise IntegrityError("checkpoint close-out requires a committed endpoint")
    record_path = os.path.join(result_dir, "035_checkpoint_closeout.json")
    if os.path.exists(record_path):
        record = read_json(record_path)
        for row in record.get("removed", []):
            checkpoint = os.path.join(output_dir, row["path"])
            if os.path.isdir(checkpoint):
                shutil.rmtree(checkpoint)
        return record
    removed = []
    for name in sorted(os.listdir(output_dir)):
        checkpoint_path = os.path.join(output_dir, name)
        if name.startswith("checkpoint-") and os.path.isdir(checkpoint_path):
            size = sum(os.path.getsize(os.path.join(root, file))
                       for root, _, files in os.walk(checkpoint_path) for file in files)
            removed.append({"path": name, "bytes": size})
    record = {"completed": True, "removed": removed,
              "timestamp": datetime.now().isoformat(timespec="seconds")}
    _write_once(record_path, record)
    for row in removed:
        shutil.rmtree(os.path.join(output_dir, row["path"]))
    return record
