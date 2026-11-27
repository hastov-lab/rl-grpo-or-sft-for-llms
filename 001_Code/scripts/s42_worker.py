# keep experiment 042 tasks in disposable processes so evaluation starts with free memory
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_utils import read_json
from scripts.grpo_worker import classify_failure, save_status
from scripts.s12_eval import IntegrityError, atomic_write_json, read_jsonl, sha256, stage_into


SEEDS_042 = (0, 1, 2)
CONFIGS_042 = tuple(f"configs/042_colab_m124_seed{seed}.yaml" for seed in SEEDS_042)
BASE_TAG = "042_COLAB_FULL_M124_base"
PILOT_TAG = "042_COLAB_FULL_M124_pilot"
ENDPOINT_FILES = {
    branch: (f"{branch}_preds.jsonl", f"{branch}_eval.json",
             f"{branch}_logprob_preds.jsonl", f"{branch}_logprob.json")
    for branch in ("base", "sft", "grpo")
}
A1_ERROR = (
    "sentinel abort: a1: over 30% of rollouts truncated in the first 50 steps. "
    "raise the completion cap or mask truncated completions"
)


def config_path(seed):
    if seed not in SEEDS_042:
        raise ValueError(f"experiment 042 does not register seed {seed}")
    return f"configs/042_colab_m124_seed{seed}.yaml"


def arm_tag(seed):
    if seed not in SEEDS_042:
        raise ValueError(f"experiment 042 does not register seed {seed}")
    return f"042_COLAB_FULL_M124_sft_grpo_seed{seed}"


def _write_once(path, payload):
    # keep a committed record immutable on every resumed session
    if os.path.exists(path):
        if read_json(path) != payload:
            raise IntegrityError(f"{path}: committed record changed")
        return read_json(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    atomic_write_json(path, payload)
    return payload


def _root_hashes(path):
    # bind final artifact files without treating resume checkpoints as outputs
    return {
        name: sha256(os.path.join(path, name))
        for name in sorted(os.listdir(path))
        if os.path.isfile(os.path.join(path, name)) and not name.endswith(".part")
    }


def _wandb_records(result_dir, cfg, package="042", stage="grpo"):
    # keep every resumed online run available when a sentinel stops local logging
    from checkpoint import config_hash

    legacy = package == "042" and stage == "grpo"
    dirname = "042_wandb_runs" if legacy else f"{package}_{stage}_wandb_runs"
    record_dir = os.path.join(result_dir, dirname)
    if not os.path.isdir(record_dir):
        raise IntegrityError(f"{result_dir}: W&B run record is missing")
    names = sorted(name for name in os.listdir(record_dir) if name.endswith(".json"))
    if not names or set(names) != set(os.listdir(record_dir)):
        raise IntegrityError(f"{result_dir}: W&B run records are incomplete")
    expected_name = f"{os.path.basename(result_dir)}_{stage}"
    rows = []
    for name in names:
        path = os.path.join(record_dir, name)
        record = read_json(path)
        if (record.get("completed") is not True or record.get("experiment") != package
                or (not legacy and record.get("stage") != stage)
                or record.get("seed") != cfg.train.seed
                or record.get("project") != cfg.log.wandb_project
                or record.get("run_id") != name[:-5]
                or record.get("run_name") != expected_name
                or record.get("config_hash") != config_hash(cfg)
                or not 1 <= record.get("first_recorded_step", 0) <= cfg.train.max_steps):
            raise IntegrityError(f"{result_dir}: W&B run record changed")
        rows.append({"path": os.path.abspath(path), "sha256": sha256(path),
                     "project": record["project"], "run_id": record["run_id"],
                     "run_name": record["run_name"]})
    return rows


def _capture_wandb_run(result_dir, cfg, step, package="042", stage="grpo"):
    import wandb
    from checkpoint import config_hash

    active = wandb.run
    if active is None:
        raise IntegrityError(f"experiment {package} W&B run is not active")
    run_id = str(getattr(active, "id", "") or "")
    project = str(getattr(active, "project", "") or "")
    run_name = str(getattr(active, "name", "") or "")
    expected_name = f"{os.path.basename(result_dir)}_{stage}"
    if (not run_id or not run_id.replace("-", "").replace("_", "").isalnum()
            or project != cfg.log.wandb_project or run_name != expected_name):
        raise IntegrityError(f"experiment {package} W&B identity differs from the worker")
    legacy = package == "042" and stage == "grpo"
    dirname = "042_wandb_runs" if legacy else f"{package}_{stage}_wandb_runs"
    record_dir = os.path.join(result_dir, dirname)
    os.makedirs(record_dir, exist_ok=True)
    path = os.path.join(record_dir, f"{run_id}.json")
    stable = {
        "completed": True,
        "experiment": package,
        "seed": cfg.train.seed,
        "project": project,
        "run_id": run_id,
        "run_name": run_name,
        "entity": str(getattr(active, "entity", "") or "") or None,
        "config_hash": config_hash(cfg),
    }
    if not legacy:
        stable["stage"] = stage
    if os.path.exists(path):
        record = read_json(path)
        if any(record.get(key) != value for key, value in stable.items()):
            raise IntegrityError(f"experiment {package} W&B run record changed")
        return record
    stable["first_recorded_step"] = int(step)
    return _write_once(path, stable)


def verify_artifact_record(record_path):
    from checkpoint import check_artifact_complete

    record = read_json(record_path)
    path = record.get("artifact")
    check_artifact_complete(path, expected="full")
    if record.get("artifact_sha256") != _root_hashes(path):
        raise IntegrityError(f"{record_path}: artifact hashes changed")
    return record


def commit_full_artifact(source, destination, record_path, role, extra=None):
    from checkpoint import check_artifact_complete

    check_artifact_complete(source, expected="full")
    if os.path.exists(record_path):
        return verify_artifact_record(record_path)
    if os.path.exists(destination):
        if os.listdir(destination):
            raise IntegrityError(f"{destination}: uncommitted artifact files already exist")
    else:
        os.makedirs(destination)
    # copy complete files first, then let the record become the comit marker
    for name in sorted(os.listdir(source)):
        src = os.path.join(source, name)
        if os.path.isfile(src):
            stage_into(destination, src, name)
    check_artifact_complete(destination, expected="full")
    payload = {
        "completed": True,
        "experiment": "042",
        "role": role,
        "artifact": os.path.abspath(destination),
        "artifact_sha256": _root_hashes(destination),
    }
    if extra:
        payload.update(extra)
    return _write_once(record_path, payload)


def convert_base(cfg, scratch_dir, destination, record_path):
    from checkpoint import write_artifact_type
    from train.model_utils import (load_scratch_model_and_tokenizer, resolve_model_id,
                                   write_model_base)

    if os.path.exists(record_path):
        return verify_artifact_record(record_path)
    if os.path.exists(scratch_dir):
        shutil.rmtree(scratch_dir)
    os.makedirs(scratch_dir)
    source = resolve_model_id(cfg)
    model, tokenizer = load_scratch_model_and_tokenizer(cfg, source, dtype="float32")
    model.save_pretrained(scratch_dir)
    tokenizer.save_pretrained(scratch_dir)
    write_model_base(scratch_dir, "hf_id", source)
    write_artifact_type(scratch_dir, "full", family="scratch_gpt", scale="M124")
    from gpt2_scratch.weights import reference_checkpoint
    _, revision = reference_checkpoint(source)
    return commit_full_artifact(
        scratch_dir, destination, record_path, "immutable_base",
        {"source": source, "source_revision": revision})


def build_length_audit(cfg, path):
    from data.gsm8k import load_sft
    from data.prompts import MC_SFT_THINK
    from data.splits import build_eval_split, build_grpo_split
    from data.tagging import wrap_continuation
    from train.model_utils import load_scratch_tokenizer, resolve_model_id
    from train.sft import build_sft_rows

    if os.path.exists(path):
        record = read_json(path)
        if (record.get("completed") is not True
                or record.get("completion_cap") != cfg.grpo.max_completion_length):
            raise IntegrityError(f"{path}: token audit changed")
        return record
    tokenizer = load_scratch_tokenizer(cfg, resolve_model_id(cfg))
    train_rows = load_sft()
    val_rows = load_sft(split="test", limit=cfg.train.sft_val_examples)
    grpo_rows = build_grpo_split()
    eval_rows = build_eval_split()

    def lengths(texts):
        return [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts]

    sft_rows = build_sft_rows(
        [*train_rows, *val_rows], tokenizer.eos_token or "", True)
    sft_lengths = lengths(row["prompt"] + row["completion"] for row in sft_rows)
    grpo_lengths = lengths(row["prompt"] for row in grpo_rows)
    eval_prompt_lengths = lengths(row["prompt"] for row in eval_rows)
    eval_probe_lengths = lengths(row["probe"] for row in eval_rows)
    mc_targets = [
        row["prompt"] + wrap_continuation(MC_SFT_THINK, row["gold"])
        for row in grpo_rows
    ]
    mc_lengths = lengths(mc_targets)
    # derive the training cap from the fixed development pool. Never the test rows
    cap = min(512, 1024 - max(grpo_lengths))
    letter_tokens = {
        letter: tokenizer.encode(" " + letter, add_special_tokens=False)
        for letter in "ABCD"
    }
    if (max(sft_lengths) > 1024 or max(grpo_lengths) >= 1024
            or max(eval_prompt_lengths) >= 1024 or max(eval_probe_lengths) >= 1024
            or max(mc_lengths) > 1024):
        raise IntegrityError("experiment 042 contains a prompt outside the GPT-2 context")
    if cap != cfg.grpo.max_completion_length or cap != cfg.eval.max_new_tokens:
        raise IntegrityError("experiment 042 config does not match the derived completion cap")
    if (any(len(ids) != 1 for ids in letter_tokens.values())
            or len({ids[0] for ids in letter_tokens.values()}) != 4
            or any(not row["probe"].endswith("Answer:") for row in eval_rows)):
        raise IntegrityError("experiment 042 probe token boundary is not valid")
    test_ids = [
        hashlib.sha256(f"{row['subject']}\0{row['prompt']}".encode()).hexdigest()
        for row in eval_rows
    ]
    payload = {
        "completed": True,
        "experiment": "042",
        "tokenizer": resolve_model_id(cfg),
        "sft_rows": len(sft_rows),
        "grpo_rows": len(grpo_rows),
        "test_rows": len(eval_rows),
        "max_sft_tokens": max(sft_lengths),
        "max_grpo_prompt_tokens": max(grpo_lengths),
        "max_mc_sft_tokens": max(mc_lengths),
        "max_test_prompt_tokens": max(eval_prompt_lengths),
        "max_test_probe_tokens": max(eval_probe_lengths),
        "completion_cap": cap,
        "letter_token_ids": {key: value[0] for key, value in letter_tokens.items()},
        "probe_surface": " leading space A/B/C/D after Answer:",
        "test_row_ids_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
    }
    return _write_once(path, payload)


def _step_rows(path, max_steps, fields=()):
    # require one finite training row for every scheduled update
    rows = read_json(path)
    selected = [row for row in rows if row.get("step") is not None and "loss" in row]
    counts = {}
    for row in selected:
        counts[int(row["step"])] = counts.get(int(row["step"]), 0) + 1
        if not math.isfinite(float(row["loss"])):
            raise IntegrityError(f"{path}: training loss is not finite")
        for field in fields:
            if field not in row or not math.isfinite(float(row[field])):
                raise IntegrityError(f"{path}: {field} is missing or not finite")
    if counts != {step: 1 for step in range(1, max_steps + 1)}:
        raise IntegrityError(f"{path}: training steps are missing or duplicated")
    return selected


def commit_sft_training(cfg, result_dir, output_dir, base_record_path):
    from checkpoint import check_artifact_complete, config_hash

    base_record = verify_artifact_record(base_record_path)
    rows = _step_rows(os.path.join(result_dir, "sft_trainlog.json"), cfg.train.max_steps)
    check_artifact_complete(output_dir, expected="full")
    record_path = os.path.join(result_dir, "042_sft_training.json")
    payload = {
        "completed": True,
        "experiment": "042",
        "stage": "sft",
        "seed": cfg.train.seed,
        "config_hash": config_hash(cfg),
        "updates": len(rows),
        "parameter_mode": "full",
        "storage_dtype": "float32",
        "compute_dtype": "bfloat16",
        "base_record": base_record,
        "artifact": os.path.abspath(output_dir),
        "artifact_sha256": _root_hashes(output_dir),
    }
    return _write_once(record_path, payload)


def _starvation_summary(rows):
    # keep the three registered reward signal fields together in one verdict
    fields = ("rewards/accuracy_reward/mean", "rewards/format_reward/mean",
              "frac_reward_zero_std")

    def means(part):
        return {
            "mean_accuracy_reward": sum(float(row[fields[0]]) for row in part) / len(part),
            "mean_format_reward": sum(float(row[fields[1]]) for row in part) / len(part),
            "share_zero_reward_spread": sum(float(row[fields[2]]) for row in part) / len(part),
        }

    complete = means(rows)
    return {
        "complete": complete,
        "first_50": means(rows[:50]),
        "last_50": means(rows[-50:]),
        "severely_reward_starved": bool(
            complete["share_zero_reward_spread"] >= 0.95
            and complete["mean_accuracy_reward"] <= 0.05
            and complete["mean_format_reward"] <= 0.02),
    }


def commit_grpo_training(cfg, result_dir, output_dir, sft_record_path, precision_path):
    from checkpoint import config_hash, check_artifact_complete

    sft_record = read_json(sft_record_path)
    if sft_record.get("completed") is not True:
        raise IntegrityError("experiment 042 GRPO requires a completed SFT record")
    fields = ("rewards/accuracy_reward/mean", "rewards/format_reward/mean",
              "frac_reward_zero_std")
    rows = _step_rows(
        os.path.join(result_dir, "grpo_trainlog.json"), cfg.train.max_steps, fields)
    precision = read_json(precision_path)
    required_families = {"embeddings", "attention", "mlp", "normalization"}
    if (precision.get("gradient_dtypes") != ["float32"]
            or precision.get("optimizer_state_dtypes") != ["float32"]
            or precision.get("bf16_enabled") is not True
            or precision.get("accelerator_mixed_precision") != "bf16"
            or precision.get("reference_gate", {}).get("verified") is not True
            or not required_families <= set(precision.get("module_changes", {}))):
        raise IntegrityError("experiment 042 precision report is incomplete")
    check_artifact_complete(output_dir, expected="full")
    verdict_path = os.path.join(output_dir, "a2_verdict.json")
    if not os.path.exists(verdict_path):
        raise IntegrityError("experiment 042 observe only A2 record is missing")
    record_path = os.path.join(result_dir, "042_grpo_training.json")
    payload = {
        "completed": True,
        "experiment": "042",
        "stage": "grpo",
        "seed": cfg.train.seed,
        "config_hash": config_hash(cfg),
        "updates": len(rows),
        "parameter_mode": "full",
        "sft_record_sha256": sha256(sft_record_path),
        "precision_report_sha256": sha256(precision_path),
        "a2_verdict_sha256": sha256(verdict_path),
        "wandb_runs": _wandb_records(result_dir, cfg),
        "starvation": _starvation_summary(rows),
        "artifact": os.path.abspath(output_dir),
        "artifact_sha256": _root_hashes(output_dir),
    }
    return _write_once(record_path, payload)


def endpoint_state(result_dir, branch, binding, package="042"):
    manifest_path = os.path.join(result_dir, f"{package}_{branch}_eval_manifest.json")
    if not os.path.exists(manifest_path):
        return "absent"
    manifest = read_json(manifest_path)
    if manifest.get("completed") is not True or manifest.get("binding") != binding:
        return "tampered"
    hashes = manifest.get("sha256", {})
    for name in ENDPOINT_FILES[branch]:
        path = os.path.join(result_dir, name)
        if not os.path.exists(path) or hashes.get(name) != sha256(path):
            return "tampered"
    return "committed"


def _validate_endpoint_scratch(scratch_dir, branch):
    # verify both instruments cover the same ordered test items before commit
    predictions = {}
    for name in (f"{branch}_preds.jsonl", f"{branch}_logprob_preds.jsonl"):
        rows = read_jsonl(os.path.join(scratch_dir, name))
        if len(rows) != 1884:
            raise IntegrityError(f"{scratch_dir}: {name} does not contain 1884 rows")
        predictions[name] = rows
    generated = predictions[f"{branch}_preds.jsonl"]
    probe = predictions[f"{branch}_logprob_preds.jsonl"]
    for left, right in zip(generated, probe):
        if (left.get("subject"), left.get("tier"), left.get("gold")) != (
                right.get("subject"), right.get("tier"), right.get("gold")):
            raise IntegrityError(f"{scratch_dir}: endpoint rows are not aligned")
    for metrics_name, preds_name in (
            (f"{branch}_eval.json", f"{branch}_preds.jsonl"),
            (f"{branch}_logprob.json", f"{branch}_logprob_preds.jsonl")):
        metrics = read_json(os.path.join(scratch_dir, metrics_name))
        rows = predictions[preds_name]
        if metrics.get("n_items") != 1884:
            raise IntegrityError(f"{scratch_dir}: {metrics_name} n_items is not 1884")
        by_tier = {}
        for row in rows:
            by_tier.setdefault(row["tier"], []).append(bool(row["correct"]))
        computed = {tier: sum(values) / len(values)
                    for tier, values in by_tier.items()}
        recorded = metrics.get("per_tier_acc", {})
        if set(recorded) != set(computed) or any(
                abs(float(recorded[tier]) - value) > 1.0e-9
                for tier, value in computed.items()):
            raise IntegrityError(f"{scratch_dir}: {metrics_name} differs from predictions")


def commit_endpoint(result_dir, scratch_dir, branch, binding, package="042"):
    state = endpoint_state(result_dir, branch, binding, package=package)
    if state == "tampered":
        raise IntegrityError(f"{result_dir}: committed {branch} endpoint changed")
    manifest_path = os.path.join(result_dir, f"{package}_{branch}_eval_manifest.json")
    if state == "committed":
        return read_json(manifest_path)
    for name in ENDPOINT_FILES[branch]:
        if not os.path.exists(os.path.join(scratch_dir, name)):
            raise IntegrityError(f"{scratch_dir}: endpoint is missing {name}")
    _validate_endpoint_scratch(scratch_dir, branch)
    os.makedirs(result_dir, exist_ok=True)
    found = [name for name in ENDPOINT_FILES[branch]
             if os.path.exists(os.path.join(result_dir, name))]
    if found:
        # preserve partial contact from an interrupted manifest transaction
        archive = os.path.join(
            result_dir, "orphaned_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(archive)
        for name in found:
            shutil.move(os.path.join(result_dir, name), os.path.join(archive, name))
    hashes = {}
    for name in ENDPOINT_FILES[branch]:
        stage_into(result_dir, os.path.join(scratch_dir, name), name)
        hashes[name] = sha256(os.path.join(result_dir, name))
    payload = {"completed": True, "experiment": package, "branch": branch,
               "binding": binding, "sha256": hashes}
    atomic_write_json(manifest_path, payload)
    return payload


def _check_a1_record(result_dir, record):
    from config import load_config

    status_path = os.path.join(result_dir, "grpo_worker_status.json")
    sft_path = os.path.join(result_dir, "042_sft_training.json")
    seed = int(read_json(status_path)["seed"])
    cfg = load_config(config_path(seed))
    # only the exact registered truncation stop may replace a GRPO endpoint
    if (record.get("completed") is not True or record.get("outcome") != "a1"
            or record.get("status_sha256") != sha256(status_path)
            or record.get("sft_record_sha256") != sha256(sft_path)
            or record.get("wandb_runs") != _wandb_records(result_dir, cfg)):
        raise IntegrityError(f"{result_dir}: A1 conclusion changed")
    status = read_json(status_path)
    if (status.get("completed") is not False or status.get("cuda_oom") is not False
            or status.get("error_type") != "RuntimeError"
            or status.get("error_message") != A1_ERROR):
        raise IntegrityError(f"{result_dir}: A1 status changed")
    for name in ("042_grpo_training.json", "042_grpo_eval_manifest.json",
                 "042_arm_complete.json"):
        if os.path.exists(os.path.join(result_dir, name)):
            raise IntegrityError(f"{result_dir}: A1 arm gained {name}")
    return record


def a1_state(result_dir):
    record_path = os.path.join(result_dir, "042_a1_conclusion.json")
    if os.path.exists(record_path):
        return "committed", _check_a1_record(result_dir, read_json(record_path))
    status_path = os.path.join(result_dir, "grpo_worker_status.json")
    if not os.path.exists(status_path):
        return "absent", None
    status = read_json(status_path)
    if (status.get("completed") is False and status.get("cuda_oom") is False
            and status.get("error_type") == "RuntimeError"
            and status.get("error_message") == A1_ERROR):
        return "pending", status
    if status.get("error_type") == "KeyboardInterrupt":
        return "interrupted", status
    if status.get("completed") is True:
        return "absent", status
    return "failed", status


def commit_a1_conclusion(result_dir):
    state, _ = a1_state(result_dir)
    if state == "committed":
        return read_json(os.path.join(result_dir, "042_a1_conclusion.json"))
    if state != "pending":
        raise IntegrityError(f"{result_dir}: no exact A1 outcome to conclude")
    status_path = os.path.join(result_dir, "grpo_worker_status.json")
    sft_path = os.path.join(result_dir, "042_sft_training.json")
    if not os.path.exists(sft_path):
        raise IntegrityError(f"{result_dir}: A1 outcome has no SFT record")
    from config import load_config
    seed = int(read_json(status_path)["seed"])
    cfg = load_config(config_path(seed))
    payload = {
        "completed": True,
        "experiment": "042",
        "outcome": "a1",
        "status_sha256": sha256(status_path),
        "sft_record_sha256": sha256(sft_path),
        "wandb_runs": _wandb_records(result_dir, cfg),
    }
    record = _write_once(os.path.join(result_dir, "042_a1_conclusion.json"), payload)
    return _check_a1_record(result_dir, record)


def commit_arm_complete(result_dir):
    sft_record = read_json(os.path.join(result_dir, "042_sft_training.json"))
    grpo_record = read_json(os.path.join(result_dir, "042_grpo_training.json"))
    if endpoint_state(result_dir, "sft", sft_record) != "committed":
        raise IntegrityError(f"{result_dir}: SFT endpoint is not committed")
    if endpoint_state(result_dir, "grpo", grpo_record) != "committed":
        raise IntegrityError(f"{result_dir}: GRPO endpoint is not committed")
    records = {}
    for name in ("042_sft_training.json", "042_grpo_training.json",
                 "042_sft_eval_manifest.json", "042_grpo_eval_manifest.json"):
        path = os.path.join(result_dir, name)
        if not os.path.exists(path):
            raise IntegrityError(f"{result_dir}: arm completion is missing {name}")
        records[name] = sha256(path)
    return _write_once(
        os.path.join(result_dir, "042_arm_complete.json"),
        {"completed": True, "experiment": "042", "sha256": records})


def arm_state(result_dir):
    a1, record = a1_state(result_dir)
    if a1 == "committed":
        return {"state": "a1", "record": record}
    if a1 in ("pending", "interrupted", "failed"):
        return {"state": a1, "status": record}
    complete_path = os.path.join(result_dir, "042_arm_complete.json")
    if not os.path.exists(complete_path):
        return {"state": "absent"}
    complete = read_json(complete_path)
    if complete.get("completed") is not True:
        raise IntegrityError(f"{result_dir}: arm completion changed")
    for name, digest in complete.get("sha256", {}).items():
        path = os.path.join(result_dir, name)
        if not os.path.exists(path) or sha256(path) != digest:
            raise IntegrityError(f"{result_dir}: {name} changed after completion")
    commit_arm_complete(result_dir)
    return {"state": "committed", "record": complete}


def pilot_passes(report, closeout):
    # memory, numerics and cleanup must all pass before science starts
    precision = report.get("precision_audit", {})
    changes = precision.get("module_changes", {})
    return bool(
        report.get("completed_updates") == 30
        and report.get("peak_reserved_fraction", 1.0) <= 0.95
        and report.get("projected_500_updates_hours", 25.0) <= 24.0
        and report.get("checkpoint_bytes", 0) > 0
        and precision.get("gradient_dtypes") == ["float32"]
        and precision.get("optimizer_state_dtypes") == ["float32"]
        and precision.get("reference_gate", {}).get("verified") is True
        and {"embeddings", "attention", "mlp", "normalization"} <= set(changes)
        and all(row.get("changed_elements", 0) > 0 for row in changes.values())
        and closeout.get("weights_deleted") is True)


def _latest_checkpoint_step(out_dir):
    values = []
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if name.startswith("checkpoint-"):
                try:
                    values.append(int(name.rsplit("-", 1)[1]))
                except ValueError:
                    pass
    return max(values, default=0)


def _run_sft(cfg, base_dir, out_dir, result_dir, pilot=False):
    from data.gsm8k import load_sft
    from train.sft import run_sft_rows

    def verify_sft_setup(trainer):
        # keep official GPT-2 dropout active during supervised training
        model = trainer.accelerator.unwrap_model(trainer.model)
        config = model.config
        if (config.embd_pdrop != 0.1 or config.attn_pdrop != 0.1
                or config.resid_pdrop != 0.1):
            raise IntegrityError("experiment 042 SFT dropout config changed")

    if pilot:
        cfg.train.max_steps = 1
        cfg.train.sft_val_examples = 0
        cfg.log.save_steps = 1
        cfg.log.save_total_limit = 1
    rows = load_sft(limit=16 if pilot else None)
    eval_rows = None if pilot else load_sft(
        split="test", limit=cfg.train.sft_val_examples)
    run_sft_rows(
        cfg, "cuda", rows, base_dir, "local",
        os.path.relpath(base_dir, cfg.output_root), out_dir=out_dir,
        report_to=[] if pilot else ["wandb"], log_dir=result_dir,
        run_name=f"{os.path.basename(result_dir)}_sft",
        allow_oom_fallback=False, eval_rows=eval_rows,
        trainer_setup=verify_sft_setup)
    if pilot:
        return {"completed": True, "updates": 1, "artifact": os.path.abspath(out_dir)}
    return commit_sft_training(
        cfg, result_dir, out_dir,
        os.path.join(cfg.output_root, "results", BASE_TAG, "042_base_manifest.json"))


def _verify_grpo_runtime(trainer):
    import torch

    # check the policy and frozen reference before the first update
    reference = getattr(trainer, "ref_model", None)
    if reference is None:
        raise IntegrityError("experiment 042 full GRPO has no frozen reference")
    for name, model in (("policy", trainer.model), ("reference", reference)):
        if any(module.p != 0.0 for module in model.modules()
               if isinstance(module, torch.nn.Dropout)):
            raise IntegrityError(f"experiment 042 {name} dropout was not disabled")
    if any(param.requires_grad for param in reference.parameters()):
        raise IntegrityError("experiment 042 GRPO reference is not frozen")


def _completion_length_gate(seen, package="042"):
    from transformers import TrainerCallback

    class completion_length_gate(TrainerCallback):
        def __init__(self):
            self.checked = False

        def on_log(self, args, state, control, logs=None, **kwargs):
            if (self.checked or not logs
                    or "rewards/accuracy_reward/mean" not in logs):
                return control
            if "completions/max_length" not in logs:
                raise IntegrityError(
                    f"experiment {package} GRPO log is missing completion length")
            value = float(logs["completions/max_length"])
            if not math.isfinite(value) or value < 0:
                raise IntegrityError(
                    f"experiment {package} GRPO completion length is invalid")
            seen.append(value)
            self.checked = True
            return control

    return completion_length_gate()


def _run_grpo(cfg, start_dir, out_dir, result_dir, precision_path, pilot=False):
    from data.splits import build_grpo_split
    from scripts.s35_matched_grpo import PrecisionAudit
    from train.grpo import run_grpo_rows

    if pilot:
        cfg.log.save_steps = 30
        cfg.log.save_total_limit = 1
    audit = PrecisionAudit(stop_after=30 if pilot else None)
    wandb_recorded = []
    completion_length_seen = []

    def verify_grpo_setup(trainer):
        from transformers import TrainerCallback

        _verify_grpo_runtime(trainer)
        audit.attach(trainer)
        trainer.add_callback(_completion_length_gate(completion_length_seen))
        if not pilot:
            class wandb_capture(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if state.global_step > 0 and not wandb_recorded:
                        wandb_recorded.append(_capture_wandb_run(
                            result_dir, cfg, state.global_step))
                    return control

            trainer.add_callback(wandb_capture())

    def finished(trainer):
        # record the observed lengths and cache state beside the precision gate
        report = audit.finish(trainer)
        history = trainer.state.log_history
        completion_lengths = [float(row["completions/max_length"])
                              for row in history
                              if row.get("completions/max_length") is not None]
        if not completion_length_seen or not completion_lengths:
            raise IntegrityError("experiment 042 GRPO completion length was not recorded")
        prompt_lengths = [
            len(trainer.processing_class.encode(
                row["prompt"], add_special_tokens=False))
            for row in build_grpo_split()
        ]
        report.update({
            "cache_enabled": bool(getattr(trainer.model.config, "use_cache", False)),
            "max_prompt_tokens": max(prompt_lengths),
            "max_observed_completion_tokens": max(completion_lengths),
            "completion_cap": cfg.grpo.max_completion_length,
        })
        atomic_write_json(precision_path, report)

    run_grpo_rows(
        cfg, "cuda", build_grpo_split(), start_dir, "local",
        os.path.relpath(start_dir, cfg.output_root), out_dir=out_dir,
        report_to=[] if pilot else ["wandb"], log_dir=result_dir,
        run_name=f"{os.path.basename(result_dir)}_grpo",
        allow_oom_fallback=False, trainer_setup=verify_grpo_setup,
        trainer_finished=finished, save_final=not pilot)
    if pilot:
        return {"precision": read_json(precision_path)}
    return commit_grpo_training(
        cfg, result_dir, out_dir,
        os.path.join(result_dir, "042_sft_training.json"), precision_path)


def _run_eval(cfg, branch, model_path, scratch_dir):
    from data.splits import build_eval_split
    from eval.harness import run_eval_rows, run_logprob_rows

    if os.path.exists(scratch_dir):
        if os.listdir(scratch_dir):
            raise IntegrityError(f"{scratch_dir}: evaluation scratch is not empty")
    else:
        os.makedirs(scratch_dir)
    rows = build_eval_split()
    generated = run_eval_rows(
        cfg, "cuda", rows, model_path=model_path, condition=branch,
        out_dir=scratch_dir)
    probe = run_logprob_rows(
        cfg, "cuda", rows, model_path=model_path, condition=branch,
        out_dir=scratch_dir)
    atomic_write_json(os.path.join(scratch_dir, f"{branch}_eval.json"), generated)
    atomic_write_json(os.path.join(scratch_dir, f"{branch}_logprob.json"), probe)
    return {"generated": generated, "probe": probe}


write_once = _write_once
wandb_records = _wandb_records
capture_wandb_run = _capture_wandb_run
step_rows = _step_rows
starvation_summary = _starvation_summary
completion_length_gate = _completion_length_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("audit", "convert", "pilot_sft", "pilot",
                                            "sft", "grpo", "eval"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--base-dir")
    parser.add_argument("--start-dir")
    parser.add_argument("--scratch-dir")
    parser.add_argument("--record")
    parser.add_argument("--precision-report")
    parser.add_argument("--pilot-report")
    parser.add_argument("--pilot-closeout")
    parser.add_argument("--branch", choices=("base", "sft", "grpo"))
    args = parser.parse_args()

    required = {
        "audit": ("record",),
        "convert": ("scratch_dir", "out_dir", "record"),
        "pilot_sft": ("base_dir", "out_dir"),
        "pilot": ("start_dir", "out_dir", "precision_report", "pilot_report",
                  "pilot_closeout"),
        "sft": ("base_dir", "out_dir"),
        "grpo": ("start_dir", "out_dir", "precision_report"),
        "eval": ("start_dir", "scratch_dir", "record", "branch"),
    }
    missing = [name for name in required[args.task] if not getattr(args, name)]
    if missing:
        raise ValueError(f"{args.task} is missing arguments: {missing}")

    import torch
    from config import load_config
    from obs import _versions, gpu_snapshot
    from train.model_utils import set_all_seeds

    cfg = load_config(args.config)
    if (cfg.experiment != "042" or cfg.scale.name != "M124"
            or cfg.train.seed not in SEEDS_042
            or cfg.train.sft_parameter_mode != "full"
            or cfg.train.grpo_parameter_mode != "full"
            or cfg.exec.allow_oom_fallback is not False):
        raise IntegrityError("worker config differs from experiment 042")
    set_all_seeds(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    status = {"completed": False, "task": args.task, "seed": cfg.train.seed,
              "gpu": gpu_snapshot(), "versions": _versions()}
    started = time.time()
    failure = None
    result = None
    start_step = _latest_checkpoint_step(args.out_dir or "")
    try:
        if args.task == "audit":
            result = build_length_audit(cfg, args.record)
        elif args.task == "convert":
            result = convert_base(cfg, args.scratch_dir, args.out_dir, args.record)
        elif args.task in ("pilot_sft", "sft"):
            result = _run_sft(
                cfg, args.base_dir, args.out_dir, args.result_dir,
                pilot=args.task == "pilot_sft")
        elif args.task in ("pilot", "grpo"):
            result = _run_grpo(
                cfg, args.start_dir, args.out_dir, args.result_dir,
                args.precision_report, pilot=args.task == "pilot")
        else:
            binding = read_json(args.record)
            state = endpoint_state(args.result_dir, args.branch, binding)
            if state == "tampered":
                raise IntegrityError(f"{args.result_dir}: endpoint changed")
            if state == "committed":
                result = read_json(os.path.join(
                    args.result_dir, f"042_{args.branch}_eval_manifest.json"))
            else:
                _run_eval(cfg, args.branch, args.start_dir, args.scratch_dir)
                result = commit_endpoint(
                    args.result_dir, args.scratch_dir, args.branch, binding)
        status["completed"] = True
    except BaseException as exc:
        failure = exc
        status.update(classify_failure(exc, torch))
    finally:
        status["wall_seconds"] = round(time.time() - started, 1)
        status["start_step"] = start_step
        status["end_step"] = _latest_checkpoint_step(args.out_dir or "")
        status["completed_updates"] = max(0, status["end_step"] - start_step)
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            peak = torch.cuda.max_memory_reserved()
            status["peak_reserved_gib"] = round(peak / 1024**3, 2)
            status["peak_reserved_fraction"] = peak / total
        else:
            status["peak_reserved_gib"] = 0.0
            status["peak_reserved_fraction"] = 0.0
        if result is not None:
            status["result"] = result
        save_status(args.status, status)
        if args.task == "pilot" and (status.get("completed") or status.get("cuda_oom")):
            # an OOM remains a technical pilot result even though it cannot pass
            history = read_json(os.path.join(args.result_dir, "grpo_trainlog.json"))
            times = [float(row["step_time"]) for row in history if row.get("step_time") is not None]
            if status.get("completed") and len(times) != 30:
                raise IntegrityError("experiment 042 pilot step timing is incomplete")
            checkpoint = os.path.join(args.out_dir, "checkpoint-30")
            checkpoint_bytes = sum(
                os.path.getsize(os.path.join(root, name))
                for root, _, names in os.walk(checkpoint) for name in names)
            report = {
                **status,
                "precision_audit": read_json(args.precision_report),
                "mean_step_seconds": sum(times) / len(times) if times else None,
                "max_step_seconds": max(times) if times else None,
                "projected_500_updates_hours": (
                    sum(times) / len(times) * 500 / 3600 if times else None),
                "checkpoint_bytes": checkpoint_bytes,
                "scientific_metrics_used": False,
                "test_evaluation_calls": 0,
            }
            pilot_record = _write_once(args.pilot_report, report)
            shutil.rmtree(args.out_dir)
            if args.start_dir and os.path.exists(args.start_dir):
                shutil.rmtree(args.start_dir)
            closeout = _write_once(args.pilot_closeout, {
                "completed": True,
                "pilot_report_sha256": sha256(args.pilot_report),
                "weights_deleted": not os.path.exists(args.out_dir)
                    and not os.path.exists(args.start_dir),
            })
            if status.get("completed") and not pilot_passes(pilot_record, closeout):
                raise RuntimeError("experiment 042 pilot did not pass")
        if failure is not None:
            raise failure


if __name__ == "__main__":
    main()
