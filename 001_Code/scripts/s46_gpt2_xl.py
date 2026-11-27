# GPT-2 XL, records and completion checks
import os

from checkpoint import check_artifact_complete, config_hash
from config import load_config
from file_utils import read_json
from scripts.s12_eval import IntegrityError, sha256, stage_into
from scripts.s28_compose import adapter_hashes_of, verify_binding
from scripts.s42_worker import (_root_hashes, _starvation_summary, _step_rows,
                                _write_once, commit_endpoint, endpoint_state)


SEEDS_046 = (0, 1, 2)
CONFIGS_046 = tuple(
    f"configs/046_colab_gpt2_b15_seed{seed}.yaml" for seed in SEEDS_046)
BASE_TAG = "046_COLAB_FULL_GPT2_B15_base"
PILOT_ROOT = "046_technical"
A1_ERROR = (
    "sentinel abort: a1: over 30% of rollouts truncated in the first 50 steps. "
    "raise the completion cap or mask truncated completions"
)
PILOT_RETRY_FILES = (
    "pilot_failure.json", "pilot_report.json", "pilot_closeout.json",
    "pilot_sft_status.json", "pilot_grpo_a_status.json",
    "pilot_grpo_b_status.json", "precision_a.json", "precision_b.json",
    "sft_trainlog.json", "grpo_trainlog.json",
)
PILOT_RETRY_DIR = "pilot_80gb_before_logit_projection_fix"
ALLOCATOR_CONFIG = "expandable_segments:True"
RESUME_MEMORY_LABELS = (
    "before trainer",
    "trainer ready",
    "before _load_from_checkpoint",
    "after _load_from_checkpoint",
    "after _load_from_checkpoint cleanup",
    "before _load_optimizer_and_scheduler",
    "after _load_optimizer_and_scheduler",
    "after _load_optimizer_and_scheduler cleanup",
    "training finished",
)


def config_path(seed):
    if seed not in SEEDS_046:
        raise ValueError(f"experiment 046 does not register seed {seed}")
    return CONFIGS_046[seed]


def arm_tag(seed):
    if seed not in SEEDS_046:
        raise ValueError(f"experiment 046 does not register seed {seed}")
    return f"046_COLAB_FULL_GPT2_B15_sft_grpo_seed{seed}"


def check_registered_config(cfg):
    if (cfg.experiment != "046" or not cfg.exploratory
            or cfg.scale.name != "GPT2_B15" or cfg.condition.name != "SFT_GRPO"
            or cfg.train.seed not in SEEDS_046
            or cfg.train.sft_parameter_mode != "lora"
            or cfg.train.grpo_parameter_mode != "lora"
            or cfg.train.sft_precision_mode != "fp32_master_bf16"
            or cfg.train.grpo_precision_mode != "fp32_master_bf16"
            or cfg.train.a2_action != "observe"
            or cfg.exec.grpo_batch_size != 8 or cfg.exec.grpo_grad_accum != 1
            or cfg.exec.gradient_checkpointing is not False
            or cfg.exec.allow_oom_fallback is not False):
        raise IntegrityError("experiment 046 config differs from the registered package")
    return cfg


def commit_token_audit(cfg, base_record_path, source_audit_path,
                       source_base_record_path, path, package="046",
                       config_check=None):
    from data.gsm8k import load_sft
    from data.splits import build_grpo_split
    from train.model_utils import load_scratch_tokenizer
    from train.sft import build_sft_rows, mc_sft_rows

    (config_check or check_registered_config)(cfg)
    base = read_json(base_record_path)
    source_base = read_json(source_base_record_path)
    source = read_json(source_audit_path)
    for record in (base, source_base):
        check_artifact_complete(record["artifact"], expected="full")
        if record.get("artifact_sha256") != _root_hashes(record["artifact"]):
            raise IntegrityError("scratch GPT base artifact changed before the token audit")
    # compare the shared tokenizer files before reusing the earlier test audit
    tokenizer_files = (
        "merges.txt", "special_tokens_map.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json",
    )
    compared = {}
    for name in tokenizer_files:
        left = base["artifact_sha256"].get(name)
        right = source_base["artifact_sha256"].get(name)
        if left is not None or right is not None:
            if left != right:
                raise IntegrityError(f"scratch GPT tokenizer asset differs from 042: {name}")
            compared[name] = left

    tokenizer = load_scratch_tokenizer(cfg, base["artifact"])

    def lengths(values):
        return [len(tokenizer.encode(value, add_special_tokens=False)) for value in values]

    sft_rows = build_sft_rows(
        [*load_sft(), *load_sft(split="test", limit=cfg.train.sft_val_examples)],
        tokenizer.eos_token or "", True)
    grpo_rows = build_grpo_split()
    sft_lengths = lengths(row["prompt"] + row["completion"] for row in sft_rows)
    grpo_lengths = lengths(row["prompt"] for row in grpo_rows)
    mc_lengths = lengths(row["prompt"] + row["target"] for row in mc_sft_rows(grpo_rows))
    letter_tokens = {
        letter: tokenizer.encode(" " + letter, add_special_tokens=False)
        for letter in "ABCD"
    }
    current = {
        "sft_rows": len(sft_rows),
        "grpo_rows": len(grpo_rows),
        "max_sft_tokens": max(sft_lengths),
        "max_grpo_prompt_tokens": max(grpo_lengths),
        "max_mc_sft_tokens": max(mc_lengths),
        "completion_cap": min(512, 1024 - max(grpo_lengths)),
        "letter_token_ids": {key: value[0] for key, value in letter_tokens.items()},
    }
    if any(len(value) != 1 for value in letter_tokens.values()):
        raise IntegrityError(f"experiment {package} probe letters are not single tokens")
    for key, value in current.items():
        if source.get(key) != value:
            raise IntegrityError(f"experiment {package} token audit differs from 042: {key}")
    if (current["completion_cap"] != cfg.grpo.max_completion_length
            or current["completion_cap"] != cfg.eval.max_new_tokens):
        raise IntegrityError(f"experiment {package} config differs from the token audit")
    # keep held out row details behind the hash already committed by 042
    payload = {
        "completed": True,
        "experiment": package,
        **current,
        "tokenizer_assets": compared,
        "source_audit": os.path.abspath(source_audit_path),
        "source_audit_sha256": sha256(source_audit_path),
        "source_test_rows": source["test_rows"],
        "source_test_row_ids_sha256": source["test_row_ids_sha256"],
        "base_record_sha256": sha256(base_record_path),
    }
    return _write_once(path, payload)


def pilot_passes(report, closeout):
    precision = report.get("precision_audit", {})
    reference = precision.get("reference_gate", {})
    reference_behaviour = precision.get("reference_behaviour", {})
    adapter_base = precision.get("adapter_base", {})
    lora = precision.get("lora_setup", {})
    sft_lora = report.get("sft_lora_setup", {})
    resume_labels = tuple(
        row.get("label") for row in report.get("resume_memory", []))
    return bool(
        report.get("sft_updates") == 5
        and report.get("grpo_updates") == 30
        and report.get("allocator_config") == ALLOCATOR_CONFIG
        and resume_labels == RESUME_MEMORY_LABELS
        and report.get("peak_reserved_fraction", 1.0) <= 0.95
        and report.get("adapter_reload_verified") is True
        and report.get("merge_verified") is True
        and report.get("resume_verified") is True
        and precision.get("gradient_dtypes") == ["float32"]
        and precision.get("optimizer_state_dtypes") == ["float32"]
        and precision.get("bf16_enabled") is True
        and precision.get("accelerator_mixed_precision") == "bf16"
        and reference.get("reference_kind") == "adapter_disabled"
        and reference_behaviour.get("verified") is True
        and reference_behaviour.get("reference_max_abs_drift", 1) <= 1.0e-3
        and reference_behaviour.get("adapter_effect_max_abs", 0) > 0
        and adapter_base.get("verified") is True
        and lora.get("verified") is True
        and lora.get("trainable_parameters", 0) > 0
        and 0 < lora.get("trainable_fraction", 1) < 0.05
        and set(lora.get("target_module_counts", {}).values()) == {48}
        and sft_lora.get("verified") is True
        and sft_lora.get("targets") == lora.get("targets")
        and sft_lora.get("target_module_counts") == lora.get("target_module_counts")
        and closeout.get("completed") is True
        and closeout.get("weights_deleted") is True)


def archive_memory_margin_pilot(
        card_dir, archive_name=PILOT_RETRY_DIR,
        reason="logit projection memory correction"):
    archive_dir = os.path.join(os.path.dirname(card_dir), archive_name)
    manifest_path = os.path.join(archive_dir, "archive_manifest.json")
    # finish an interrupted cleanup before any retry uses the active names
    if os.path.exists(manifest_path):
        manifest = read_json(manifest_path)
        if (manifest.get("completed") is not True
                or set(manifest.get("files", {})) != set(PILOT_RETRY_FILES)):
            raise IntegrityError("experiment 046 pilot archive changed")
        for name, expected in manifest["files"].items():
            archived = os.path.join(archive_dir, name)
            if not os.path.exists(archived) or sha256(archived) != expected:
                raise IntegrityError("experiment 046 pilot archive changed")
            active = os.path.join(card_dir, name)
            if os.path.exists(active):
                if sha256(active) != expected:
                    raise IntegrityError("experiment 046 active pilot differs from its archive")
                os.remove(active)
        return True

    failure_path = os.path.join(card_dir, "pilot_failure.json")
    if not os.path.exists(failure_path):
        return False
    report_path = os.path.join(card_dir, "pilot_report.json")
    closeout_path = os.path.join(card_dir, "pilot_closeout.json")
    failure = read_json(failure_path)
    report = read_json(report_path)
    closeout = read_json(closeout_path)
    # only the completed 80GB margin stop belongs in this archive
    if (failure.get("completed") is not True
            or failure.get("failure_kind") != "memory"
            or failure.get("stage") != "memory margin"
            or failure.get("report_sha256") != sha256(report_path)
            or report.get("completed") is not True
            or report.get("experiment") != "046"
            or report.get("card_requested") != 80
            or report.get("sft_updates") != 5
            or report.get("grpo_updates") != 30
            or report.get("peak_reserved_fraction", 0) <= 0.95
            or report.get("scientific_metrics_used") is not False
            or report.get("mmlu_test_passes") != 0
            or closeout.get("completed") is not True
            or closeout.get("weights_deleted") is not True
            or closeout.get("pilot_report_sha256") != sha256(report_path)):
        raise IntegrityError("experiment 046 memory pilot is not the registered retry state")
    status_steps = {
        "pilot_sft_status.json": ("pilot_sft", 5),
        "pilot_grpo_a_status.json": ("pilot_grpo_a", 15),
        "pilot_grpo_b_status.json": ("pilot_grpo_b", 30),
    }
    for name, (task, step) in status_steps.items():
        status = read_json(os.path.join(card_dir, name))
        if (status.get("completed") is not True or status.get("task") != task
                or status.get("end_step") != step):
            raise IntegrityError("experiment 046 memory pilot status changed")
    if any(not os.path.exists(os.path.join(card_dir, name))
           for name in PILOT_RETRY_FILES):
        raise IntegrityError("experiment 046 memory pilot files are incomplete")

    os.makedirs(archive_dir, exist_ok=True)
    hashes = {}
    for name in PILOT_RETRY_FILES:
        source = os.path.join(card_dir, name)
        archived = os.path.join(archive_dir, name)
        if os.path.exists(archived):
            if sha256(archived) != sha256(source):
                raise IntegrityError("experiment 046 partial pilot archive changed")
        else:
            stage_into(archive_dir, source, name)
        hashes[name] = sha256(archived)
    _write_once(manifest_path, {
        "completed": True,
        "experiment": "046",
        "reason": reason,
        "source": os.path.abspath(card_dir),
        "files": hashes,
    })
    for name in PILOT_RETRY_FILES:
        os.remove(os.path.join(card_dir, name))
    return True


def commit_card_lock(output_root, report_path, closeout_path):
    report = read_json(report_path)
    closeout = read_json(closeout_path)
    if (not pilot_passes(report, closeout)
            or closeout.get("pilot_report_sha256") != sha256(report_path)):
        raise IntegrityError("experiment 046 pilot did not pass")
    total = float(report.get("total_vram_gib", 0))
    if not (39 <= total < 48 or total >= 79):
        raise IntegrityError("experiment 046 pilot used an unregistered card")
    card = "A100_40GB" if total < 48 else "A100_80GB"
    failed_40_sha = None
    if card == "A100_80GB":
        failed_40 = os.path.join(
            output_root, "results", PILOT_ROOT, "pilot_40gb", "pilot_failure.json")
        failed_40_sha = sha256(failed_40)
        if report.get("failed_40_sha256") != failed_40_sha:
            raise IntegrityError("experiment 046 80GB pilot has no bound 40GB failure")
    path = os.path.join(output_root, "results", PILOT_ROOT, "card_lock.json")
    # one pilot selected card applies to every seed in the cell
    payload = {
        "completed": True,
        "experiment": "046",
        "card": card,
        "minimum_vram_gib": 39 if card == "A100_40GB" else 79,
        "pilot_report": os.path.abspath(report_path),
        "pilot_closeout": os.path.abspath(closeout_path),
        "pilot_report_sha256": sha256(report_path),
        "pilot_closeout_sha256": sha256(closeout_path),
        "failed_40_sha256": failed_40_sha,
        "allocator_config": report.get("allocator_config"),
        "config_hashes": {
            str(seed): config_hash(check_registered_config(
                load_config(config_path(seed))))
            for seed in SEEDS_046
        },
    }
    return _write_once(path, payload)


def check_card_lock(cfg):
    check_registered_config(cfg)
    path = os.path.join(cfg.output_root, "results", PILOT_ROOT, "card_lock.json")
    record = read_json(path)
    if (record.get("completed") is not True or record.get("experiment") != "046"
            or record.get("config_hashes", {}).get(str(cfg.train.seed)) != config_hash(cfg)
            or record.get("card") not in ("A100_40GB", "A100_80GB")
            or record.get("allocator_config") != ALLOCATOR_CONFIG
            or sha256(record.get("pilot_report", ""))
            != record.get("pilot_report_sha256")
            or sha256(record.get("pilot_closeout", ""))
            != record.get("pilot_closeout_sha256")):
        raise IntegrityError("experiment 046 card lock changed")
    return record


def check_card_vram(lock, total_vram_gib):
    total = float(total_vram_gib)
    expected = "A100_40GB" if 39 <= total < 48 else (
        "A100_80GB" if total >= 79 else None)
    if expected != lock.get("card"):
        raise IntegrityError("experiment 046 runtime differs from the card lock")
    return expected


def verify_technical_records(cfg, base_record_path, parity_path, audit_path,
                             package="046", config_check=None):
    from gpt2_scratch.weights import reference_checkpoint
    from scripts.s42_worker import verify_artifact_record
    from train.model_utils import resolve_model_id

    # recheck the technical records before a later session reaches an endpoint
    (config_check or check_registered_config)(cfg)
    base = verify_artifact_record(base_record_path)
    parity = read_json(parity_path)
    audit = read_json(audit_path)
    source, revision = reference_checkpoint(resolve_model_id(cfg))
    if (parity.get("completed") is not True or parity.get("experiment") != package
            or parity.get("source") != source
            or parity.get("source_revision") != revision
            or parity.get("base_record_sha256") != sha256(base_record_path)
            or parity.get("max_abs_unpadded_logit_error", 1) > 1.0e-4
            or parity.get("max_abs_padded_logit_error", 1) > 1.0e-4
            or parity.get("abs_loss_error", 1) > 1.0e-5
            or parity.get("max_abs_cached_logit_error", 1) > 3.0e-4
            or parity.get("reference_cache_error", 1) > 3.0e-4
            or parity.get("scratch_cache_error", 1) > 3.0e-4
            or parity.get("reload_logit_error", 1) > 1.0e-7
            or parity.get("max_new_tokens") != 32
            or parity.get("generation_tokens_equal") is not True
            or parity.get("generation_diagnostic", {}).get("first_divergence") is not None
            or parity.get("tied_parameter_and_storage") is not True):
        raise IntegrityError(f"experiment {package} parity record changed")
    if (audit.get("completed") is not True or audit.get("experiment") != package
            or audit.get("base_record_sha256") != sha256(base_record_path)
            or audit.get("completion_cap") != cfg.grpo.max_completion_length
            or sha256(audit.get("source_audit", ""))
            != audit.get("source_audit_sha256")):
        raise IntegrityError(f"experiment {package} token audit changed")
    return {"base": base, "parity": parity, "token_audit": audit}


def _adapter_record(output_dir):
    # bind adapter bytes rather than trusting a completed folder name
    check_artifact_complete(output_dir, expected="adapter")
    return {"kind": "adapter", "path": os.path.abspath(output_dir),
            "sha256": adapter_hashes_of(output_dir)}


def verify_grpo_adapter_base(cfg, adapter_dir, merged_dir):
    from train.model_utils import resolve_adapter_base

    record_path = os.path.join(adapter_dir, "adapter_base.json")
    resolved = resolve_adapter_base(adapter_dir, cfg)
    expected = os.path.realpath(merged_dir)
    if os.path.realpath(resolved) != expected:
        raise IntegrityError("experiment 046 GRPO adapter base differs from SFT merge")
    return {
        "verified": True,
        "adapter_base_sha256": sha256(record_path),
        "resolved_path": expected,
        "merged_path": expected,
    }


def commit_sft_training(cfg, result_dir, output_dir, base_record_path, wandb_records):
    check_registered_config(cfg)
    if not wandb_records:
        raise IntegrityError("experiment 046 SFT W&B records are missing")
    rows = _step_rows(os.path.join(result_dir, "sft_trainlog.json"), cfg.train.max_steps)
    base = read_json(base_record_path)
    payload = {
        "completed": True,
        "experiment": "046",
        "stage": "sft",
        "seed": cfg.train.seed,
        "config_hash": config_hash(cfg),
        "updates": len(rows),
        "parameter_mode": "lora",
        "storage_dtype": "float32",
        "compute_dtype": "bfloat16",
        "base_record_sha256": sha256(base_record_path),
        "base_artifact_sha256": base["artifact_sha256"],
        "wandb_runs": wandb_records,
        "artifact": _adapter_record(output_dir),
    }
    return _write_once(os.path.join(result_dir, "046_sft_training.json"), payload)


def commit_grpo_training(cfg, result_dir, output_dir, sft_record_path,
                         merged_dir, precision_path):
    from scripts.s42_worker import _wandb_records
    check_registered_config(cfg)
    fields = ("rewards/accuracy_reward/mean", "rewards/format_reward/mean",
              "frac_reward_zero_std")
    rows = _step_rows(
        os.path.join(result_dir, "grpo_trainlog.json"), cfg.train.max_steps, fields)
    precision = read_json(precision_path)
    adapter_base = verify_grpo_adapter_base(cfg, output_dir, merged_dir)
    if (precision.get("gradient_dtypes") != ["float32"]
            or precision.get("optimizer_state_dtypes") != ["float32"]
            or precision.get("bf16_enabled") is not True
            or precision.get("accelerator_mixed_precision") != "bf16"
            or precision.get("reference_gate", {}).get("reference_kind")
            != "adapter_disabled"
            or precision.get("reference_behaviour", {}).get("verified") is not True
            or precision.get("adapter_base") != adapter_base
            or precision.get("lora_setup", {}).get("verified") is not True):
        raise IntegrityError("experiment 046 precision report is incomplete")
    verdict = os.path.join(output_dir, "a2_verdict.json")
    if not os.path.exists(verdict):
        raise IntegrityError("experiment 046 observe only A2 record is missing")
    binding = verify_binding(
        read_json(sft_record_path)["artifact"]["path"], merged_dir)
    payload = {
        "completed": True,
        "experiment": "046",
        "stage": "grpo",
        "seed": cfg.train.seed,
        "config_hash": config_hash(cfg),
        "updates": len(rows),
        "parameter_mode": "lora",
        "sft_record_sha256": sha256(sft_record_path),
        "merged_path": os.path.abspath(merged_dir),
        "merge_binding": binding,
        "adapter_base": adapter_base,
        "precision_report_sha256": sha256(precision_path),
        "a2_verdict_sha256": sha256(verdict),
        "wandb_runs": _wandb_records(
            result_dir, cfg, package="046", stage="grpo"),
        "starvation": _starvation_summary(rows),
        "artifact": _adapter_record(output_dir),
    }
    return _write_once(os.path.join(result_dir, "046_grpo_training.json"), payload)


def training_record(result_dir, branch):
    path = os.path.join(result_dir, f"046_{branch}_training.json")
    record = read_json(path)
    artifact = record.get("artifact", {})
    if (record.get("completed") is not True or record.get("experiment") != "046"
            or record.get("stage") != branch or artifact.get("kind") != "adapter"
            or adapter_hashes_of(artifact.get("path", "")) != artifact.get("sha256")):
        raise IntegrityError(f"{result_dir}: {branch} training record changed")
    for row in record.get("wandb_runs", []):
        if sha256(row.get("path", "")) != row.get("sha256"):
            raise IntegrityError(f"{result_dir}: {branch} W&B record changed")
    if not record.get("wandb_runs"):
        raise IntegrityError(f"{result_dir}: {branch} W&B records are missing")
    if branch == "grpo":
        binding = verify_binding(
            record.get("merge_binding", {}).get("adapter_dir", ""),
            record.get("merged_path", ""))
        if binding != record.get("merge_binding"):
            raise IntegrityError(f"{result_dir}: GRPO merge binding changed")
        base = record.get("adapter_base", {})
        if (base.get("verified") is not True
                or base.get("merged_path") != os.path.realpath(
                    record.get("merged_path", ""))
                or base.get("resolved_path") != base.get("merged_path")
                or sha256(os.path.join(
                    artifact.get("path", ""), "adapter_base.json"))
                != base.get("adapter_base_sha256")):
            raise IntegrityError(f"{result_dir}: GRPO adapter base changed")
    return record


def eval_state(result_dir, branch, binding):
    return endpoint_state(result_dir, branch, binding, package="046")


def commit_eval(result_dir, scratch_dir, branch, binding):
    return commit_endpoint(
        result_dir, scratch_dir, branch, binding, package="046")


def a1_state(result_dir):
    # only the exact recorded truncation stop can replace a GRPO endpoint
    record_path = os.path.join(result_dir, "046_a1_conclusion.json")
    if os.path.exists(record_path):
        record = read_json(record_path)
        status_path = os.path.join(result_dir, "grpo_worker_status.json")
        if (record.get("completed") is not True or record.get("outcome") != "a1"
                or record.get("status_sha256") != sha256(status_path)):
            raise IntegrityError(f"{result_dir}: A1 conclusion changed")
        for name in ("046_grpo_training.json", "046_grpo_eval_manifest.json",
                     "046_arm_complete.json"):
            if os.path.exists(os.path.join(result_dir, name)):
                raise IntegrityError(f"{result_dir}: A1 arm gained {name}")
        return "committed", record
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
        return read_json(os.path.join(result_dir, "046_a1_conclusion.json"))
    if state != "pending":
        raise IntegrityError(f"{result_dir}: no exact A1 outcome to conclude")
    status_path = os.path.join(result_dir, "grpo_worker_status.json")
    sft_path = os.path.join(result_dir, "046_sft_training.json")
    if not os.path.exists(sft_path):
        raise IntegrityError(f"{result_dir}: A1 outcome has no SFT record")
    payload = {
        "completed": True,
        "experiment": "046",
        "outcome": "a1",
        "status_sha256": sha256(status_path),
        "sft_record_sha256": sha256(sft_path),
    }
    return _write_once(os.path.join(result_dir, "046_a1_conclusion.json"), payload)


def commit_arm_complete(result_dir):
    sft = training_record(result_dir, "sft")
    grpo = training_record(result_dir, "grpo")
    if eval_state(result_dir, "sft", sft) != "committed":
        raise IntegrityError(f"{result_dir}: SFT endpoint is not committed")
    if eval_state(result_dir, "grpo", grpo) != "committed":
        raise IntegrityError(f"{result_dir}: GRPO endpoint is not committed")
    names = (
        "046_sft_training.json", "046_grpo_training.json",
        "046_sft_eval_manifest.json", "046_grpo_eval_manifest.json",
    )
    hashes = {name: sha256(os.path.join(result_dir, name)) for name in names}
    return _write_once(
        os.path.join(result_dir, "046_arm_complete.json"),
        {"completed": True, "experiment": "046", "sha256": hashes})


def arm_state(result_dir):
    # a complete arm stays tied to all four records written before it
    a1, record = a1_state(result_dir)
    if a1 == "committed":
        return {"state": "a1", "record": record}
    if a1 in ("pending", "interrupted", "failed"):
        return {"state": a1, "status": record}
    path = os.path.join(result_dir, "046_arm_complete.json")
    if not os.path.exists(path):
        return {"state": "absent"}
    record = read_json(path)
    for name, digest in record.get("sha256", {}).items():
        target = os.path.join(result_dir, name)
        if not os.path.exists(target) or sha256(target) != digest:
            raise IntegrityError(f"{result_dir}: {name} changed after completion")
    commit_arm_complete(result_dir)
    return {"state": "committed", "record": record}
