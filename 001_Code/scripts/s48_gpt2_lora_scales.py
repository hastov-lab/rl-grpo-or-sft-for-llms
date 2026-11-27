# GPT-2 LoRA scale records and completion checks
import os
import shutil

from checkpoint import check_artifact_complete, config_hash
from config import load_config
from file_utils import read_json
from scripts.s12_eval import IntegrityError, sha256
from scripts.s28_compose import adapter_hashes_of, verify_binding
from scripts.s42_worker import (_starvation_summary, _step_rows, _write_once,
                                commit_endpoint, endpoint_state,
                                verify_artifact_record)


PACKAGE = "048"
SEEDS_048 = (0, 1, 2)
SCALE_SPECS_048 = {
    "M124": {
        "config": "configs/048_colab_gpt2_m124_seed{seed}.yaml",
        "tag": "048_COLAB_FULL_GPT2_M124_sft_grpo_seed{seed}",
        "base_tag": "042_COLAB_FULL_M124_base",
        "base_package": "042",
        "layers": 12,
    },
    "M355": {
        "config": "configs/048_colab_gpt2_m355_seed{seed}.yaml",
        "tag": "048_COLAB_FULL_GPT2_M355_sft_grpo_seed{seed}",
        "base_tag": "048_COLAB_FULL_GPT2_M355_base",
        "base_package": "048",
        "layers": 24,
    },
    "M774": {
        "config": "configs/048_colab_gpt2_m774_seed{seed}.yaml",
        "tag": "048_COLAB_FULL_GPT2_M774_sft_grpo_seed{seed}",
        "base_tag": "048_COLAB_FULL_GPT2_M774_base",
        "base_package": "048",
        "layers": 36,
    },
}
CONFIGS_048_BY_SCALE = {
    scale: tuple(spec["config"].format(seed=seed) for seed in SEEDS_048)
    for scale, spec in SCALE_SPECS_048.items()
}
CONFIGS_048 = CONFIGS_048_BY_SCALE["M124"]
BASE_TAG = "042_COLAB_FULL_M124_base"
A1_ERROR = (
    "sentinel abort: a1: over 30% of rollouts truncated in the first 50 steps. "
    "raise the completion cap or mask truncated completions"
)
ALLOCATOR_CONFIG = "expandable_segments:True"
ZERO_GRADIENT_ERROR = "matched GRPO first-step gradients are invalid"
ZERO_GRADIENT_RETRY = "048_seed2_gradient_retry.json"


def _scale_and_seed(scale, seed):
    if seed is None:
        scale, seed = "M124", scale
    if scale not in SCALE_SPECS_048:
        raise ValueError(f"experiment 048 does not register scale {scale}")
    if seed not in SEEDS_048:
        raise ValueError(f"experiment 048 does not register seed {seed}")
    return scale, seed


def config_path(scale, seed=None):
    scale, seed = _scale_and_seed(scale, seed)
    return SCALE_SPECS_048[scale]["config"].format(seed=seed)


def arm_tag(scale, seed=None):
    scale, seed = _scale_and_seed(scale, seed)
    return SCALE_SPECS_048[scale]["tag"].format(seed=seed)


def base_tag(scale="M124"):
    if scale not in SCALE_SPECS_048:
        raise ValueError(f"experiment 048 does not register scale {scale}")
    return SCALE_SPECS_048[scale]["base_tag"]


def check_registered_config(cfg):
    # keep the new scale on the completed 046 LoRA recipe
    if (cfg.experiment != PACKAGE or not cfg.exploratory
            or cfg.scale.name not in SCALE_SPECS_048
            or cfg.condition.name != "SFT_GRPO"
            or cfg.train.seed not in SEEDS_048
            or cfg.train.sft_parameter_mode != "lora"
            or cfg.train.grpo_parameter_mode != "lora"
            or cfg.train.sft_precision_mode != "fp32_master_bf16"
            or cfg.train.grpo_precision_mode != "fp32_master_bf16"
            or cfg.train.a2_action != "observe"
            or cfg.train.max_seq_len != 1024
            or cfg.grpo.max_completion_length != 512
            or cfg.eval.max_new_tokens != 512
            or cfg.exec.grpo_batch_size != 8 or cfg.exec.grpo_grad_accum != 1
            or cfg.exec.gradient_checkpointing is not False
            or cfg.exec.allow_oom_fallback is not False):
        raise IntegrityError("experiment 048 config differs from the registered package")
    return cfg


def check_card_vram(total_vram_gib, scale="M124"):
    # the card is registered up front instead of being picked by a pilot like 046,
    # so there is no lock to match here and no pilot run to pay for
    total = float(total_vram_gib)
    if scale not in SCALE_SPECS_048:
        raise IntegrityError("experiment 048 model scale is not registered")
    if scale == "M774":
        if not 75 <= total < 90:
            raise IntegrityError("experiment 048 M774 requires an A100 80GB runtime")
        return "A100_80GB"
    if not 39 <= total < 48:
        raise IntegrityError(f"experiment 048 {scale} requires an A100 40GB runtime")
    return "A100_40GB"


def base_eval_state(base_result_dir, binding, scale="M124"):
    package = SCALE_SPECS_048[scale]["base_package"]
    return endpoint_state(
        base_result_dir, "base", binding, package=package)


def verify_base_reuse(base_record_path, token_audit_path, base_result_dir,
                      scale="M124"):
    # reuse the exact 042 base rather than create another pretrained endpoint
    base = verify_artifact_record(base_record_path)
    artifact = check_artifact_complete(base["artifact"], expected="full")
    audit = read_json(token_audit_path)
    package = SCALE_SPECS_048[scale]["base_package"]
    if (artifact.get("family") != "scratch_gpt"
            or artifact.get("scale") != scale
            or audit.get("completed") is not True
            or audit.get("experiment") != package
            or audit.get("completion_cap") != 512
            or base_eval_state(base_result_dir, base, scale) != "committed"):
        raise IntegrityError(f"experiment 048 cannot reuse the {scale} base")
    return base


def verify_training_base(cfg, base_record_path):
    from train.model_utils import resolve_model_id

    check_registered_config(cfg)
    base = verify_artifact_record(base_record_path)
    artifact = check_artifact_complete(base["artifact"], expected="full")
    if (artifact.get("family") != "scratch_gpt"
            or artifact.get("scale") != cfg.scale.name
            or base.get("source") != resolve_model_id(cfg)):
        raise IntegrityError("experiment 048 training base differs from the config")
    return base


def commit_token_audit(cfg, base_record_path, source_audit_path,
                       source_base_record_path, path):
    from scripts.s46_gpt2_xl import commit_token_audit as commit_source_audit

    return commit_source_audit(
        cfg, base_record_path, source_audit_path, source_base_record_path,
        path, package=PACKAGE, config_check=check_registered_config)


def verify_technical_records(cfg, base_record_path, parity_path, audit_path):
    from scripts.s46_gpt2_xl import verify_technical_records as verify_source_records

    return verify_source_records(
        cfg, base_record_path, parity_path, audit_path, package=PACKAGE,
        config_check=check_registered_config)


def _adapter_record(output_dir, scale):
    artifact = check_artifact_complete(output_dir, expected="adapter")
    if artifact.get("family") != "scratch_gpt" or artifact.get("scale") != scale:
        raise IntegrityError("experiment 048 adapter has the wrong model scale")
    return {"kind": "adapter", "path": os.path.abspath(output_dir),
            "sha256": adapter_hashes_of(output_dir)}


def verify_grpo_adapter_base(cfg, adapter_dir, merged_dir):
    from train.model_utils import resolve_adapter_base

    record_path = os.path.join(adapter_dir, "adapter_base.json")
    resolved = os.path.realpath(resolve_adapter_base(adapter_dir, cfg))
    expected = os.path.realpath(merged_dir)
    if resolved != expected:
        raise IntegrityError("experiment 048 GRPO adapter base differs from the SFT merge")
    return {
        "verified": True,
        "adapter_base_sha256": sha256(record_path),
        "resolved_path": expected,
        "merged_path": expected,
    }


def commit_sft_training(cfg, result_dir, output_dir, base_record_path, wandb_records):
    # bind the adapter to the reused base before GRPO can start
    check_registered_config(cfg)
    if not wandb_records:
        raise IntegrityError("experiment 048 SFT W&B records are missing")
    rows = _step_rows(os.path.join(result_dir, "sft_trainlog.json"), cfg.train.max_steps)
    base = read_json(base_record_path)
    payload = {
        "completed": True,
        "experiment": PACKAGE,
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
        "artifact": _adapter_record(output_dir, cfg.scale.name),
    }
    return _write_once(os.path.join(result_dir, "048_sft_training.json"), payload)


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
    target_counts = precision.get("lora_setup", {}).get("target_module_counts", {})
    expected_layers = SCALE_SPECS_048[cfg.scale.name]["layers"]
    # every registered target must appear once in each transformer block
    if (precision.get("gradient_dtypes") != ["float32"]
            or precision.get("optimizer_state_dtypes") != ["float32"]
            or not 1 <= precision.get("first_nonzero_gradient_update", 0) <= 500
            or precision.get("bf16_enabled") is not True
            or precision.get("accelerator_mixed_precision") != "bf16"
            or precision.get("reference_gate", {}).get("reference_kind")
            != "adapter_disabled"
            or precision.get("reference_behaviour", {}).get("verified") is not True
            or precision.get("adapter_base") != adapter_base
            or precision.get("lora_setup", {}).get("verified") is not True
            or set(target_counts.values()) != {expected_layers}):
        raise IntegrityError("experiment 048 precision report is incomplete")
    verdict = os.path.join(output_dir, "a2_verdict.json")
    if not os.path.exists(verdict):
        raise IntegrityError("experiment 048 observe only A2 record is missing")
    binding = verify_binding(read_json(sft_record_path)["artifact"]["path"], merged_dir)
    payload = {
        "completed": True,
        "experiment": PACKAGE,
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
        "wandb_runs": _wandb_records(result_dir, cfg, package=PACKAGE, stage="grpo"),
        "starvation": _starvation_summary(rows),
        "artifact": _adapter_record(output_dir, cfg.scale.name),
    }
    return _write_once(os.path.join(result_dir, "048_grpo_training.json"), payload)


def _tree_hashes(path):
    return {
        os.path.relpath(os.path.join(root, name), path):
        sha256(os.path.join(root, name))
        for root, _, names in os.walk(path)
        for name in sorted(names)
        if name != "archive_manifest.json"
    }


def _move_retry_item(source, destination):
    if os.path.exists(source) and os.path.exists(destination):
        raise IntegrityError("experiment 048 retry has both active and archived files")
    if os.path.exists(source):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(source, destination)
    elif not os.path.exists(destination):
        raise IntegrityError("experiment 048 retry evidence is missing")


def prepare_zero_gradient_retry(result_dir):
    # keep the completed SFT while the failed GRPO checkpoint moves aside
    tag = arm_tag("M355", 2)
    if os.path.basename(os.path.normpath(result_dir)) != tag:
        raise IntegrityError("experiment 048 gradient retry is only registered for M355 seed 2")
    output_root = os.path.dirname(os.path.dirname(os.path.abspath(result_dir)))
    checkpoint_dir = os.path.join(output_root, "checkpoints", tag)
    archive = os.path.join(
        output_root, "failed_attempts", PACKAGE,
        "M355_seed2_initial_zero_gradient")
    archive_manifest = os.path.join(archive, "archive_manifest.json")
    retry_path = os.path.join(result_dir, ZERO_GRADIENT_RETRY)
    active_status = os.path.join(result_dir, "grpo_worker_status.json")

    if os.path.exists(retry_path):
        retry = read_json(retry_path)
        manifest = read_json(archive_manifest)
        if (retry.get("completed") is not True
                or retry.get("archive_manifest_sha256") != sha256(archive_manifest)
                or manifest.get("files") != _tree_hashes(archive)):
            raise IntegrityError("experiment 048 gradient retry archive changed")
        if os.path.exists(active_status):
            status = read_json(active_status)
            if status.get("completed") is not True:
                raise IntegrityError("experiment 048 seed 2 gradient retry already failed")
        return retry

    status = read_json(active_status)
    forbidden = (
        "048_grpo_training.json", "048_sft_eval_manifest.json",
        "048_grpo_eval_manifest.json", "048_arm_complete.json",
        "048_a1_conclusion.json", "precision_report.json",
    )
    if (status.get("completed") is not False
            or status.get("error_type") != "IntegrityError"
            or status.get("error_message") != ZERO_GRADIENT_ERROR
            or status.get("cuda_oom") is not False
            or status.get("start_step") != 0
            or status.get("end_step") != 500
            or status.get("completed_updates") != 500
            or any(os.path.exists(os.path.join(result_dir, name))
                   for name in forbidden)):
        raise IntegrityError("experiment 048 failure does not match the registered retry")
    sft_record = training_record(result_dir, "sft")
    destinations = (
        (os.path.join(checkpoint_dir, "grpo"),
         os.path.join(archive, "checkpoints", "grpo")),
        (active_status,
         os.path.join(archive, "results", "grpo_worker_status.json")),
        (os.path.join(result_dir, "048_grpo_wandb_runs"),
         os.path.join(archive, "results", "048_grpo_wandb_runs")),
    )
    for source, destination in destinations:
        _move_retry_item(source, destination)
    manifest = {
        "completed": True,
        "experiment": PACKAGE,
        "scale": "M355",
        "seed": 2,
        "reason": "initial_zero_gradient_gate",
        "files": _tree_hashes(archive),
    }
    _write_once(archive_manifest, manifest)
    return _write_once(retry_path, {
        "completed": True,
        "experiment": PACKAGE,
        "scale": "M355",
        "seed": 2,
        "clean_grpo_restart": True,
        "reuse_sft": True,
        "sft_record_sha256": sha256(
            os.path.join(result_dir, "048_sft_training.json")),
        "sft_artifact_sha256": sft_record["artifact"]["sha256"],
        "archive": os.path.abspath(archive),
        "archive_manifest_sha256": sha256(archive_manifest),
    })


def training_record(result_dir, branch):
    # recheck artifact bytes whenever a completed stage is reused
    path = os.path.join(result_dir, f"048_{branch}_training.json")
    record = read_json(path)
    artifact = record.get("artifact", {})
    if (record.get("completed") is not True or record.get("experiment") != PACKAGE
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
        adapter_base_path = os.path.join(artifact.get("path", ""), "adapter_base.json")
        if (base.get("verified") is not True
                or base.get("merged_path") != os.path.realpath(
                    record.get("merged_path", ""))
                or base.get("resolved_path") != base.get("merged_path")
                or sha256(adapter_base_path) != base.get("adapter_base_sha256")):
            raise IntegrityError(f"{result_dir}: GRPO adapter base changed")
    return record


def eval_state(result_dir, branch, binding):
    return endpoint_state(result_dir, branch, binding, package=PACKAGE)


def commit_eval(result_dir, scratch_dir, branch, binding):
    return commit_endpoint(result_dir, scratch_dir, branch, binding, package=PACKAGE)


def a1_state(result_dir):
    record_path = os.path.join(result_dir, "048_a1_conclusion.json")
    if os.path.exists(record_path):
        record = read_json(record_path)
        status_path = os.path.join(result_dir, "grpo_worker_status.json")
        if (record.get("completed") is not True or record.get("outcome") != "a1"
                or record.get("status_sha256") != sha256(status_path)):
            raise IntegrityError(f"{result_dir}: A1 conclusion changed")
        for name in ("048_grpo_training.json", "048_grpo_eval_manifest.json",
                     "048_arm_complete.json"):
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
        return read_json(os.path.join(result_dir, "048_a1_conclusion.json"))
    if state != "pending":
        raise IntegrityError(f"{result_dir}: no exact A1 outcome to conclude")
    status_path = os.path.join(result_dir, "grpo_worker_status.json")
    sft_path = os.path.join(result_dir, "048_sft_training.json")
    if not os.path.exists(sft_path):
        raise IntegrityError(f"{result_dir}: A1 outcome has no SFT record")
    payload = {
        "completed": True,
        "experiment": PACKAGE,
        "outcome": "a1",
        "status_sha256": sha256(status_path),
        "sft_record_sha256": sha256(sft_path),
    }
    return _write_once(os.path.join(result_dir, "048_a1_conclusion.json"), payload)


def commit_arm_complete(result_dir):
    sft = training_record(result_dir, "sft")
    grpo = training_record(result_dir, "grpo")
    if eval_state(result_dir, "sft", sft) != "committed":
        raise IntegrityError(f"{result_dir}: SFT endpoint is not committed")
    if eval_state(result_dir, "grpo", grpo) != "committed":
        raise IntegrityError(f"{result_dir}: GRPO endpoint is not committed")
    names = (
        "048_sft_training.json", "048_grpo_training.json",
        "048_sft_eval_manifest.json", "048_grpo_eval_manifest.json",
    )
    hashes = {name: sha256(os.path.join(result_dir, name)) for name in names}
    return _write_once(
        os.path.join(result_dir, "048_arm_complete.json"),
        {"completed": True, "experiment": PACKAGE, "sha256": hashes})


def arm_state(result_dir):
    retry_path = os.path.join(result_dir, ZERO_GRADIENT_RETRY)
    if os.path.exists(retry_path):
        prepare_zero_gradient_retry(result_dir)
    a1, record = a1_state(result_dir)
    if a1 == "committed":
        return {"state": "a1", "record": record}
    if a1 == "failed" and os.path.basename(os.path.normpath(result_dir)) == arm_tag(
            "M355", 2):
        prepare_zero_gradient_retry(result_dir)
        return {"state": "absent"}
    if a1 in ("pending", "interrupted", "failed"):
        return {"state": a1, "status": record}
    path = os.path.join(result_dir, "048_arm_complete.json")
    if not os.path.exists(path):
        return {"state": "absent"}
    record = read_json(path)
    for name, digest in record.get("sha256", {}).items():
        target = os.path.join(result_dir, name)
        if not os.path.exists(target) or sha256(target) != digest:
            raise IntegrityError(f"{result_dir}: {name} changed after completion")
    commit_arm_complete(result_dir)
    return {"state": "committed", "record": record}


def registered_configs(scale="M124"):
    return {
        seed: check_registered_config(load_config(config_path(scale, seed)))
        for seed in SEEDS_048
    }
