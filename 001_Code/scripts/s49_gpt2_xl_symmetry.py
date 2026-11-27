# GPT2 XL final symmetry package records
import copy
import os
from dataclasses import asdict

from checkpoint import check_artifact_complete, config_hash
from config import load_config, run_tag
from enums import Condition
from file_utils import read_json
from scripts.s12_eval import IntegrityError, sha256
from scripts.s28_compose import adapter_hashes_of, verify_binding
from scripts.s42_worker import (commit_endpoint, endpoint_state, starvation_summary,
                                step_rows, verify_artifact_record, wandb_records,
                                write_once)


PACKAGE = "049"
SEEDS_049 = (0, 1, 2)
MINIMUM_LAUNCH_UNITS = 350.0
BASE_TAG = "046_COLAB_FULL_GPT2_B15_base"
TECHNICAL_TAG = "046_technical"
A1_ERROR = (
    "sentinel abort: a1: over 30% of rollouts truncated in the first 50 steps. "
    "raise the completion cap or mask truncated completions"
)
FAMILY_SPECS_049 = {
    "p2": {"experiment": "049_p2", "steps": 31, "kind": "sft"},
    "p8": {"experiment": "049_p8", "steps": 123, "kind": "sft"},
    "p32": {"experiment": "049_p32", "steps": 490, "kind": "sft"},
    "wsft_p8": {"experiment": "049_wsft_p8", "steps": 123,
                  "kind": "sft", "warm": True},
    "base_grpo": {"experiment": "049_base_grpo", "steps": 500,
                    "kind": "grpo"},
}
EXPERIMENT_TO_FAMILY = {
    spec["experiment"]: family for family, spec in FAMILY_SPECS_049.items()
}
ANALYSIS_FAMILIES_049 = {
    "F1": ("p8",),
    "F2a": ("p2", "p8"),
    "F2b": ("p32", "p8"),
    "F3": ("wsft_p8", "p8"),
    "F4": ("base_grpo",),
}


def config_path(family, seed):
    if family not in FAMILY_SPECS_049:
        raise ValueError(f"experiment 049 does not register family {family}")
    if seed not in SEEDS_049:
        raise ValueError(f"experiment 049 does not register seed {seed}")
    return f"configs/049_{family}_colab_gpt2_b15_seed{seed}.yaml"


def family_of(cfg):
    family = EXPERIMENT_TO_FAMILY.get(cfg.experiment)
    if family is None:
        raise IntegrityError("experiment 049 config has an unregistered family")
    return family


def _expected_config(family, seed):
    # derive every arm from the completed seed matched XL pipeline
    cfg = copy.deepcopy(load_config(
        f"configs/046_colab_gpt2_b15_seed{seed}.yaml"))
    spec = FAMILY_SPECS_049[family]
    cfg.experiment = spec["experiment"]
    if spec["kind"] == "sft":
        cfg.condition = Condition.SFT_ONLY
        cfg.train.max_steps = spec["steps"]
        cfg.train.sft_data = "mmlu_mc"
        cfg.train.sft_val_examples = 0
        cfg.train.a2_action = "abort"
        if spec.get("warm"):
            cfg.train.sft_from_adapter = (
                "checkpoints/046_COLAB_FULL_GPT2_B15_sft_grpo_"
                f"seed{seed}/sft")
    else:
        cfg.condition = Condition.BASE_GRPO
    return cfg


def check_registered_config(cfg):
    family = family_of(cfg)
    if cfg.train.seed not in SEEDS_049:
        raise IntegrityError("experiment 049 config has an unregistered seed")
    expected = _expected_config(family, cfg.train.seed)
    if asdict(cfg) != asdict(expected):
        raise IntegrityError("experiment 049 config differs from the fixed package")
    return cfg


def registered_configs():
    return {
        (family, seed): check_registered_config(
            load_config(config_path(family, seed)))
        for family in FAMILY_SPECS_049 for seed in SEEDS_049
    }


def arm_tag(family, seed):
    return run_tag(check_registered_config(load_config(config_path(family, seed))))


def check_card_vram(total_vram_gib, family):
    total = float(total_vram_gib)
    if family == "base_grpo":
        if not 75 <= total < 90:
            raise IntegrityError("experiment 049 base GRPO requires an A100 80GB runtime")
        return "A100_80GB"
    if family not in FAMILY_SPECS_049:
        raise IntegrityError("experiment 049 family is not registered")
    if not 39 <= total < 48:
        raise IntegrityError("experiment 049 MC-SFT requires an A100 40GB runtime")
    return "A100_40GB"


def _base_start(base_record_path, base_result_dir):
    from scripts.s46_gpt2_xl import eval_state as source_eval_state

    base = verify_artifact_record(base_record_path)
    artifact = check_artifact_complete(base["artifact"], expected="full")
    if (base.get("completed") is not True or base.get("experiment") != "046"
            or base.get("role") != "immutable_base"
            or base.get("source") != "gpt2-xl"
            or artifact.get("family") != "scratch_gpt"
            or artifact.get("scale") != "GPT2_B15"
            or source_eval_state(base_result_dir, "base", base) != "committed"):
        raise IntegrityError("experiment 049 immutable base is not reusable")
    endpoint_path = os.path.join(base_result_dir, "046_base_eval_manifest.json")
    return {
        "kind": "immutable_base",
        "path": os.path.realpath(base["artifact"]),
        "base_record_path": os.path.abspath(base_record_path),
        "base_record_sha256": sha256(base_record_path),
        "base_artifact_sha256": base["artifact_sha256"],
        "base_endpoint_path": os.path.abspath(endpoint_path),
        "base_endpoint_sha256": sha256(endpoint_path),
    }


def verify_warm_start(cfg, parent_result_dir):
    from scripts.s46_gpt2_xl import arm_state as source_arm_state
    from scripts.s46_gpt2_xl import training_record as source_training_record

    check_registered_config(cfg)
    if family_of(cfg) != "wsft_p8":
        raise IntegrityError("experiment 049 warm start used by another family")
    if source_arm_state(parent_result_dir).get("state") != "committed":
        raise IntegrityError("experiment 049 warm parent is not complete")
    sft_path = os.path.join(parent_result_dir, "046_sft_training.json")
    grpo_path = os.path.join(parent_result_dir, "046_grpo_training.json")
    sft = source_training_record(parent_result_dir, "sft")
    grpo = source_training_record(parent_result_dir, "grpo")
    configured_adapter = os.path.realpath(os.path.join(
        cfg.output_root, cfg.train.sft_from_adapter))
    merged = os.path.realpath(os.path.join(
        os.path.dirname(configured_adapter), "sft_merged"))
    # use the exact merge that trained the earlier GRPO adapter
    binding = verify_binding(configured_adapter, merged)
    check_artifact_complete(merged, expected="full")
    if (sft.get("seed") != cfg.train.seed or grpo.get("seed") != cfg.train.seed
            or os.path.realpath(sft["artifact"]["path"]) != configured_adapter
            or grpo.get("sft_record_sha256") != sha256(sft_path)
            or os.path.realpath(grpo.get("merged_path", "")) != merged
            or grpo.get("merge_binding") != binding):
        raise IntegrityError("experiment 049 warm start differs from its 046 parent")
    return {
        "kind": "046_sft_merge",
        "seed": cfg.train.seed,
        "path": merged,
        "parent_result_dir": os.path.abspath(parent_result_dir),
        "sft_record_path": os.path.abspath(sft_path),
        "sft_record_sha256": sha256(sft_path),
        "grpo_record_path": os.path.abspath(grpo_path),
        "grpo_record_sha256": sha256(grpo_path),
        "merge_binding": binding,
    }


def _preflight_payload(base_record_path, base_result_dir, parity_path,
                       token_audit_path, parent_result_dirs):
    from scripts.s46_gpt2_xl import verify_technical_records

    configs = registered_configs()
    source_cfg = load_config("configs/046_colab_gpt2_b15_seed0.yaml")
    verify_technical_records(
        source_cfg, base_record_path, parity_path, token_audit_path)
    base = _base_start(base_record_path, base_result_dir)
    # commit the full source roster before any new test contact
    starts = {}
    for (family, seed), cfg in configs.items():
        starts[arm_tag(family, seed)] = (
            verify_warm_start(cfg, parent_result_dirs[seed])
            if family == "wsft_p8" else base)
    return {
        "completed": True,
        "experiment": PACKAGE,
        "roster": [arm_tag(family, seed)
                   for family in FAMILY_SPECS_049 for seed in SEEDS_049],
        "config_sha256": {
            config_path(family, seed): config_hash(cfg)
            for (family, seed), cfg in configs.items()
        },
        "technical": {
            "parity_path": os.path.abspath(parity_path),
            "parity_sha256": sha256(parity_path),
            "token_audit_path": os.path.abspath(token_audit_path),
            "token_audit_sha256": sha256(token_audit_path),
        },
        "starts": starts,
        "execution": {
            "mc_sft_card": "A100_40GB",
            "base_grpo_card": "A100_80GB",
            "gradient_checkpointing": False,
            "allow_oom_fallback": False,
        },
    }


def commit_preflight(path, base_record_path, base_result_dir, parity_path,
                     token_audit_path, parent_result_dirs):
    if set(parent_result_dirs) != set(SEEDS_049):
        raise IntegrityError("experiment 049 warm parent roster is incomplete")
    payload = _preflight_payload(
        base_record_path, base_result_dir, parity_path,
        token_audit_path, parent_result_dirs)
    return write_once(path, payload)


def verify_preflight(path):
    record = read_json(path)
    expected_roster = [arm_tag(family, seed)
                       for family in FAMILY_SPECS_049 for seed in SEEDS_049]
    if (record.get("completed") is not True or record.get("experiment") != PACKAGE
            or record.get("roster") != expected_roster
            or record.get("execution") != {
                "mc_sft_card": "A100_40GB",
                "base_grpo_card": "A100_80GB",
                "gradient_checkpointing": False,
                "allow_oom_fallback": False,
            }):
        raise IntegrityError("experiment 049 preflight record changed")
    configs = registered_configs()
    expected_hashes = {
        config_path(family, seed): config_hash(cfg)
        for (family, seed), cfg in configs.items()
    }
    if record.get("config_sha256") != expected_hashes:
        raise IntegrityError("experiment 049 preflight config hashes changed")
    technical = record.get("technical", {})
    if any(sha256(technical.get(f"{key}_path", "")) != technical.get(
            f"{key}_sha256") for key in ("parity", "token_audit")):
        raise IntegrityError("experiment 049 technical record changed")
    starts = record.get("starts", {})
    if set(starts) != set(expected_roster):
        raise IntegrityError("experiment 049 preflight start roster changed")
    base_start = starts[arm_tag("p2", 0)]
    if base_start.get("kind") != "immutable_base":
        raise IntegrityError("experiment 049 base start kind changed")
    from scripts.s46_gpt2_xl import verify_technical_records

    source_cfg = load_config("configs/046_colab_gpt2_b15_seed0.yaml")
    verify_technical_records(
        source_cfg, base_start.get("base_record_path", ""),
        technical.get("parity_path", ""), technical.get("token_audit_path", ""))
    expected_base = _base_start(
        base_start["base_record_path"],
        os.path.dirname(base_start["base_endpoint_path"]))
    # rederive every start so seed swaps and path edits cannot pass by hash alone
    for (family, seed), cfg in configs.items():
        tag = arm_tag(family, seed)
        start = starts[tag]
        expected = (verify_warm_start(cfg, start.get("parent_result_dir", ""))
                    if family == "wsft_p8" else expected_base)
        if start != expected:
            raise IntegrityError(f"experiment 049 start changed: {tag}")
    return record


def commit_launch(path, preflight_path, available_units):
    preflight = verify_preflight(preflight_path)
    if os.path.exists(path):
        return verify_launch(path, preflight_path)
    # check the funding threshold once before the fixed roster starts
    units = float(available_units)
    if units < MINIMUM_LAUNCH_UNITS:
        raise IntegrityError("experiment 049 needs at least 350 units before launch")
    return write_once(path, {
        "completed": True,
        "experiment": PACKAGE,
        "initial_available_units": units,
        "minimum_units": MINIMUM_LAUNCH_UNITS,
        "preflight_path": os.path.abspath(preflight_path),
        "preflight_sha256": sha256(preflight_path),
        "roster": preflight["roster"],
    })


def verify_launch(path, preflight_path):
    record = read_json(path)
    preflight = verify_preflight(preflight_path)
    if (record.get("completed") is not True or record.get("experiment") != PACKAGE
            or record.get("minimum_units") != MINIMUM_LAUNCH_UNITS
            or record.get("initial_available_units", 0) < MINIMUM_LAUNCH_UNITS
            or record.get("preflight_path") != os.path.abspath(preflight_path)
            or record.get("preflight_sha256") != sha256(preflight_path)
            or record.get("roster") != preflight.get("roster")):
        raise IntegrityError("experiment 049 launch record changed")
    return record


def start_for(cfg, preflight_path):
    check_registered_config(cfg)
    preflight = verify_preflight(preflight_path)
    return preflight["starts"][run_tag(cfg)]


def _adapter_record(output_dir):
    artifact = check_artifact_complete(output_dir, expected="adapter")
    if artifact.get("family") != "scratch_gpt" or artifact.get("scale") != "GPT2_B15":
        raise IntegrityError("experiment 049 adapter has the wrong model scale")
    return {"kind": "adapter", "path": os.path.abspath(output_dir),
            "sha256": adapter_hashes_of(output_dir)}


def verify_adapter_base(cfg, adapter_dir, start):
    from train.model_utils import resolve_adapter_base

    record_path = os.path.join(adapter_dir, "adapter_base.json")
    resolved = os.path.realpath(resolve_adapter_base(adapter_dir, cfg))
    expected = os.path.realpath(start["path"])
    if resolved != expected:
        raise IntegrityError("experiment 049 adapter base differs from the bound start")
    return {"verified": True, "resolved_path": resolved,
            "adapter_base_sha256": sha256(record_path)}


def commit_sft_training(cfg, result_dir, output_dir, preflight_path):
    check_registered_config(cfg)
    if FAMILY_SPECS_049[family_of(cfg)]["kind"] != "sft":
        raise IntegrityError("experiment 049 SFT record used by the GRPO family")
    trainlog_path = os.path.join(result_dir, "sft_trainlog.json")
    rows = step_rows(trainlog_path, cfg.train.max_steps)
    start = start_for(cfg, preflight_path)
    payload = {
        "completed": True,
        "experiment": PACKAGE,
        "family": family_of(cfg),
        "stage": "sft",
        "seed": cfg.train.seed,
        "config_hash": config_hash(cfg),
        "updates": len(rows),
        "parameter_mode": "lora",
        "storage_dtype": "float32",
        "compute_dtype": "bfloat16",
        "preflight_path": os.path.abspath(preflight_path),
        "preflight_sha256": sha256(preflight_path),
        "trainlog_path": os.path.abspath(trainlog_path),
        "trainlog_sha256": sha256(trainlog_path),
        "start": start,
        "adapter_base": verify_adapter_base(cfg, output_dir, start),
        "wandb_runs": wandb_records(
            result_dir, cfg, package=PACKAGE, stage="sft"),
        "artifact": _adapter_record(output_dir),
    }
    return write_once(os.path.join(result_dir, "049_sft_training.json"), payload)


def commit_grpo_training(cfg, result_dir, output_dir, precision_path,
                         preflight_path):
    check_registered_config(cfg)
    if family_of(cfg) != "base_grpo":
        raise IntegrityError("experiment 049 GRPO record used by an SFT family")
    fields = ("rewards/accuracy_reward/mean", "rewards/format_reward/mean",
              "frac_reward_zero_std", "kl")
    trainlog_path = os.path.join(result_dir, "grpo_trainlog.json")
    rows = step_rows(trainlog_path, cfg.train.max_steps, fields)
    start = start_for(cfg, preflight_path)
    precision = read_json(precision_path)
    adapter_base = verify_adapter_base(cfg, output_dir, start)
    targets = precision.get("lora_setup", {}).get("target_module_counts", {})
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
            or set(targets.values()) != {48}):
        raise IntegrityError("experiment 049 precision report is incomplete")
    verdict_path = os.path.join(output_dir, "a2_verdict.json")
    if not os.path.exists(verdict_path):
        raise IntegrityError("experiment 049 observe only A2 record is missing")
    _verify_a2_record(verdict_path, cfg, rows)
    payload = {
        "completed": True,
        "experiment": PACKAGE,
        "family": "base_grpo",
        "stage": "grpo",
        "seed": cfg.train.seed,
        "config_hash": config_hash(cfg),
        "updates": len(rows),
        "parameter_mode": "lora",
        "preflight_path": os.path.abspath(preflight_path),
        "preflight_sha256": sha256(preflight_path),
        "trainlog_path": os.path.abspath(trainlog_path),
        "trainlog_sha256": sha256(trainlog_path),
        "start": start,
        "adapter_base": adapter_base,
        "precision_report_path": os.path.abspath(precision_path),
        "precision_report_sha256": sha256(precision_path),
        "a2_verdict_path": os.path.abspath(verdict_path),
        "a2_verdict_sha256": sha256(verdict_path),
        "wandb_runs": wandb_records(
            result_dir, cfg, package=PACKAGE, stage="grpo"),
        "starvation": starvation_summary(rows),
        "artifact": _adapter_record(output_dir),
    }
    return write_once(os.path.join(result_dir, "049_grpo_training.json"), payload)


def _verify_a2_record(path, cfg, rows):
    from obs import verify_a2_verdict

    prefix = [row for row in rows if 1 <= int(row.get("step", 0)) <= 150]
    try:
        return verify_a2_verdict(path, cfg, prefix)
    except RuntimeError as exc:
        raise IntegrityError("experiment 049 A2 verdict is invalid") from exc


def training_record(cfg, result_dir):
    # recheck every source record before an endpoint can use the adapter
    check_registered_config(cfg)
    branch = FAMILY_SPECS_049[family_of(cfg)]["kind"]
    path = os.path.join(result_dir, f"049_{branch}_training.json")
    record = read_json(path)
    artifact = record.get("artifact", {})
    trainlog_path = os.path.abspath(os.path.join(
        result_dir, f"{branch}_trainlog.json"))
    if (record.get("completed") is not True or record.get("experiment") != PACKAGE
            or record.get("family") != family_of(cfg)
            or record.get("stage") != branch or record.get("seed") != cfg.train.seed
            or record.get("config_hash") != config_hash(cfg)
            or record.get("updates") != cfg.train.max_steps
            or record.get("preflight_sha256")
            != sha256(record.get("preflight_path", ""))
            or record.get("trainlog_path") != trainlog_path
            or not os.path.exists(trainlog_path)
            or record.get("trainlog_sha256") != sha256(trainlog_path)
            or record.get("artifact") != _adapter_record(
                artifact.get("path", ""))):
        raise IntegrityError(f"{result_dir}: training record changed")
    fields = (() if branch == "sft" else (
        "rewards/accuracy_reward/mean", "rewards/format_reward/mean",
        "frac_reward_zero_std", "kl"))
    rows = step_rows(trainlog_path, cfg.train.max_steps, fields)
    start = start_for(cfg, record["preflight_path"])
    if record.get("start") != start or record.get("adapter_base") != verify_adapter_base(
            cfg, artifact["path"], start):
        raise IntegrityError(f"{result_dir}: training start changed")
    current_wandb = wandb_records(
        result_dir, cfg, package=PACKAGE, stage=branch)
    if not current_wandb or record.get("wandb_runs") != current_wandb:
        raise IntegrityError(f"{result_dir}: W&B records are missing")
    if branch == "grpo":
        verdict_path = os.path.join(artifact["path"], "a2_verdict.json")
        if (sha256(record.get("precision_report_path", ""))
                != record.get("precision_report_sha256")
                or record.get("a2_verdict_path") != os.path.abspath(verdict_path)
                or sha256(verdict_path) != record.get("a2_verdict_sha256")
                or record.get("starvation") != starvation_summary(rows)):
            raise IntegrityError(f"{result_dir}: GRPO evidence changed")
        _verify_a2_record(verdict_path, cfg, rows)
    return record


def _a1_status_matches(cfg, status):
    return bool(
        status.get("completed") is False
        and status.get("cuda_oom") is False
        and status.get("task") == "grpo"
        and status.get("seed") == cfg.train.seed
        and status.get("family") == "base_grpo"
        and status.get("error_type") == "RuntimeError"
        and status.get("error_message") == A1_ERROR)


def eval_state(cfg, result_dir, binding):
    branch = FAMILY_SPECS_049[family_of(cfg)]["kind"]
    return endpoint_state(result_dir, branch, binding, package=PACKAGE)


def commit_eval(cfg, result_dir, scratch_dir, binding):
    branch = FAMILY_SPECS_049[family_of(cfg)]["kind"]
    return commit_endpoint(
        result_dir, scratch_dir, branch, binding, package=PACKAGE)


def a1_state(cfg, result_dir, preflight_path):
    check_registered_config(cfg)
    if family_of(cfg) != "base_grpo":
        return "absent", None
    record_path = os.path.join(result_dir, "049_a1_conclusion.json")
    status_path = os.path.join(result_dir, "grpo_worker_status.json")
    if os.path.exists(record_path):
        record = read_json(record_path)
        status = read_json(status_path)
        wandb = wandb_records(result_dir, cfg, package=PACKAGE, stage="grpo")
        if (record.get("completed") is not True
                or record.get("experiment") != PACKAGE
                or record.get("family") != "base_grpo"
                or record.get("outcome") != "a1"
                or record.get("seed") != cfg.train.seed
                or record.get("config_hash") != config_hash(cfg)
                or record.get("status_sha256") != sha256(status_path)
                or record.get("preflight_path") != os.path.abspath(preflight_path)
                or record.get("preflight_sha256") != sha256(preflight_path)
                or record.get("start") != start_for(cfg, preflight_path)
                or not wandb or record.get("wandb_runs") != wandb
                or not _a1_status_matches(cfg, status)):
            raise IntegrityError(f"{result_dir}: A1 conclusion changed")
        for name in ("049_grpo_training.json", "049_grpo_eval_manifest.json",
                     "049_arm_complete.json"):
            if os.path.exists(os.path.join(result_dir, name)):
                raise IntegrityError(f"{result_dir}: A1 arm gained {name}")
        return "committed", record
    if not os.path.exists(status_path):
        return "absent", None
    status = read_json(status_path)
    if _a1_status_matches(cfg, status):
        return "pending", status
    if status.get("error_type") == "KeyboardInterrupt":
        return "interrupted", status
    if status.get("completed") is True:
        return "absent", status
    return "failed", status


def commit_a1_conclusion(cfg, result_dir, preflight_path):
    state, _ = a1_state(cfg, result_dir, preflight_path)
    if state == "committed":
        return read_json(os.path.join(result_dir, "049_a1_conclusion.json"))
    if state != "pending":
        raise IntegrityError(f"{result_dir}: no exact A1 outcome to conclude")
    status_path = os.path.join(result_dir, "grpo_worker_status.json")
    # only the exact truncation stop may replace a base GRPO endpoint
    wandb = wandb_records(result_dir, cfg, package=PACKAGE, stage="grpo")
    if not wandb:
        raise IntegrityError(f"{result_dir}: A1 W&B record is missing")
    payload = {
        "completed": True,
        "experiment": PACKAGE,
        "family": "base_grpo",
        "outcome": "a1",
        "seed": cfg.train.seed,
        "config_hash": config_hash(cfg),
        "status_sha256": sha256(status_path),
        "preflight_path": os.path.abspath(preflight_path),
        "preflight_sha256": sha256(preflight_path),
        "start": start_for(cfg, preflight_path),
        "wandb_runs": wandb,
    }
    record = write_once(
        os.path.join(result_dir, "049_a1_conclusion.json"), payload)
    return a1_state(cfg, result_dir, preflight_path)[1]


def commit_arm_complete(cfg, result_dir, preflight_path):
    branch = FAMILY_SPECS_049[family_of(cfg)]["kind"]
    training = training_record(cfg, result_dir)
    if eval_state(cfg, result_dir, training) != "committed":
        raise IntegrityError(f"{result_dir}: endpoint is not committed")
    names = (f"049_{branch}_training.json",
             f"049_{branch}_eval_manifest.json")
    # write arm record only after training and both endpoint files verify
    hashes = {name: sha256(os.path.join(result_dir, name)) for name in names}
    return write_once(os.path.join(result_dir, "049_arm_complete.json"), {
        "completed": True,
        "experiment": PACKAGE,
        "family": family_of(cfg),
        "seed": cfg.train.seed,
        "preflight_sha256": sha256(preflight_path),
        "sha256": hashes,
    })


def arm_state(cfg, result_dir, preflight_path):
    # distinguish unfinished work from a completed arm that changed later
    check_registered_config(cfg)
    a1, record = a1_state(cfg, result_dir, preflight_path)
    if a1 == "committed":
        return {"state": "a1", "record": record}
    if a1 in ("pending", "interrupted", "failed"):
        return {"state": a1, "status": record}
    path = os.path.join(result_dir, "049_arm_complete.json")
    if not os.path.exists(path):
        return {"state": "absent"}
    record = read_json(path)
    if (record.get("completed") is not True or record.get("experiment") != PACKAGE
            or record.get("family") != family_of(cfg)
            or record.get("seed") != cfg.train.seed
            or record.get("preflight_sha256") != sha256(preflight_path)):
        raise IntegrityError(f"{result_dir}: arm completion changed")
    for name, digest in record.get("sha256", {}).items():
        target = os.path.join(result_dir, name)
        if not os.path.exists(target) or sha256(target) != digest:
            raise IntegrityError(f"{result_dir}: {name} changed after completion")
    commit_arm_complete(cfg, result_dir, preflight_path)
    return {"state": "committed", "record": record}


def analysis_inputs(family, results_root, preflight_path):
    if family not in ANALYSIS_FAMILIES_049:
        raise ValueError(f"experiment 049 does not register analysis family {family}")
    preflight = verify_preflight(preflight_path)
    # collect every fixed seed before a formal family correction is allowed
    pairs = []
    incomplete = []
    for seed in SEEDS_049:
        endpoints = {}
        for arm_family in ANALYSIS_FAMILIES_049[family]:
            cfg = load_config(config_path(arm_family, seed))
            result_dir = os.path.join(results_root, arm_tag(arm_family, seed))
            state = arm_state(cfg, result_dir, preflight_path)["state"]
            if state != "committed":
                incomplete.append({"seed": seed, "family": arm_family,
                                   "state": state})
                continue
            branch = FAMILY_SPECS_049[arm_family]["kind"]
            endpoints[arm_family] = {
                "result_dir": os.path.abspath(result_dir),
                "manifest": os.path.abspath(os.path.join(
                    result_dir, f"049_{branch}_eval_manifest.json")),
            }
        pairs.append({"seed": seed, "endpoints": endpoints})

    reference = None
    if family == "F1":
        reference = preflight["starts"][arm_tag("p8", 0)]
    elif family == "F4":
        from scripts.s46_gpt2_xl import arm_state as source_arm_state
        from scripts.s46_gpt2_xl import eval_state as source_eval_state
        from scripts.s46_gpt2_xl import training_record as source_training_record

        for pair in pairs:
            seed = pair["seed"]
            source_cfg = load_config(
                f"configs/046_colab_gpt2_b15_seed{seed}.yaml")
            source_dir = os.path.join(results_root, run_tag(source_cfg))
            state = source_arm_state(source_dir).get("state")
            if state != "committed":
                incomplete.append({"seed": seed, "family": "046_grpo",
                                   "state": state})
                continue
            binding = source_training_record(source_dir, "grpo")
            if source_eval_state(source_dir, "grpo", binding) != "committed":
                raise IntegrityError(
                    f"{source_dir}: experiment 046 GRPO endpoint changed")
            pair["historical_046_grpo"] = {
                "result_dir": os.path.abspath(source_dir),
                "manifest": os.path.abspath(os.path.join(
                    source_dir, "046_grpo_eval_manifest.json")),
            }

    # formal correction and a common reading both require all three fixed pairs
    if incomplete:
        raise IntegrityError(
            f"experiment 049 {family} needs three completed pairs: {incomplete}")
    return {
        "completed": True,
        "experiment": PACKAGE,
        "family": family,
        "pair_count": len(pairs),
        "pairs": pairs,
        "shared_reference": reference,
        "formal_analysis_allowed": len(pairs) == len(SEEDS_049),
    }
