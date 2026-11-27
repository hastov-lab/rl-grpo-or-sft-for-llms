# run experiment 049 tasks in disposable processes
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_utils import read_json
from scripts.grpo_worker import classify_failure, save_status
from scripts.s12_eval import IntegrityError, atomic_write_json
from scripts.s42_worker import capture_wandb_run, completion_length_gate
from scripts.s46_worker import (check_lora_setup, latest_checkpoint_step,
                                reference_behaviour_report, run_eval)


def _wandb_callback(result_dir, cfg, stage, target):
    from transformers import TrainerCallback

    class wandb_record(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step > 0 and not target:
                target.append(capture_wandb_run(
                    result_dir, cfg, state.global_step,
                    package="049", stage=stage))
            return control

    return wandb_record()


def _run_sft(cfg, out_dir, result_dir, preflight_path):
    from data.splits import build_grpo_split
    from scripts.s49_gpt2_xl_symmetry import (commit_sft_training,
                                              start_for)
    from train.sft import mc_sft_rows, run_sft_rows

    rows = mc_sft_rows(build_grpo_split())
    if len(rows) != 245:
        raise IntegrityError("experiment 049 MC-SFT row count changed")
    start = start_for(cfg, preflight_path)
    setup = {}
    recorded = []

    def trainer_setup(trainer):
        setup.update(check_lora_setup(trainer, "sft", package="049"))
        trainer.add_callback(_wandb_callback(
            result_dir, cfg, "sft", recorded))

    # train the answer-key rows from the preflight-bound start
    run_sft_rows(
        cfg, "cuda", rows, start["path"],
        "local", os.path.relpath(start["path"], cfg.output_root),
        out_dir=out_dir, report_to=["wandb"], log_dir=result_dir,
        run_name=f"{os.path.basename(result_dir)}_sft",
        allow_oom_fallback=False, trainer_setup=trainer_setup)
    if (setup.get("verified") is not True
            or set(setup.get("target_module_counts", {}).values()) != {48}):
        raise IntegrityError("experiment 049 SFT LoRA setup is incomplete")
    return commit_sft_training(
        cfg, result_dir, out_dir, preflight_path)


def _run_grpo(cfg, out_dir, result_dir, precision_path, preflight_path):
    from data.splits import build_grpo_split
    from scripts.s35_matched_grpo import PrecisionAudit, parity_fingerprint
    from scripts.s49_gpt2_xl_symmetry import (commit_grpo_training, start_for,
                                              verify_adapter_base)
    from train.grpo import run_grpo_rows

    check_grpo_launch_state(cfg, result_dir, preflight_path)
    start = start_for(cfg, preflight_path)
    rows = build_grpo_split()
    if len(rows) != 245:
        raise IntegrityError("experiment 049 GRPO row count changed")
    prompts = [row["prompt"] for row in rows[:2]]
    audit = PrecisionAudit(allow_initial_zero_gradient=True)
    setup = {}
    recorded = []
    length_seen = []
    reference_before = {}

    def trainer_setup(trainer):
        setup.update(check_lora_setup(trainer, "grpo", package="049"))
        audit.attach(trainer)
        reference_before.update(parity_fingerprint(
            trainer, trainer.processing_class, prompts, start, cfg, "A"))
        trainer.add_callback(completion_length_gate(
            length_seen, package="049"))
        trainer.add_callback(_wandb_callback(
            result_dir, cfg, "grpo", recorded))

    def trainer_finished(trainer):
        # compare the disabled reference before and after the GRPO schedule
        report = audit.finish(trainer)
        reference_after = parity_fingerprint(
            trainer, trainer.processing_class, prompts, start, cfg, "A")
        report["reference_behaviour"] = reference_behaviour_report(
            reference_before, reference_after, package="049")
        report["lora_setup"] = setup
        report["wandb_run_record"] = recorded[0] if recorded else None
        report["completion_length_verified"] = bool(length_seen)
        atomic_write_json(precision_path, report)

    run_grpo_rows(
        cfg, "cuda", rows, start["path"], "local",
        os.path.relpath(start["path"], cfg.output_root), out_dir=out_dir,
        report_to=["wandb"], log_dir=result_dir,
        run_name=f"{os.path.basename(result_dir)}_grpo",
        allow_oom_fallback=False, trainer_setup=trainer_setup,
        trainer_finished=trainer_finished, save_final=True)
    precision = read_json(precision_path)
    precision["adapter_base"] = verify_adapter_base(
        cfg, out_dir, start)
    atomic_write_json(precision_path, precision)
    return commit_grpo_training(
        cfg, result_dir, out_dir, precision_path, preflight_path)


def _run_endpoint(cfg, result_dir, scratch_dir, adapter_dir):
    from scripts.s49_gpt2_xl_symmetry import (FAMILY_SPECS_049, commit_eval,
                                              eval_state, family_of,
                                              training_record)

    branch = FAMILY_SPECS_049[family_of(cfg)]["kind"]
    binding = training_record(cfg, result_dir)
    # keep evaluation tied to the adapter in the training record
    expected_adapter = os.path.realpath(binding["artifact"]["path"])
    if os.path.realpath(adapter_dir) != expected_adapter:
        raise IntegrityError("experiment 049 evaluation adapter differs")
    state = eval_state(cfg, result_dir, binding)
    if state == "tampered":
        raise IntegrityError(f"{result_dir}: endpoint changed")
    if state == "committed":
        return read_json(os.path.join(
            result_dir, f"049_{branch}_eval_manifest.json"))
    run_eval(
        cfg, branch, scratch_dir, adapter_dir=adapter_dir, package="049")
    return commit_eval(cfg, result_dir, scratch_dir, binding)


def _required_arguments(args):
    required = {
        "sft": ("out_dir",),
        "grpo": ("out_dir", "precision_report"),
        "eval": ("scratch_dir", "adapter_dir"),
    }
    missing = [name for name in required[args.task]
               if not getattr(args, name)]
    if missing:
        raise ValueError(f"{args.task} is missing arguments: {missing}")


def check_task(cfg, task):
    from scripts.s49_gpt2_xl_symmetry import family_of

    family = family_of(cfg)
    expected = "grpo" if family == "base_grpo" else "sft"
    if task != "eval" and task != expected:
        raise IntegrityError("experiment 049 task differs from its family")
    return family


def check_grpo_launch_state(cfg, result_dir, preflight_path):
    from scripts.s49_gpt2_xl_symmetry import arm_state

    state = arm_state(cfg, result_dir, preflight_path)["state"]
    training_path = os.path.join(result_dir, "049_grpo_training.json")
    if state not in ("absent", "interrupted") or os.path.exists(training_path):
        raise IntegrityError(
            f"{result_dir}: GRPO training cannot start from state {state}")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("sft", "grpo", "eval"),
                        required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--launch-record", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--scratch-dir")
    parser.add_argument("--precision-report")
    parser.add_argument("--adapter-dir")
    args = parser.parse_args()
    _required_arguments(args)

    import torch
    from config import load_config
    from obs import gpu_snapshot, versions
    from scripts.s46_gpt2_xl import ALLOCATOR_CONFIG
    from scripts.s49_gpt2_xl_symmetry import (check_card_vram,
                                              check_registered_config,
                                              verify_launch)
    from train.model_utils import set_all_seeds

    cfg = check_registered_config(load_config(args.config))
    family = check_task(cfg, args.task)
    # check the fixed launch and card before my trainer is constructed
    verify_launch(args.launch_record, args.preflight)
    allocator_config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if allocator_config != ALLOCATOR_CONFIG:
        raise IntegrityError("experiment 049 CUDA allocator setting changed")
    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    check_card_vram(total_vram_gib, family)
    set_all_seeds(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    status = {
        "completed": False,
        "task": args.task,
        "seed": cfg.train.seed,
        "family": family,
        "allocator_config": allocator_config,
        "gpu": gpu_snapshot(),
        "versions": versions(),
    }
    started = time.time()
    failure = None
    result = None
    start_step = latest_checkpoint_step(args.out_dir or "")
    try:
        if args.task == "sft":
            result = _run_sft(
                cfg, args.out_dir, args.result_dir, args.preflight)
        elif args.task == "grpo":
            result = _run_grpo(
                cfg, args.out_dir, args.result_dir,
                args.precision_report, args.preflight)
        else:
            result = _run_endpoint(
                cfg, args.result_dir, args.scratch_dir, args.adapter_dir)
        status["completed"] = True
    except BaseException as exc:
        failure = exc
        status.update(classify_failure(exc, torch))
    finally:
        status["wall_seconds"] = round(time.time() - started, 1)
        status["start_step"] = start_step
        status["end_step"] = latest_checkpoint_step(args.out_dir or "")
        status["completed_updates"] = max(
            0, status["end_step"] - start_step)
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            peak = torch.cuda.max_memory_reserved()
            status["peak_reserved_gib"] = round(peak / 1024**3, 2)
            status["peak_reserved_fraction"] = peak / total
            status["total_vram_gib"] = round(total / 1024**3, 2)
        if result is not None:
            status["result"] = result
        save_status(args.status, status)
        if failure is not None:
            raise failure


if __name__ == "__main__":
    main()
