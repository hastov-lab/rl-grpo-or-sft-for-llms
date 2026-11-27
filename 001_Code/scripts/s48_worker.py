# run experiment 048 tasks in disposable processes
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_utils import read_json
from scripts.grpo_worker import classify_failure, save_status
from scripts.s12_eval import IntegrityError, atomic_write_json
from scripts.s46_worker import (_check_lora_setup, _convert, _latest_checkpoint_step,
                                _merge, _parity, _reference_behaviour_report,
                                _run_eval)


def _wandb_callback(result_dir, cfg, stage, target):
    from scripts.s42_worker import _capture_wandb_run
    from transformers import TrainerCallback

    # keep the first online run identity across an interrupted stage
    class wandb_record(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step > 0 and not target:
                target.append(_capture_wandb_run(
                    result_dir, cfg, state.global_step,
                    package="048", stage=stage))
            return control

    return wandb_record()


def _run_sft(cfg, base_record_path, out_dir, result_dir):
    from data.gsm8k import load_sft
    from scripts.s42_worker import _wandb_records
    from scripts.s48_gpt2_lora_scales import (SCALE_SPECS_048,
                                              commit_sft_training,
                                              verify_training_base)
    from train.sft import run_sft_rows

    # train from the bound immutable base through the proven LoRA path
    base = verify_training_base(cfg, base_record_path)
    recorded = []
    setup = {}

    def trainer_setup(trainer):
        setup.update(_check_lora_setup(trainer, "sft"))
        trainer.add_callback(_wandb_callback(
            result_dir, cfg, "sft", recorded))

    run_sft_rows(
        cfg, "cuda", load_sft(), base["artifact"], "local",
        os.path.relpath(base["artifact"], cfg.output_root), out_dir=out_dir,
        report_to=["wandb"], log_dir=result_dir,
        run_name=f"{os.path.basename(result_dir)}_sft",
        allow_oom_fallback=False,
        eval_rows=load_sft(split="test", limit=cfg.train.sft_val_examples),
        trainer_setup=trainer_setup)
    expected_layers = SCALE_SPECS_048[cfg.scale.name]["layers"]
    if set(setup.get("target_module_counts", {}).values()) != {expected_layers}:
        raise IntegrityError("experiment 048 SFT target count differs")
    return commit_sft_training(
        cfg, result_dir, out_dir, base_record_path,
        _wandb_records(result_dir, cfg, package="048", stage="sft"))


def _run_grpo(cfg, start_dir, out_dir, result_dir, precision_path,
              sft_record_path):
    from data.splits import build_grpo_split
    from scripts.s35_matched_grpo import PrecisionAudit, parity_fingerprint
    from scripts.s42_worker import _completion_length_gate
    from scripts.s48_gpt2_lora_scales import (commit_grpo_training,
                                              verify_grpo_adapter_base)
    from train.grpo import run_grpo_rows

    # retain the disabled reference and completion length checks used by 046
    recorded = []
    length_seen = []
    audit = PrecisionAudit(allow_initial_zero_gradient=True)
    setup = {}
    rows = build_grpo_split()
    prompts = [row["prompt"] for row in rows[:2]]
    reference_before = {}

    def trainer_setup(trainer):
        setup.update(_check_lora_setup(trainer, "grpo"))
        audit.attach(trainer)
        reference_before.update(parity_fingerprint(
            trainer, trainer.processing_class, prompts,
            {"experiment": "048", "stage": "merged_sft",
             "path": os.path.abspath(start_dir)}, cfg, "A"))
        trainer.add_callback(_completion_length_gate(length_seen, package="048"))
        trainer.add_callback(_wandb_callback(
            result_dir, cfg, "grpo", recorded))

    def trainer_finished(trainer):
        report = audit.finish(trainer)
        reference_after = parity_fingerprint(
            trainer, trainer.processing_class, prompts,
            {"experiment": "048", "stage": "merged_sft",
             "path": os.path.abspath(start_dir)}, cfg, "A")
        report["reference_behaviour"] = _reference_behaviour_report(
            reference_before, reference_after)
        report["lora_setup"] = setup
        report["wandb_run_record"] = recorded[0] if recorded else None
        report["completion_length_verified"] = bool(length_seen)
        atomic_write_json(precision_path, report)

    run_grpo_rows(
        cfg, "cuda", rows, start_dir, "local",
        os.path.relpath(start_dir, cfg.output_root), out_dir=out_dir,
        report_to=["wandb"], log_dir=result_dir,
        run_name=f"{os.path.basename(result_dir)}_grpo",
        allow_oom_fallback=False, trainer_setup=trainer_setup,
        trainer_finished=trainer_finished, save_final=True)
    precision = read_json(precision_path)
    precision["adapter_base"] = verify_grpo_adapter_base(cfg, out_dir, start_dir)
    atomic_write_json(precision_path, precision)
    return commit_grpo_training(
        cfg, result_dir, out_dir, sft_record_path, start_dir, precision_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=(
        "convert", "parity", "sft", "merge", "grpo", "eval"),
                        required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--scratch-dir")
    parser.add_argument("--record")
    parser.add_argument("--base-record")
    parser.add_argument("--sft-record")
    parser.add_argument("--start-dir")
    parser.add_argument("--precision-report")
    parser.add_argument("--branch", choices=("base", "sft", "grpo"))
    parser.add_argument("--adapter-dir")
    parser.add_argument("--model-path")
    args = parser.parse_args()

    import torch
    from config import load_config
    from obs import _versions, gpu_snapshot
    from scripts.s48_gpt2_lora_scales import (ALLOCATOR_CONFIG,
                                              check_card_vram,
                                              check_registered_config,
                                              commit_eval, eval_state)
    from train.model_utils import set_all_seeds

    cfg = check_registered_config(load_config(args.config))
    allocator_config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if allocator_config != ALLOCATOR_CONFIG:
        raise IntegrityError("experiment 048 CUDA allocator setting changed")
    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    check_card_vram(total_vram_gib, cfg.scale.name)
    set_all_seeds(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    status = {
        "completed": False,
        "task": args.task,
        "seed": cfg.train.seed,
        "allocator_config": allocator_config,
        "gpu": gpu_snapshot(),
        "versions": _versions(),
    }
    started = time.time()
    failure = None
    result = None
    start_step = _latest_checkpoint_step(args.out_dir or "")
    try:
        if args.task == "convert":
            result = _convert(
                cfg, args.scratch_dir, args.out_dir, args.record, package="048")
        elif args.task == "parity":
            result = _parity(
                cfg, args.base_record, args.record, package="048")
        elif args.task == "sft":
            result = _run_sft(
                cfg, args.base_record, args.out_dir, args.result_dir)
        elif args.task == "merge":
            result = _merge(
                cfg, args.adapter_dir, args.start_dir, args.out_dir)
        elif args.task == "grpo":
            result = _run_grpo(
                cfg, args.start_dir, args.out_dir, args.result_dir,
                args.precision_report, args.sft_record)
        else:
            binding = read_json(args.record)
            state = eval_state(args.result_dir, args.branch, binding)
            if state == "tampered":
                raise IntegrityError(f"{args.result_dir}: endpoint changed")
            if state == "committed":
                result = read_json(os.path.join(
                    args.result_dir, f"048_{args.branch}_eval_manifest.json"))
            else:
                _run_eval(
                    cfg, args.branch, args.scratch_dir,
                    adapter_dir=args.adapter_dir, model_path=args.model_path)
                result = commit_eval(
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
