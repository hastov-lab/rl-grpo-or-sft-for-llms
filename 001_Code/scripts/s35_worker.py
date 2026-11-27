#!/usr/bin/env python3
# 035 task: keep every model heavy inside one disposable process
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_utils import read_json
from scripts.grpo_worker import classify_failure, save_status


def _save_json(path, payload):
    from scripts.s12_eval import atomic_write_json
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    atomic_write_json(path, payload)


def _latest_checkpoint_step(out_dir):
    steps = []
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if name.startswith("checkpoint-"):
                try:
                    steps.append(int(name.rsplit("-", 1)[1]))
                except ValueError:
                    pass
    return max(steps, default=0)


def _train(cfg, device, start, rung, out_dir, result_dir, report_to,
           precision_path, pilot=False):
    from checkpoint import config_hash
    from data.splits import build_grpo_split
    from scripts.s12_eval import sha256
    from scripts.s35_matched_grpo import (B15_WANDB_RUNS, EXECUTION_BINDING,
                                          PrecisionAudit,
                                          commit_training_complete)
    from train.grpo import run_grpo_rows

    rows = build_grpo_split()
    if pilot:
        cfg.log.save_steps = 30
        cfg.log.save_total_limit = 1
    base_value = os.path.relpath(start["start"], cfg.output_root)
    run_name = f"{os.path.basename(result_dir)}_grpo"
    wandb_record_dir = None
    wandb_context = None
    if cfg.scale.name == "B15" and "wandb" in report_to:
        wandb_record_dir = os.path.join(result_dir, B15_WANDB_RUNS)
        wandb_context = {
            "scale": "B15",
            "seed": cfg.train.seed,
            "parameter_mode": cfg.train.grpo_parameter_mode,
            "rung": rung,
            "project": cfg.log.wandb_project,
            "run_name": run_name,
            "config_hash": config_hash(cfg),
            "execution_binding_sha256": sha256(
                os.path.join(result_dir, EXECUTION_BINDING)),
        }
    # the pilot keeps the 500-step schedule and stops through a callback
    audit = PrecisionAudit(
        stop_after=30 if pilot else None,
        wandb_record_dir=wandb_record_dir, wandb_context=wandb_context)

    def finished(trainer):
        _save_json(precision_path, audit.finish(trainer))

    run_grpo_rows(
        cfg, device, rows, start["start"], "local", base_value,
        out_dir=out_dir, report_to=report_to, log_dir=result_dir,
        run_name=run_name,
        allow_oom_fallback=False, trainer_setup=audit.attach,
        trainer_finished=finished, save_final=not pilot)
    if pilot:
        return None
    return commit_training_complete(
        cfg, result_dir, out_dir, start, rung, precision_path)


def _parity(cfg, start, rung, out_dir, record_path):
    from transformers import AutoTokenizer
    from data.splits import build_grpo_split
    from scripts.s35_matched_grpo import parity_fingerprint
    from obs import _versions, gpu_snapshot
    from train.grpo import _grpo_dataset, make_grpo_trainer

    rows = build_grpo_split()
    trainer = make_grpo_trainer(
        cfg, _grpo_dataset(rows), start["start"], out_dir, [],
        f"{os.path.basename(out_dir)}_parity",
        cfg.exec.grpo_batch_size, cfg.exec.grpo_grad_accum,
        cfg.exec.gradient_checkpointing)
    tokenizer = AutoTokenizer.from_pretrained(start["start"])
    record = parity_fingerprint(
        trainer, tokenizer, [row["prompt"] for row in rows[:4]], start, cfg, rung)
    record["gpu"] = gpu_snapshot()
    record["versions"] = _versions()
    record["allocator_config"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    _save_json(record_path, record)
    return record


def _eval(cfg, device, branch, start, output_dir, scratch_dir):
    from data.splits import build_eval_split
    from eval.harness import run_eval_rows, run_logprob_rows

    os.makedirs(scratch_dir, exist_ok=True)
    if os.listdir(scratch_dir):
        raise RuntimeError(f"{scratch_dir}: evaluation scratch is not empty")
    rows = build_eval_split()
    if branch == "start":
        probe = run_logprob_rows(
            cfg, device, rows, model_path=start["start"],
            condition="start", out_dir=scratch_dir)
        _save_json(os.path.join(scratch_dir, "start_logprob.json"), probe)
        return {"probe": probe}
    if branch == "lora":
        adapter_path, model_path = output_dir, None
    elif branch == "full":
        adapter_path, model_path = None, output_dir
    else:
        raise ValueError(f"bad experiment 035 evaluation branch: {branch}")
    generated = run_eval_rows(
        cfg, device, rows, adapter_path=adapter_path, model_path=model_path,
        condition="grpo", out_dir=scratch_dir)
    probe = run_logprob_rows(
        cfg, device, rows, adapter_path=adapter_path, model_path=model_path,
        condition="grpo", out_dir=scratch_dir)
    _save_json(os.path.join(scratch_dir, "grpo_eval.json"), generated)
    _save_json(os.path.join(scratch_dir, "grpo_logprob.json"), probe)
    return {"generated": generated, "probe": probe}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("pilot", "parity", "train", "eval"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-record", required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--branch", choices=("start", "lora", "full"))
    parser.add_argument("--scratch-dir")
    parser.add_argument("--pilot-report")
    parser.add_argument("--precision-report")
    parser.add_argument("--parity-record")
    parser.add_argument("--report-to", nargs="*", default=[])
    args = parser.parse_args()

    import torch
    from config import load_config
    from obs import _versions, gpu_snapshot
    from scripts.env_switch import resolve_device
    from scripts.s35_matched_grpo import (apply_rung, bind_execution,
                                          check_agc_authorisation,
                                          commit_pilot_report, pilot_passes)
    from train.model_utils import set_all_seeds

    cfg = load_config(args.config)
    check_agc_authorisation(cfg, args.rung, args.task)
    execution = apply_rung(cfg, args.rung)
    set_all_seeds(cfg.train.seed)
    start = read_json(args.start_record)
    if start.get("scale") != cfg.scale.name or start.get("seed") != cfg.train.seed:
        raise RuntimeError("experiment 035 start record does not match the config")
    device = resolve_device(cfg.env)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    status = {"completed": False, "task": args.task, "rung": args.rung,
              "execution": execution, "gpu": gpu_snapshot(), "versions": _versions(),
              "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}
    start_step = _latest_checkpoint_step(args.out_dir) if args.task in ("pilot", "train") else 0
    environment = {
        "gpu_device": status["gpu"].get("device"),
        "versions": status["versions"],
        "allocator_config": status["allocator_config"],
    }
    started = time.time()
    failure = None
    training_record = None
    try:
        if args.task in ("pilot", "train"):
            if not args.precision_report:
                raise ValueError("training requires a precision report path")
            bind_execution(args.result_dir, cfg, start, args.rung, args.task, environment)
            training_record = _train(
                cfg, device, start, args.rung, args.out_dir, args.result_dir,
                args.report_to, args.precision_report, pilot=args.task == "pilot")
        elif args.task == "parity":
            if not args.parity_record:
                raise ValueError("parity requires a record path")
            _parity(cfg, start, args.rung, args.out_dir, args.parity_record)
        else:
            if not args.branch or not args.scratch_dir:
                raise ValueError("evaluation requires a branch and scratch directory")
            _eval(cfg, device, args.branch, start, args.out_dir, args.scratch_dir)
        status["completed"] = True
    except BaseException as exc:
        failure = exc
        # classify the failure before anything else touches the status
        status.update(classify_failure(exc, torch))
    finally:
        status["wall_seconds"] = round(time.time() - started, 1)
        if args.task in ("pilot", "train"):
            status["start_step"] = start_step
            status["end_step"] = _latest_checkpoint_step(args.out_dir)
            status["completed_updates"] = max(0, status["end_step"] - start_step)
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            peak_reserved = torch.cuda.max_memory_reserved()
            peak_allocated = torch.cuda.max_memory_allocated()
            status["peak_vram_gib"] = round(peak_reserved / 1024**3, 2)
            status["peak_reserved_gib"] = round(peak_reserved / 1024**3, 2)
            status["peak_allocated_gib"] = round(peak_allocated / 1024**3, 2)
            status["total_vram_gib"] = round(total / 1024**3, 2)
            status["peak_reserved_fraction"] = peak_reserved / total
        else:
            status["peak_vram_gib"] = 0.0
            status["total_vram_gib"] = 0.0
            status["peak_reserved_fraction"] = 0.0
        if training_record is not None:
            status["training_record"] = training_record
        save_status(args.status, status)
        if args.task == "pilot":
            if not args.pilot_report:
                raise ValueError("pilot requires a report path")
            precision = (read_json(args.precision_report)
                         if args.precision_report and os.path.exists(args.precision_report) else {})
            checkpoint_bytes = 0
            checkpoint_dir = os.path.join(args.out_dir, "checkpoint-30")
            if os.path.isdir(checkpoint_dir):
                checkpoint_bytes = sum(
                    os.path.getsize(os.path.join(root, name))
                    for root, _, names in os.walk(checkpoint_dir) for name in names)
            report = {
                **status,
                "scale": cfg.scale.name,
                "seed": cfg.train.seed,
                "storage_dtype": "float32",
                "compute_dtype": "bfloat16",
                "gradient_dtype": (precision.get("gradient_dtypes") or [None])[0],
                "optimizer_state_dtypes": precision.get("optimizer_state_dtypes", []),
                "finite": bool(precision.get("gradients_finite", False)
                               and precision.get("parameters_finite", False)
                               and precision.get("optimizer_states_finite", False)),
                "precision_audit": precision,
                "checkpoint_bytes": checkpoint_bytes,
                "pilot_weights_retained": False,
            }
            if status["completed"] or status.get("cuda_oom"):
                # commit the pilot report even on OOM, that outcome is the measurement
                commit_pilot_report(args.pilot_report, report)
                status["pilot_passed"] = pilot_passes(read_json(args.pilot_report))
            else:
                _save_json(args.pilot_report + ".failed", report)
        if failure is not None:
            # raise the original failure last, after the status is safely written
            raise failure


if __name__ == "__main__":
    main()
