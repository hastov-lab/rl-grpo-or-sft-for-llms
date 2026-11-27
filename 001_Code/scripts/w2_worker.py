# run W2 training or evaluation in a disposable process
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.grpo_worker import classify_failure, save_status


def _base_value(cfg, start):
    return os.path.relpath(start["merged_dir"], cfg.output_root)


def _latest_checkpoint(path):
    steps = []
    for checkpoint in glob.glob(os.path.join(path or "", "checkpoint-*")):
        try:
            steps.append(int(checkpoint.rsplit("-", 1)[1]))
        except ValueError:
            continue
    return max(steps, default=0)


def train_branch(cfg, device, branch, bundle, start, out_dir, log_dir,
                 run_name, report_to):
    if cfg.train.sft_data != "synthetic_facts":
        raise ValueError("W2 worker requires synthetic_facts")
    base_val = _base_value(cfg, start)
    if branch == "sft":
        from train.sft import run_sft_rows
        return run_sft_rows(
            cfg, device, bundle["sft_rows"], start["merged_dir"], "local", base_val,
            out_dir=out_dir, report_to=report_to, log_dir=log_dir, run_name=run_name,
            allow_oom_fallback=cfg.exec.allow_oom_fallback, eval_rows=None)
    if branch == "grpo":
        from train.grpo import run_grpo_rows
        return run_grpo_rows(
            cfg, device, bundle["grpo_rows"], start["merged_dir"], "local", base_val,
            out_dir=out_dir, report_to=report_to, log_dir=log_dir, run_name=run_name,
            allow_oom_fallback=cfg.exec.allow_oom_fallback)
    raise ValueError(f"bad W2 training branch: {branch}")


def verify_adapter_start(adapter_dir, start):
    from file_utils import read_json

    try:
        base = read_json(os.path.join(adapter_dir, "adapter_base.json"))
    except Exception as exc:
        raise RuntimeError(f"{adapter_dir}: adapter base record is missing") from exc
    if base != {"kind": "local", "value": start["base_value"]}:
        raise RuntimeError(f"{adapter_dir}: adapter was trained from a different W2 start")


def eval_branch(cfg, device, branch, bundle, start, scratch_dir, adapter_dir=None):
    from eval.harness import run_eval_rows, run_logprob_rows
    from scripts.w2_probe import IDENTITY_FIELDS

    # load the baseline directly and endpoints through their saved adapters
    if branch == "baseline":
        model_path, adapter_path = start["merged_dir"], None
    elif branch in ("sft", "grpo"):
        if not adapter_dir:
            raise ValueError(f"{branch} evaluation requires an adapter")
        verify_adapter_start(adapter_dir, start)
        model_path, adapter_path = None, adapter_dir
    else:
        raise ValueError(f"bad W2 evaluation branch: {branch}")

    os.makedirs(scratch_dir, exist_ok=True)
    expected = [
        os.path.join(scratch_dir, f"{branch}_preds.jsonl"),
        os.path.join(scratch_dir, f"{branch}_eval.json"),
        os.path.join(scratch_dir, f"{branch}_logprob_preds.jsonl"),
        os.path.join(scratch_dir, f"{branch}_logprob.json"),
    ]
    if any(os.path.exists(path) for path in expected):
        raise RuntimeError(f"{scratch_dir}: evaluation scratch is not empty")

    gen = run_eval_rows(
        cfg, device, bundle["eval_rows"], adapter_path=adapter_path,
        model_path=model_path, condition=branch, out_dir=scratch_dir,
        identity_fields=IDENTITY_FIELDS)
    probe = run_logprob_rows(
        cfg, device, bundle["eval_rows"], adapter_path=adapter_path,
        model_path=model_path, condition=branch, out_dir=scratch_dir,
        identity_fields=IDENTITY_FIELDS, include_scores=True)
    with open(os.path.join(scratch_dir, f"{branch}_eval.json"), "w") as f:
        json.dump(gen, f, indent=2)
    with open(os.path.join(scratch_dir, f"{branch}_logprob.json"), "w") as f:
        json.dump(probe, f, indent=2)
    return {"generated": gen, "probe": probe}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("train", "eval"), required=True)
    parser.add_argument("--branch", choices=("baseline", "sft", "grpo"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--start-record", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--log-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--scratch-dir")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--report-to", nargs="*", default=[])
    args = parser.parse_args()

    import torch

    from config import load_config
    from obs import gpu_snapshot
    from scripts.env_switch import resolve_device
    from scripts.w2_probe import load_bundle, verify_start

    cfg = load_config(args.config)
    bundle = load_bundle(args.bundle)
    start = verify_start(cfg, args.start_record)
    device = resolve_device(cfg.env)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    status = {"completed": False, "task": args.task, "branch": args.branch,
              "gpu": gpu_snapshot()}
    if args.task == "train":
        status["start_step"] = _latest_checkpoint(args.out_dir)
    started = time.time()
    try:
        if args.task == "train":
            if args.branch == "baseline":
                raise ValueError("baseline has no training stage")
            if not args.out_dir or not args.log_dir or not args.run_name:
                raise ValueError("training requires out-dir, log-dir and run-name")
            train_branch(cfg, device, args.branch, bundle, start, args.out_dir,
                         args.log_dir, args.run_name, args.report_to)
        else:
            if not args.scratch_dir:
                raise ValueError("evaluation requires scratch-dir")
            eval_branch(cfg, device, args.branch, bundle, start,
                        args.scratch_dir, args.adapter_dir)
        status["completed"] = True
    except BaseException as exc:
        status.update(classify_failure(exc, torch))
        raise
    finally:
        status["wall_seconds"] = round(time.time() - started, 1)
        if args.task == "train":
            status["end_step"] = (cfg.train.max_steps if status["completed"]
                                  else _latest_checkpoint(args.out_dir))
            status["steps_this_attempt"] = max(
                0, status["end_step"] - status["start_step"])
        if torch.cuda.is_available():
            status["peak_vram_gib"] = round(
                torch.cuda.max_memory_reserved() / 1024**3, 2)
        else:
            status["peak_vram_gib"] = 0.0
        save_status(args.status, status)


if __name__ == "__main__":
    main()
