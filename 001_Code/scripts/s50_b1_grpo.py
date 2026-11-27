# Qwen 0.5B Cold GRPO with the B1 answer reader.

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config, run_tag
from file_utils import read_json
from scripts.s12_eval import IntegrityError, atomic_write_json, read_jsonl, sha256


CODE = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B"
PINS = {"trl": "1.7.1", "transformers": "5.5.4", "peft": "0.19.1",
        "accelerate": "1.14.0", "datasets": "4.8.4", "torch": "2.11.0+cu128",
        "numpy": "2.0.2"}
BASE_DIR = "models/050_qwen_b05"
BASE_RECORD = "050_pretrained_source.json"


def comparator_order(cfg, rows):
    from data.subjects import TIERS_PROPOSAL, all_subjects
    # seed 0 predates the tier swap, so its flattened subject order differs
    subjects = ([s for group in TIERS_PROPOSAL.values() for s in group]
                if cfg.train.seed == 0 else all_subjects())
    rank = {subject: index for index, subject in enumerate(subjects)}
    return sorted(rows, key=lambda row: rank[row["subject"]])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def validate_config(cfg):
    # identity and the disabled fallback are the only config diffs against 005
    if cfg.experiment != "050" or cfg.train.seed not in (0, 1, 2):
        raise IntegrityError("050 requires one of the three registered seeds")
    old = load_config(CODE / "configs" / f"005_colab_full_b05_base_grpo_seed{cfg.train.seed}.yaml")
    old.experiment, old.exploratory = "050", True
    old.exec.allow_oom_fallback = False
    if asdict(cfg) != asdict(old):
        raise IntegrityError("050 differs from its 005 comparator beyond the registered changes")


def paths_for(cfg):
    root = Path(cfg.output_root).resolve()
    tag = run_tag(cfg)
    result = root / "results" / tag
    checkpoint = root / "checkpoints" / tag / "grpo"
    for path in (result, checkpoint):
        if not path.resolve().is_relative_to(root) or path.resolve() != path:
            raise IntegrityError("050 output directory is redirected by a symlink")
    return root, result, checkpoint


def base_files(directory):
    directory = Path(directory)
    names = sorted(p for p in directory.rglob("*") if p.is_file()
                   and ".cache" not in p.parts and p.suffix in (".json", ".txt", ".safetensors"))
    if not (directory / "config.json").is_file() or not (directory / "tokenizer.json").is_file():
        raise IntegrityError("050 pretrained model or tokenizer is missing")
    if not any(p.suffix == ".safetensors" for p in names):
        raise IntegrityError("050 pretrained weights are missing")
    return {str(p.relative_to(directory)): sha256(str(p)) for p in names}


def run_binding(cfg, rows, eval_rows):
    # the usual training hash leaves out several execution and eval fields
    from obs import versions
    validate_config(cfg)
    root, _, _ = paths_for(cfg)
    source = read_json(root / BASE_RECORD)
    if (source.get("model_id") != MODEL_ID or source.get("files") != base_files(root / BASE_DIR)
            or len(source.get("revision", "")) != 40):
        raise IntegrityError("050 pretrained source binding changed")
    actual = versions()
    if any(actual.get(name) != version for name, version in PINS.items()):
        raise IntegrityError(f"050 library versions differ from the registered environment: {actual}")
    if len(rows) != 245 or len(eval_rows) != 1884:
        raise IntegrityError("050 requires 245 training prompts and 1884 evaluation rows")
    files = [*CODE.glob("data/*.py"), *CODE.glob("train/*.py"), *CODE.glob("eval/*.py")]
    files += [CODE / p for p in ("config.py", "checkpoint.py", "enums.py", "file_utils.py",
                               "obs.py", "scripts/s50_b1_grpo.py", "scripts/s12_eval.py",
                               "scripts/s42_worker.py", "scripts/grpo_worker.py")]
    return {"parser": "B1", "historical_comparator": f"005_seed{cfg.train.seed}",
            "config": json.loads(json.dumps(asdict(cfg), default=str)),
            "source": source, "versions": actual,
            "grpo_rows_sha256": digest(rows), "eval_rows_sha256": digest(eval_rows),
            "code": {str(p.relative_to(CODE)): sha256(str(p)) for p in sorted(files)}}


@contextmanager
def run_lock(directory):
    import fcntl
    import tempfile
    directory.mkdir(parents=True, exist_ok=True)
    # lock on the local runtime, Drive gives no guarantee on posix advisory locks
    path = Path(tempfile.gettempdir()) / f"inm363_050_{digest(str(directory.resolve()))}.lock"
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise IntegrityError("another worker holds this seed") from exc
        yield


def bind_run(result, checkpoint, binding):
    path = result / "run_binding.json"
    if path.exists():
        if read_json(path) != binding:
            raise IntegrityError("050 run binding changed; do not resume or overwrite it")
    else:
        if any(p.name != ".run.lock" for p in result.iterdir()) or (checkpoint.exists() and any(checkpoint.iterdir())):
            raise IntegrityError("050 has unbound files; inspect them before starting")
        atomic_write_json(str(path), binding)


def verify_record(path, result, checkpoint, binding):
    record = read_json(path)
    if record.get("binding_sha256") != digest(binding) or not record.get("files"):
        raise IntegrityError(f"{path.name}: wrong run binding")
    if path.name.endswith("_complete.json") and record.get("completed") is not True:
        raise IntegrityError(f"{path.name}: incomplete record")
    if path.name == "terminal_stop.json":
        outcome = record.get("outcome", {})
        if (record.get("completed") is not False or outcome.get("kind") not in ("A1", "A2", "A4")
                or outcome.get("step") != {"A1": 50, "A2": 150, "A4": 20}[outcome["kind"]]):
            raise IntegrityError("terminal stop has an invalid outcome")
    for tracker in record.get("wandb_runs", []):
        target = Path(tracker["path"])
        if not target.resolve().is_relative_to(result.resolve()) or sha256(str(target)) != tracker["sha256"]:
            raise IntegrityError("W&B run record changed")
    for name, expected in record.get("files", {}).items():
        parent, relative = name.split("/", 1)
        if parent not in ("results", "checkpoints") or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise IntegrityError(f"{path.name}: invalid artifact path")
        base = result if parent == "results" else checkpoint
        target = base / relative
        if not target.is_file() or sha256(str(target)) != expected:
            raise IntegrityError(f"{path.name}: {name} changed")
    return record


def checkpoint_step(checkpoint):
    # dropping back from a half written save would silently redo training updates
    candidates = sorted(checkpoint.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not candidates:
        return 0
    latest = candidates[-1]
    required = ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json",
                "optimizer.pt", "scheduler.pt", "rng_state.pth", "parser_diagnostics_binding.json")
    if any(not (latest / name).is_file() for name in required):
        raise IntegrityError("latest checkpoint is incomplete; do not silently fall back")
    step = int(latest.name.split("-")[-1])
    if read_json(latest / "trainer_state.json").get("global_step") != step:
        raise IntegrityError("checkpoint step differs from trainer state")
    return step


def stopped_outcome(exc, cfg, diagnostic):
    # a matching error message on its own is not enough to call a scientific stop
    from obs import a1_truncation, a2_immobility, projected_hours
    if type(exc) is not RuntimeError or diagnostic is None:
        return None
    rows = [row["training_log"] for row in diagnostic.record["steps"]]
    message, step = str(exc), len(rows)
    if message.startswith("sentinel abort: a1:") and step == 50:
        values = [r["completions/clipped_ratio"] for r in rows]
        if a1_truncation(values):
            return {"kind": "A1", "step": step, "mean_clipped_ratio": sum(values) / 50}
    if message.startswith("sentinel abort: a2:") and step == 150:
        early = sum(r["rewards/accuracy_reward/mean"] for r in rows[:50]) / 50
        mid = sum(r["rewards/accuracy_reward/mean"] for r in rows[100:150]) / 50
        kl = sum(r["kl"] for r in rows[100:150]) / 50
        if a2_immobility(early, mid, kl):
            return {"kind": "A2", "step": step, "early_accuracy": early,
                    "mid_accuracy": mid, "mid_kl": kl}
    if message.startswith("sentinel abort: a4:") and step == 20:
        hours = projected_hours([r["step_time"] for r in rows], cfg.train.max_steps)
        if hours > cfg.exec.max_train_hours:
            return {"kind": "A4", "step": step, "projected_hours": hours}
    return None


def train(cfg, rows, result, checkpoint, binding):
    from obs import resolve_report_to, write_manifest
    from train.grpo import run_grpo_rows
    from train.parser_diagnostics import attach_parser_diagnostics, DIAGNOSTICS_FILE
    from scripts.grpo_worker import classify_failure
    if (result / "training_complete.json").exists() and (result / "terminal_stop.json").exists():
        raise IntegrityError("050 cannot be both completed and stopped")
    for name in ("training_complete.json", "terminal_stop.json"):
        if (result / name).exists():
            return verify_record(result / name, result, checkpoint, binding)
    failure = result / "last_failure.json"
    if failure.exists() and read_json(failure).get("error_type") != "KeyboardInterrupt":
        raise IntegrityError("previous technical failure needs review before resume")
    initial_step = checkpoint_step(checkpoint)
    diagnostic = None
    started = time.time()
    root = Path(cfg.output_root)

    def setup(trainer):
        nonlocal diagnostic
        diagnostic = attach_parser_diagnostics(
            trainer, cfg, str(result), str(checkpoint), digest(binding))
        from transformers import TrainerCallback
        from scripts.s42_worker import capture_wandb_run

        class capture_run(TrainerCallback):
            recorded = False

            def on_log(self, args, state, control, logs=None, **kwargs):
                if "loss" in (logs or {}) and not self.recorded:
                    capture_wandb_run(str(result), cfg, state.global_step, package="050")
                    self.recorded = True
                return control

        trainer.add_callback(capture_run())

    # the local source is a pinned pretrained snapshot, not a trained checkpoint
    status = {"binding_sha256": digest(binding), "initial_step": initial_step}
    try:
        report_to = resolve_report_to(cfg)
        if "wandb" not in report_to:
            raise IntegrityError("050 requires the registered W&B tracking")
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ["WANDB_RESUME"] = "never"
        write_manifest(cfg, str(result), {"parser": "B1", "source": binding["source"]})
        run_grpo_rows(cfg, "cuda", rows, str(root / BASE_DIR), "local", BASE_DIR,
                      out_dir=str(checkpoint), report_to=report_to,
                      log_dir=str(result), run_name=run_tag(cfg) + "_grpo",
                      allow_oom_fallback=False, trainer_setup=setup)
        if diagnostic is None or diagnostic.summary()["completed_updates"] != cfg.train.max_steps:
            raise IntegrityError("050 did not reach the registered endpoint")
        from scripts.s12_eval import ADAPTER_FILES
        if read_json(checkpoint / "adapter_base.json") != {"kind": "local", "value": BASE_DIR}:
            raise IntegrityError("050 adapter does not point to its bound pretrained source")
        from scripts.s42_worker import wandb_records
        status["wandb_runs"] = wandb_records(str(result), cfg, package="050")
        status.update(completed=True, parser_diagnostics=diagnostic.summary(), files={
            **{f"checkpoints/{name}": sha256(str(checkpoint / name)) for name in ADAPTER_FILES},
            **{f"results/{name}": sha256(str(result / name)) for name in
               (DIAGNOSTICS_FILE, "grpo_trainlog.json", "manifest.json")}})
        atomic_write_json(str(result / "training_complete.json"), status)
    except BaseException as exc:
        outcome = stopped_outcome(exc, cfg, diagnostic)
        if outcome:
            from scripts.s42_worker import wandb_records
            status["wandb_runs"] = wandb_records(str(result), cfg, package="050")
            status.update(completed=False, outcome=outcome, parser_diagnostics=diagnostic.summary(),
                          files={f"results/{DIAGNOSTICS_FILE}": sha256(str(result / DIAGNOSTICS_FILE)),
                                 "results/manifest.json": sha256(str(result / "manifest.json"))})
            atomic_write_json(str(result / "terminal_stop.json"), status)
        else:
            atomic_write_json(str(failure), dict(status, **classify_failure(exc)))
            raise
    finally:
        if diagnostic is not None:
            diagnostic.close_attempt()
        from obs import append_ledger, gpu_snapshot
        append_ledger(str(result / "timing.jsonl"), {
            "stage": "grpo", "initial_step": initial_step,
            "seconds": time.time() - started, "gpu": gpu_snapshot()})
    return status


def evaluate(cfg, rows, result, checkpoint, binding):
    from data.mmlu_b1 import extract_letter
    from eval.harness import run_eval_rows, run_logprob_rows
    from eval.metrics import build_pred_row
    if (result / "terminal_stop.json").exists():
        stopped = verify_record(result / "terminal_stop.json", result, checkpoint, binding)
        if any((result / name).exists() for name in ("grpo_preds.jsonl", "grpo_logprob_preds.jsonl", "evaluation_complete.json")):
            raise IntegrityError("stopped arm has an evaluated endpoint")
        return {"evaluation": "not applicable", "outcome": stopped["outcome"]}
    verify_record(result / "training_complete.json", result, checkpoint, binding)
    if (result / "evaluation_complete.json").exists():
        return verify_record(result / "evaluation_complete.json", result, checkpoint, binding)
    files = {}
    # each test pass commits on its own, an interrupted one needs a look before resume
    for stage, runner, pred_name, metric_name in (
            ("generation", run_eval_rows, "grpo_preds.jsonl", "grpo_eval.json"),
            ("probe", run_logprob_rows, "grpo_logprob_preds.jsonl", "grpo_logprob.json")):
        complete, pending = result / f"{stage}_complete.json", result / f"{stage}_pending.json"
        if complete.exists():
            record = verify_record(complete, result, checkpoint, binding)
            files.update(record["files"])
            continue
        if pending.exists() or (result / pred_name).exists():
            raise IntegrityError(f"interrupted {stage} test contact requires review before another pass")
        atomic_write_json(str(pending), {"binding_sha256": digest(binding), "n_items": len(rows),
                                         "started": time.time(), "stage": stage})
        kwargs = {"extractor": extract_letter} if stage == "generation" else {}
        metrics = runner(cfg, "cuda", rows, adapter_path=str(checkpoint),
                         condition="grpo", out_dir=str(result), **kwargs)
        saved = read_jsonl(str(result / pred_name))
        if len(saved) != len(rows) or any(
                (x["gold"], x["subject"], x["tier"]) != (y["gold"], y["subject"], y["tier"])
                for x, y in zip(saved, rows)):
            raise IntegrityError("050 evaluation rows are not aligned")
        atomic_write_json(str(result / metric_name), metrics)
        current = {f"results/{name}": sha256(str(result / name)) for name in (pred_name, metric_name)}
        if stage == "generation":
            historical = [build_pred_row(r["output"], r["gold"], r["subject"], r["tier"], r["n_tokens"])
                          for r in saved]
            name = "grpo_historical_preds.jsonl"
            from scripts.s12_eval import atomic_write_text
            atomic_write_text(str(result / name), "".join(json.dumps(r) + "\n" for r in historical))
            current[f"results/{name}"] = sha256(str(result / name))
        atomic_write_json(str(complete), {"binding_sha256": digest(binding), "completed": True,
                                         "files": current, "seconds": time.time() - read_json(pending)["started"]})
        files.update(current)
    record = {"binding_sha256": digest(binding), "completed": True, "parser": "B1", "files": files}
    atomic_write_json(str(result / "evaluation_complete.json"), record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", choices=("preflight", "train", "eval"), required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    validate_config(cfg)
    from data.splits import build_grpo_split, build_eval_split
    rows = comparator_order(cfg, build_grpo_split())
    eval_rows = comparator_order(cfg, build_eval_split())
    binding = run_binding(cfg, rows, eval_rows)
    _, result, checkpoint = paths_for(cfg)
    if args.task == "preflight":
        if (result / "run_binding.json").exists() and read_json(result / "run_binding.json") != binding:
            raise IntegrityError("050 existing run binding changed")
        print(json.dumps({"run": run_tag(cfg), "binding_sha256": digest(binding),
                          "result_dir": str(result), "checkpoint_dir": str(checkpoint)}, indent=2))
        return
    import torch
    if (not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0)
            or not 39 <= torch.cuda.get_device_properties(0).total_memory / 1024**3 < 45):
        raise IntegrityError("050 requires an A100 40GB")
    with run_lock(result):
        bind_run(result, checkpoint, binding)
        status = (train(cfg, rows, result, checkpoint, binding) if args.task == "train" else
                  evaluate(cfg, eval_rows, result, checkpoint, binding))
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
