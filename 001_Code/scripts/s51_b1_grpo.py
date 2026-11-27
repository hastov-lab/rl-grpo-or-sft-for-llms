# GPT-2 XL Cold GRPO with B1 answer reader

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config, run_tag
from file_utils import read_json
from scripts.s12_eval import IntegrityError, atomic_write_json, sha256
from scripts.s50_b1_grpo import (PINS, bind_run, checkpoint_step, digest, evaluate,
                                 paths_for, run_lock, stopped_outcome, verify_record)


CODE = Path(__file__).resolve().parents[1]
BASE_TAG = "046_COLAB_FULL_GPT2_B15_base"


def validate_config(cfg):
    if cfg.experiment != "051" or cfg.train.seed not in (0, 1, 2):
        raise IntegrityError("051 requires one of the three registered seeds")
    old = load_config(CODE / "configs" / f"049_base_grpo_colab_gpt2_b15_seed{cfg.train.seed}.yaml")
    old.experiment = "051"
    if asdict(cfg) != asdict(old):
        raise IntegrityError("051 differs from its 049 comparator beyond experiment identity")


def source_for(cfg):
    from checkpoint import check_artifact_complete
    from scripts.s46_gpt2_xl import eval_state, verify_technical_records

    root = Path(cfg.output_root)
    result = root / "results" / BASE_TAG
    base_path = result / "046_base_manifest.json"
    parity = root / "results/046_technical/046_xl_parity.json"
    tokens = root / "results/046_technical/046_token_audit.json"
    source_cfg = load_config(CODE / "configs/046_colab_gpt2_b15_seed0.yaml")
    verified = verify_technical_records(source_cfg, str(base_path), str(parity), str(tokens))
    base = verified["base"]
    artifact = check_artifact_complete(base["artifact"], expected="full")
    # reuse the same pretrained base, not SFT parent or old Cold adapter
    if (base.get("completed") is not True or base.get("experiment") != "046"
            or base.get("role") != "immutable_base" or base.get("source") != "gpt2-xl"
            or artifact.get("family") != "scratch_gpt" or artifact.get("scale") != "GPT2_B15"
            or eval_state(str(result), "base", base) != "committed"):
        raise IntegrityError("051 pretrained GPT-2 XL source is not reusable")
    endpoint = result / "046_base_eval_manifest.json"
    return {"kind": "immutable_base", "path": str(Path(base["artifact"]).resolve()),
            "base_artifact_sha256": base["artifact_sha256"],
            "records": {str(p): sha256(str(p)) for p in (base_path, parity, tokens, endpoint)}}


def run_binding(cfg, rows, eval_rows):
    import gpt2_scratch
    from obs import versions

    validate_config(cfg)
    source = source_for(cfg)
    actual = versions()
    if any(actual.get(name) != version for name, version in PINS.items()):
        raise IntegrityError(f"051 library versions differ from the registered environment: {actual}")
    if len(rows) != 245 or len(eval_rows) != 1884:
        raise IntegrityError("051 requires 245 training prompts and 1884 evaluation rows")
    files = [*CODE.glob("data/*.py"), *CODE.glob("train/*.py"), *CODE.glob("eval/*.py")]
    files += [CODE / p for p in (
        "config.py", "checkpoint.py", "enums.py", "file_utils.py", "obs.py",
        "scripts/s51_b1_grpo.py", "scripts/s50_b1_grpo.py", "scripts/s12_eval.py",
        "scripts/s42_worker.py", "scripts/s46_worker.py", "scripts/s46_gpt2_xl.py",
        "scripts/s49_gpt2_xl_symmetry.py", "scripts/s35_matched_grpo.py", "scripts/grpo_worker.py")]
    scratch = Path(gpt2_scratch.__file__).resolve().parent
    return {"parser": "B1", "historical_comparator": f"049_base_grpo_seed{cfg.train.seed}",
            "config": json.loads(json.dumps(asdict(cfg), default=str)),
            "source": source, "versions": actual,
            "grpo_rows_sha256": digest(rows), "eval_rows_sha256": digest(eval_rows),
            "code": {str(p.relative_to(CODE)): sha256(str(p)) for p in sorted(files)},
            "scratch_code": {str(p.relative_to(scratch)): sha256(str(p)) for p in sorted(scratch.rglob("*.py"))}}


def train(cfg, rows, result, checkpoint, binding):
    from obs import resolve_report_to, verify_a2_verdict, write_manifest
    from train.grpo import run_grpo_rows
    from train.parser_diagnostics import attach_parser_diagnostics, DIAGNOSTICS_FILE
    from scripts.grpo_worker import classify_failure
    from scripts.s35_matched_grpo import PrecisionAudit, parity_fingerprint
    from scripts.s42_worker import (capture_wandb_run, completion_length_gate,
                                    starvation_summary, step_rows, wandb_records)
    from scripts.s46_worker import check_lora_setup, reference_behaviour_report
    from scripts.s49_gpt2_xl_symmetry import verify_adapter_base

    if (result / "training_complete.json").exists() and (result / "terminal_stop.json").exists():
        raise IntegrityError("051 cannot be both completed and stopped")
    for name in ("training_complete.json", "terminal_stop.json"):
        if (result / name).exists():
            return verify_record(result / name, result, checkpoint, binding)
    failure = result / "last_failure.json"
    if failure.exists() and read_json(failure).get("error_type") != "KeyboardInterrupt":
        raise IntegrityError("previous technical failure needs review before resume")
    initial_step = checkpoint_step(checkpoint)
    diagnostic = None
    started = time.time()
    start = binding["source"]
    base_value = os.path.relpath(start["path"], cfg.output_root)
    prompts = [row["prompt"] for row in rows[:2]]
    audit = PrecisionAudit(allow_initial_zero_gradient=True)
    lora_setup, reference_before = {}, {}
    length_seen = []
    precision_path = result / "051_precision.json"

    def setup(trainer):
        nonlocal diagnostic
        lora_setup.update(check_lora_setup(trainer, "grpo", package="051"))
        audit.attach(trainer)
        reference_before.update(parity_fingerprint(
            trainer, trainer.processing_class, prompts, start, cfg, "A"))
        trainer.add_callback(completion_length_gate(length_seen, package="051"))
        diagnostic = attach_parser_diagnostics(
            trainer, cfg, str(result), str(checkpoint), digest(binding))
        from transformers import TrainerCallback

        class capture_run(TrainerCallback):
            recorded = False

            def on_log(self, args, state, control, logs=None, **kwargs):
                if "loss" in (logs or {}) and not self.recorded:
                    capture_wandb_run(str(result), cfg, state.global_step, package="051")
                    self.recorded = True
                return control

        trainer.add_callback(capture_run())

    def finished(trainer):
        # keep the original XL precision and disabled reference checks
        report = audit.finish(trainer)
        after = parity_fingerprint(trainer, trainer.processing_class, prompts, start, cfg, "A")
        report["reference_behaviour"] = reference_behaviour_report(reference_before, after, package="051")
        report["lora_setup"] = lora_setup
        if not length_seen:
            raise IntegrityError("051 completion length was not recorded")
        report["completion_length_verified"] = True
        atomic_write_json(str(precision_path), report)

    status = {"binding_sha256": digest(binding), "initial_step": initial_step}
    try:
        report_to = resolve_report_to(cfg)
        if "wandb" not in report_to:
            raise IntegrityError("051 requires the registered W&B tracking")
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ["WANDB_RESUME"] = "never"
        write_manifest(cfg, str(result), {"parser": "B1", "source": start})
        run_grpo_rows(cfg, "cuda", rows, start["path"], "local", base_value,
                      out_dir=str(checkpoint), report_to=report_to, log_dir=str(result),
                      run_name=run_tag(cfg) + "_grpo", allow_oom_fallback=False,
                      trainer_setup=setup, trainer_finished=finished, save_final=True)
        if diagnostic is None or diagnostic.summary()["completed_updates"] != cfg.train.max_steps:
            raise IntegrityError("051 did not reach the registered endpoint")
        precision = read_json(precision_path)
        precision["adapter_base"] = verify_adapter_base(cfg, str(checkpoint), start)
        atomic_write_json(str(precision_path), precision)
        from scripts.s12_eval import ADAPTER_FILES
        trainlog = step_rows(str(result / "grpo_trainlog.json"), cfg.train.max_steps,
                             ("rewards/accuracy_reward/mean", "rewards/format_reward/mean",
                              "frac_reward_zero_std", "kl"))
        # A2 remains observed, while A1 still stops at the original threshold.
        verify_a2_verdict(str(checkpoint / "a2_verdict.json"), cfg, trainlog[:150])
        status["wandb_runs"] = wandb_records(str(result), cfg, package="051")
        status.update(completed=True, parser_diagnostics=diagnostic.summary(),
                      starvation=starvation_summary(trainlog), files={
            **{f"checkpoints/{name}": sha256(str(checkpoint / name)) for name in (*ADAPTER_FILES, "a2_verdict.json")},
            **{f"results/{name}": sha256(str(result / name)) for name in
               (DIAGNOSTICS_FILE, "grpo_trainlog.json", "manifest.json", "051_precision.json")}})
        atomic_write_json(str(result / "training_complete.json"), status)
    except BaseException as exc:
        outcome = stopped_outcome(exc, cfg, diagnostic)
        if outcome:
            status["wandb_runs"] = wandb_records(str(result), cfg, package="051")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", choices=("preflight", "train", "eval"), required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    validate_config(cfg)
    from data.splits import build_grpo_split, build_eval_split
    rows, eval_rows = build_grpo_split(), build_eval_split()
    binding = run_binding(cfg, rows, eval_rows)
    _, result, checkpoint = paths_for(cfg)
    if args.task == "preflight":
        if (result / "run_binding.json").exists() and read_json(result / "run_binding.json") != binding:
            raise IntegrityError("051 existing run binding changed")
        print(json.dumps({"run": run_tag(cfg), "binding_sha256": digest(binding),
                          "result_dir": str(result), "checkpoint_dir": str(checkpoint)}, indent=2))
        return
    import torch
    if (not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0)
            or not 75 <= torch.cuda.get_device_properties(0).total_memory / 1024**3 < 90):
        raise IntegrityError("051 requires an A100 80GB")
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != "expandable_segments:True":
        raise IntegrityError("051 requires the registered expandable segments allocator")
    with run_lock(result):
        bind_run(result, checkpoint, binding)
        status = (train(cfg, rows, result, checkpoint, binding) if args.task == "train" else
                  evaluate(cfg, eval_rows, result, checkpoint, binding))
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
