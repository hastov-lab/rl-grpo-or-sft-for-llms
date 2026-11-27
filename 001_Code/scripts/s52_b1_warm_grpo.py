# Qwen 0.5B Warm GRPO with B1 and the stored GSM8K-SFT parents

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
from scripts.s12_eval import ADAPTER_FILES, IntegrityError, atomic_write_json, sha256
from scripts.s28_compose import check_merged_complete, merged_hashes, resolve_source_adapter
from scripts.s50_b1_grpo import (MODEL_ID, PINS, bind_run, checkpoint_step, comparator_order,
                                digest, evaluate, paths_for, run_lock, stopped_outcome,
                                verify_record)


CODE = Path(__file__).resolve().parents[1]


def validate_config(cfg):
    if cfg.experiment != "052" or cfg.train.seed not in (0, 1, 2):
        raise IntegrityError("052 requires one of the three registered seeds")
    old = load_config(CODE / "configs" / f"006_colab_full_b05_seed{cfg.train.seed}.yaml")
    old.experiment, old.exploratory = "052", True
    old.exec.allow_oom_fallback = False
    if asdict(cfg) != asdict(old):
        raise IntegrityError("052 differs from its 006 comparator beyond the registered changes")


def source_for(cfg):
    root = Path(cfg.output_root).resolve()
    tag = f"006_COLAB_FULL_B05_sft_grpo_seed{cfg.train.seed}"
    parent = root / "checkpoints" / tag
    merged = parent / "sft_merged"
    adapter = parent / "sft"
    manifest_path = root / "results" / tag / "manifest.json"
    grpo_base_path = parent / "grpo/adapter_base.json"
    for path in (parent, merged, adapter, manifest_path, grpo_base_path):
        if path.resolve() != path:
            raise IntegrityError("052 historical source is redirected by a symlink")
    manifest = read_json(manifest_path)
    expected = {"run_tag": tag, "experiment": "006", "seed": cfg.train.seed,
                "condition": "SFT_GRPO", "scale": "B05", "env": "COLAB", "mode": "FULL",
                "model_id": MODEL_ID}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise IntegrityError("052 historical source has the wrong seed, model or route")
    old = load_config(CODE / "configs" / f"006_colab_full_b05_seed{cfg.train.seed}.yaml")
    # Check the recorded fields before comparing the hash from earlier schema.
    core = {k: manifest[k] for k in ("experiment", "env", "mode", "scale", "condition")}
    for section in ("train", "grpo", "lora", "reward"):
        recorded = manifest.get("config", {}).get(section, {})
        current = asdict(getattr(old, section))
        if not recorded or any(k not in current or current[k] != v for k, v in recorded.items()):
            raise IntegrityError(f"052 historical {section} settings differ")
        core[section] = recorded
    _, adapter_hashes = resolve_source_adapter(str(adapter), str(root))
    if (adapter / "config_hash.txt").read_text().strip() != digest(core):
        raise IntegrityError("052 SFT config hash differs from its historical manifest")
    value = str(merged.relative_to(root))
    if (read_json(adapter / "adapter_base.json") != {"kind": "hf_id", "value": MODEL_ID}
            or read_json(grpo_base_path) != {"kind": "local", "value": value}):
        raise IntegrityError("052 parent GRPO did not use this GSM8K-SFT source")
    # Reuse the existing merge. Do not rebuild it or load the historical GRPO adapter.
    check_merged_complete(str(merged))
    return {"kind": "gsm8k_sft", "source_run": tag, "path": str(merged), "base_value": value,
            "binding_level": "path_and_provenance", "byte_reconstruction": False,
            "adapter_sha256": adapter_hashes, "merged_sha256": merged_hashes(str(merged)),
            "records": {str(p.relative_to(root)): sha256(str(p))
                        for p in (manifest_path, grpo_base_path)}}


def run_binding(cfg, rows, eval_rows):
    from obs import versions
    validate_config(cfg)
    source = source_for(cfg)
    actual = versions()
    if any(actual.get(name) != version for name, version in PINS.items()):
        raise IntegrityError(f"052 library versions differ from the registered environment: {actual}")
    if len(rows) != 245 or len(eval_rows) != 1884:
        raise IntegrityError("052 requires 245 training prompts and 1884 evaluation rows")
    files = [*CODE.glob("data/*.py"), *CODE.glob("train/*.py"), *CODE.glob("eval/*.py")]
    files += [CODE / p for p in (
        "config.py", "checkpoint.py", "enums.py", "file_utils.py", "obs.py",
        "scripts/s52_b1_warm_grpo.py", "scripts/s50_b1_grpo.py", "scripts/s12_eval.py",
        "scripts/s28_compose.py", "scripts/s42_worker.py", "scripts/grpo_worker.py")]
    return {"parser": "B1", "historical_comparator": f"006_seed{cfg.train.seed}",
            "config": json.loads(json.dumps(asdict(cfg), default=str)),
            "source": source, "versions": actual,
            "grpo_rows_sha256": digest(rows), "eval_rows_sha256": digest(eval_rows),
            "code": {str(p.relative_to(CODE)): sha256(str(p)) for p in sorted(files)}}


def train(cfg, rows, result, checkpoint, binding):
    from obs import resolve_report_to, write_manifest
    from train.grpo import run_grpo_rows
    from train.parser_diagnostics import attach_parser_diagnostics, DIAGNOSTICS_FILE
    from scripts.grpo_worker import classify_failure
    if (result / "training_complete.json").exists() and (result / "terminal_stop.json").exists():
        raise IntegrityError("052 cannot be both completed and stopped")
    for name in ("training_complete.json", "terminal_stop.json"):
        if (result / name).exists():
            return verify_record(result / name, result, checkpoint, binding)
    failure = result / "last_failure.json"
    if failure.exists() and read_json(failure).get("error_type") != "KeyboardInterrupt":
        raise IntegrityError("previous technical failure needs review before resume")
    initial_step = checkpoint_step(checkpoint)
    diagnostic = None
    started = time.time()
    source = binding["source"]

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
                    capture_wandb_run(str(result), cfg, state.global_step, package="052")
                    self.recorded = True
                return control

        trainer.add_callback(capture_run())

    # A fresh GRPO adapter starts on the bound seed matched SFT merge
    status = {"binding_sha256": digest(binding), "initial_step": initial_step}
    try:
        report_to = resolve_report_to(cfg)
        if "wandb" not in report_to:
            raise IntegrityError("052 requires the registered W&B tracking")
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ["WANDB_RESUME"] = "never"
        write_manifest(cfg, str(result), {"parser": "B1", "source": source})
        run_grpo_rows(cfg, "cuda", rows, source["path"], "local", source["base_value"],
                      out_dir=str(checkpoint), report_to=report_to,
                      log_dir=str(result), run_name=run_tag(cfg) + "_grpo",
                      allow_oom_fallback=False, trainer_setup=setup)
        if diagnostic is None or diagnostic.summary()["completed_updates"] != cfg.train.max_steps:
            raise IntegrityError("052 did not reach the registered endpoint")
        if read_json(checkpoint / "adapter_base.json") != {"kind": "local", "value": source["base_value"]}:
            raise IntegrityError("052 adapter does not point to its bound GSM8K-SFT source")
        from scripts.s42_worker import wandb_records
        status["wandb_runs"] = wandb_records(str(result), cfg, package="052")
        status.update(completed=True, parser_diagnostics=diagnostic.summary(), files={
            **{f"checkpoints/{name}": sha256(str(checkpoint / name)) for name in ADAPTER_FILES},
            **{f"results/{name}": sha256(str(result / name)) for name in
               (DIAGNOSTICS_FILE, "grpo_trainlog.json", "manifest.json")}})
        atomic_write_json(str(result / "training_complete.json"), status)
    except BaseException as exc:
        outcome = stopped_outcome(exc, cfg, diagnostic)
        if outcome:
            from scripts.s42_worker import wandb_records
            status["wandb_runs"] = wandb_records(str(result), cfg, package="052")
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
    rows = comparator_order(cfg, build_grpo_split())
    eval_rows = comparator_order(cfg, build_eval_split())
    binding = run_binding(cfg, rows, eval_rows)
    _, result, checkpoint = paths_for(cfg)
    if args.task == "preflight":
        if (result / "run_binding.json").exists() and read_json(result / "run_binding.json") != binding:
            raise IntegrityError("052 existing run binding changed")
        print(json.dumps({"run": run_tag(cfg), "binding_sha256": digest(binding),
                          "source": binding["source"], "result_dir": str(result),
                          "checkpoint_dir": str(checkpoint)}, indent=2))
        return
    import torch
    if (not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0)
            or not 39 <= torch.cuda.get_device_properties(0).total_memory / 1024**3 < 45):
        raise IntegrityError("052 requires an A100 40GB")
    with run_lock(result):
        bind_run(result, checkpoint, binding)
        status = (train(cfg, rows, result, checkpoint, binding) if args.task == "train" else
                  evaluate(cfg, eval_rows, result, checkpoint, binding))
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
