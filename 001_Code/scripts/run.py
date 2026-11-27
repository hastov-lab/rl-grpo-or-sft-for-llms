# run configured stages and reuse completed outputs on resume
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config, MODEL_IDS, SMOKE_MODEL_ID, run_tag
from enums import Condition, Mode
from scripts.env_switch import resolve_device, apply_smoke


def _fmt_secs(s):
    # 5025 -> '1h 23m 45s', keeps the timing lines readable on long runs
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def plan_stages(condition):
    # Which stages a condition needs, in order. Base eval always runs
    if condition == Condition.SFT_GRPO:
        return ["base_eval", "sft", "sft_eval", "grpo", "grpo_eval"]
    if condition in (Condition.BASE_GRPO, Condition.MCSFT_GRPO,
                     Condition.EXTERNAL_SFT_GRPO, Condition.MATCHED_LORA_GRPO,
                     Condition.MATCHED_FULL_GRPO):
        return ["base_eval", "grpo", "grpo_eval"]
    return ["base_eval", "sft", "sft_eval"]


def load_or_run(json_path, key, fn, timings=None, log=None):
    # reuse a completed stage output
    from checkpoint import save_metrics, load_metrics
    from obs import Timer
    if os.path.exists(json_path):
        if log:
            log.info(f"{key} already done, skipping")
        return load_metrics(json_path)
    with Timer() as t:
        res = fn()
    if timings is not None:
        timings[key] = t.seconds
    save_metrics(json_path, res)
    return res


def base_eval_dir(cfg):
    # base evals depend on the model and the eval protocol, not the seed or
    # condition, so they are shared inside an experiment but never across them
    return os.path.join(cfg.output_root, "results", "base_evals",
                        f"{cfg.experiment}_{cfg.env.name}_{cfg.mode.name}_{cfg.scale.name}")


def build_summary(cfg, base_eval, sft_eval, grpo_eval, base_lp, sft_lp, grpo_lp):
    # One json with everything the plots need. Deltas are against the reference
    # condition: sft for sft_grpo (and sft_only), the base for base_grpo
    from eval.metrics import grpo_delta

    def acc(ev):
        return ev["per_tier_acc"] if ev else None

    if cfg.condition == Condition.BASE_GRPO:
        ref = "base"
        ref_acc, ref_lp = acc(base_eval), acc(base_lp)
    elif cfg.condition in (Condition.MCSFT_GRPO, Condition.EXTERNAL_SFT_GRPO):
        # no SFT stage inside this run: the reference is the external
        # seed-matched SFT parent, copied by value with its tag recorded
        from scripts.s28_compose import external_reference, parent_results_dir_for
        parent = external_reference(parent_results_dir_for(cfg))
        ref = f"{parent['reference_kind']}:{parent['tag']}"
        sft_eval = {"per_tier_acc": parent["sft_acc"]}
        sft_lp = {"per_tier_acc": parent["sft_logprob_acc"]}
        ref_acc, ref_lp = parent["sft_acc"], parent["sft_logprob_acc"]
    else:
        ref = "sft"
        ref_acc, ref_lp = acc(sft_eval), acc(sft_lp)
    return {
        "scale": cfg.scale.name, "mode": cfg.mode.name,
        "condition": cfg.condition.name, "seed": cfg.train.seed,
        "base_acc": acc(base_eval), "sft_acc": acc(sft_eval), "grpo_acc": acc(grpo_eval),
        "reference": ref,
        "grpo_delta": grpo_delta(ref_acc, acc(grpo_eval)) if grpo_eval and ref_acc else None,
        "base_logprob_acc": acc(base_lp), "sft_logprob_acc": acc(sft_lp),
        "grpo_logprob_acc": acc(grpo_lp),
        "grpo_logprob_delta": grpo_delta(ref_lp, acc(grpo_lp)) if grpo_lp and ref_lp else None,
    }


def wipe_run_dirs(cfg):
    # --fresh support: clear this tag's old outputs so the run starts over instead
    # of skipping. Smoke only, full-run artifacts are too expensive to delete by flag
    import shutil
    if cfg.mode != Mode.SMOKE:
        raise SystemExit("--fresh applies only to smoke runs. For a fresh full run, set output_root to a new directory.")
    tag = run_tag(cfg)
    for d in (os.path.join(cfg.output_root, "results", tag),
              os.path.join(cfg.output_root, "checkpoints", tag),
              base_eval_dir(cfg)):
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"fresh: wiped {d}")


def wiring(cfg, device):
    from obs import resolve_report_to
    tag = run_tag(cfg)
    model_id = SMOKE_MODEL_ID if cfg.mode.name == "SMOKE" else MODEL_IDS[cfg.scale]
    print(f"env={cfg.env.name} mode={cfg.mode.name} scale={cfg.scale.name} condition={cfg.condition.name}")
    print(f"model={model_id} device={device} stages={plan_stages(cfg.condition)}")
    print(f"steps={cfg.train.max_steps} N={cfg.train.num_generations} seed={cfg.train.seed}")
    print(f"trackers={resolve_report_to(cfg)}")
    print(f"results -> {cfg.output_root}/results/{tag}/  checkpoints -> {cfg.output_root}/checkpoints/{tag}/")
    print("local smoke wiring ok")


def _adapter(d):
    return os.path.join(d, "adapter_model.safetensors")


_live = {}   # the in-flight ledger row, see full() and _flush_ledger()


def _flush_ledger():
    # Bill the session whatever happened. Called from main() on every exit path
    if "row" not in _live or _live.get("suppress"):
        return
    from obs import append_ledger, gpu_snapshot
    row = dict(_live["row"])
    row["timings_sec"] = dict(row["timings_sec"])
    row["eval_items"] = dict(row["eval_items"])
    row["gpu"] = gpu_snapshot()
    row["train_sec"] = row["timings_sec"].get("sft_train", 0) + row["timings_sec"].get("grpo_train", 0)
    append_ledger(_live["path"], row)
    _live.clear()


def full(cfg, device):
    from obs import (get_logger, write_manifest, gpu_snapshot, Timer, append_ledger,
                     resolve_report_to, configure_wandb)
    from train.model_utils import set_all_seeds, resolve_model_id
    from train.sft import run_sft
    from train.grpo import run_grpo
    from eval.harness import run_eval, run_logprob_eval
    from checkpoint import save_metrics, load_metrics

    if cfg.condition in (Condition.MATCHED_LORA_GRPO, Condition.MATCHED_FULL_GRPO):
        raise ValueError("experiment 035 matched arms require scripts/s35_worker.py")
    if cfg.experiment == "052":
        raise ValueError("experiment 052 requires scripts/s52_b1_warm_grpo.py")
    if cfg.experiment == "051":
        raise ValueError("experiment 051 requires scripts/s51_b1_grpo.py")
    if cfg.experiment == "050":
        raise ValueError("experiment 050 requires scripts/s50_b1_grpo.py")

    tag = run_tag(cfg)
    out = os.path.join(cfg.output_root, "results", tag)
    ckpt = os.path.join(cfg.output_root, "checkpoints", tag)
    bdir = base_eval_dir(cfg)
    os.makedirs(out, exist_ok=True)
    log = get_logger(out)
    set_all_seeds(cfg.train.seed)
    stages = plan_stages(cfg.condition)

    report_to = resolve_report_to(cfg)
    if "wandb" in report_to:
        tracking = configure_wandb(cfg, report_to)
        if tracking["key_source"] == "env":
            log.info("using wandb key from env")
        elif tracking["key_source"] == "file":
            log.info("loaded wandb api key from file")
        else:
            log.info(f"no wandb key file at {cfg.log.wandb_key_file}, using login/offline")
    log.info(f"run {tag} device={device} stages={stages} trackers={report_to} out={out}")

    timings = {}
    eval_items = {}
    t0 = time.time()
    # Live ledger row, flushed by main() even if this session crashes or gets
    # interrupted. The 006 session that died mid grpo-eval never got billed,
    # which is how the ledger ended up undercounting by hours
    _live["row"] = {"run_tag": tag, "condition": cfg.condition.name,
                    "timings_sec": timings, "eval_items": eval_items,
                    "train_steps": cfg.train.max_steps, "completed": False}
    _live["path"] = os.path.join(cfg.output_root, "results", "compute_ledger.jsonl")

    def _load_or_run(json_path, key, fn):
        had = os.path.exists(json_path)
        res = load_or_run(json_path, key, fn, timings=timings, log=log)
        if not had:
            log.info(f"{key} done in {_fmt_secs(timings[key])} | elapsed {_fmt_secs(time.time() - t0)}")
        return res

    def gen_or_skip(json_path, key, fn):
        # skip generation for cheap exploratory probe sweeps, generative evals take hours
        if cfg.eval.skip_generative:
            log.info(f"{key} skipped, probe-only run")
            return None
        return _load_or_run(json_path, key, fn)

    # base evals live in a shared dir: greedy decoding of the same base model does
    # not depend on the training seed or the condition, so never recompute them
    base_eval = gen_or_skip(os.path.join(bdir, "base_eval.json"), "base_eval",
                            lambda: run_eval(cfg, device, adapter_path=None, condition="base", out_dir=bdir))
    if base_eval:
        eval_items["base"] = base_eval.get("n_items")
    base_lp = _load_or_run(os.path.join(bdir, "base_logprob.json"), "base_logprob_eval",
                          lambda: run_logprob_eval(cfg, device, adapter_path=None, condition="base", out_dir=bdir))
    eval_items["base_logprob"] = base_lp.get("n_items")
    if base_eval:
        log.info(f"base per-tier acc: {base_eval['per_tier_acc']}")

    sft_eval = sft_lp = None
    sft_dir = os.path.join(ckpt, "sft")
    if "sft" in stages:
        if not os.path.exists(_adapter(sft_dir)):
            with Timer() as t:
                run_sft(cfg, device, out_dir=sft_dir, report_to=report_to, log_dir=out, run_name=f"{tag}_sft",
                        allow_oom_fallback=cfg.exec.allow_oom_fallback)
            timings["sft_train"] = t.seconds
            log.info(f"sft train done in {_fmt_secs(t.seconds)} | elapsed {_fmt_secs(time.time() - t0)}")
        else:
            log.info("sft adapter already there, not retraining")
        sft_eval = gen_or_skip(os.path.join(out, "sft_eval.json"), "sft_eval",
                               lambda: run_eval(cfg, device, adapter_path=sft_dir, condition="sft", out_dir=out))
        if sft_eval:
            eval_items["sft"] = sft_eval.get("n_items")
        sft_lp = _load_or_run(os.path.join(out, "sft_logprob.json"), "sft_logprob_eval",
                             lambda: run_logprob_eval(cfg, device, adapter_path=sft_dir, condition="sft", out_dir=out))
        eval_items["sft_logprob"] = sft_lp.get("n_items")
        if sft_eval:
            log.info(f"sft per-tier acc: {sft_eval['per_tier_acc']}")

    grpo_eval = grpo_lp = None
    if "grpo" in stages:
        grpo_dir = os.path.join(ckpt, "grpo")
        # sft_grpo chains from the sft adapter (merged inside run_grpo), base_grpo
        # does not, and external conditions chain from the resolved parent adapter
        if cfg.condition == Condition.SFT_GRPO:
            chain_from = sft_dir
        elif cfg.condition in (Condition.MCSFT_GRPO, Condition.EXTERNAL_SFT_GRPO):
            from scripts.s28_compose import resolve_source_adapter
            chain_from, _ = resolve_source_adapter(cfg.train.grpo_from_adapter, cfg.output_root)
        else:
            chain_from = None
        if not os.path.exists(_adapter(grpo_dir)):
            with Timer() as t:
                run_grpo(cfg, device, adapter_path=chain_from, out_dir=grpo_dir,
                         report_to=report_to, log_dir=out, run_name=f"{tag}_grpo",
                         allow_oom_fallback=cfg.exec.allow_oom_fallback)
            timings["grpo_train"] = t.seconds
            log.info(f"grpo train done in {_fmt_secs(t.seconds)} | elapsed {_fmt_secs(time.time() - t0)}")
        else:
            log.info("grpo adapter already there, not retraining")
        grpo_eval = gen_or_skip(os.path.join(out, "grpo_eval.json"), "grpo_eval",
                                lambda: run_eval(cfg, device, adapter_path=grpo_dir, condition="grpo", out_dir=out))
        if grpo_eval:
            eval_items["grpo"] = grpo_eval.get("n_items")
        grpo_lp = _load_or_run(os.path.join(out, "grpo_logprob.json"), "grpo_logprob_eval",
                              lambda: run_logprob_eval(cfg, device, adapter_path=grpo_dir, condition="grpo", out_dir=out))
        eval_items["grpo_logprob"] = grpo_lp.get("n_items")
        if grpo_eval:
            log.info(f"grpo per-tier acc: {grpo_eval['per_tier_acc']}")

    summary = build_summary(cfg, base_eval, sft_eval, grpo_eval, base_lp, sft_lp, grpo_lp)
    save_metrics(os.path.join(out, "summary.json"), summary)

    gpu = gpu_snapshot()
    write_manifest(cfg, out, extra={"model_id": resolve_model_id(cfg), "timings_sec": timings, "gpu": gpu})
    _live["row"]["completed"] = True   # actual ledger write happens in _flush_ledger
    log.info(f"done. total wall time {_fmt_secs(time.time() - t0)}")
    log.info(f"results -> {out}/  checkpoints -> {ckpt}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke_local.yaml")
    ap.add_argument("--stage", default="wiring", choices=["wiring", "all"])
    ap.add_argument("--fresh", action="store_true",
                    help="wipe this smoke tag's old outputs first instead of skipping them")
    ap.add_argument("--seed", type=int, default=None,
                    help="override train.seed, saves a config file per seed in sweeps")
    ap.add_argument("--no-ledger", action="store_true",
                    help="skip the ledger row, for summary passes where everything skipped")
    args = ap.parse_args()
    if args.no_ledger:
        _live["suppress"] = True

    cfg = apply_smoke(load_config(args.config))
    if args.seed is not None:
        cfg.train.seed = args.seed
        print(f"seed overridden to {args.seed}")
    device = resolve_device(cfg.env)

    if args.fresh:
        wipe_run_dirs(cfg)

    if args.stage == "wiring":
        wiring(cfg, device)
    else:
        try:
            full(cfg, device)
        finally:
            _flush_ledger()   # keep the GPU time when a session stops before completion


if __name__ == "__main__":
    main()
