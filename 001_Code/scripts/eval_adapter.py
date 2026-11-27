# evaluate a saved adapter on the frozen MMLU test set
import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from scripts.env_switch import resolve_device, apply_smoke


def ledger_path_for(out_dir):
    # out dirs look like <root>/results/<tag>, the ledger sits next to the tags
    return os.path.join(os.path.dirname(out_dir.rstrip("/")), "compute_ledger.jsonl")


def run_adapter_eval(cfg, adapter, condition, out, gen_eval, probe_eval, append_row, extra_manifest=None):
    # run generative and probe evaluation with resumable billing
    from checkpoint import save_metrics, load_metrics

    timings = {}
    row = {"run_tag": os.path.basename(out.rstrip("/")), "condition": "adapter_eval",
           "timings_sec": timings, "train_sec": 0, "train_steps": 0,
           "eval_items": {}, "completed": False}
    ev = lp = None

    def timed(key, fn):
        # keep the active stage time even if colab interrupts before it returns
        t0 = time.time()
        try:
            return fn()
        finally:
            timings[key] = round(time.time() - t0, 3)

    try:
        ev_path = os.path.join(out, f"{condition}_eval.json")
        if os.path.exists(ev_path):
            print(f"{condition} gen eval already done, skipping")
            ev = load_metrics(ev_path)
        else:
            ev = timed(f"{condition}_eval", gen_eval)
            save_metrics(ev_path, ev)

        lp_path = os.path.join(out, f"{condition}_logprob.json")
        if os.path.exists(lp_path):
            print(f"{condition} probe already done, skipping")
            lp = load_metrics(lp_path)
        else:
            lp = timed(f"{condition}_logprob_eval", probe_eval)
            save_metrics(lp_path, lp)

        # complete the fields deliberately left null by the earlier probe run
        sm_path = os.path.join(out, "summary.json")
        if os.path.exists(sm_path):
            sm = load_metrics(sm_path)
            sm[f"{condition}_acc"] = ev["per_tier_acc"]
            sm[f"{condition}_logprob_acc"] = lp["per_tier_acc"]
            save_metrics(sm_path, sm)
            print("summary completed with the new evals")

        row["eval_items"][condition] = ev.get("n_items")
        row["completed"] = True
        return ev, lp
    finally:
        # Bill whatever actually ran, even when the session dies mid eval. A run
        # where everything skipped computed nothing and writes no row
        if timings:
            append_row(row)
        manifest_path = os.path.join(out, f"{condition}_eval_manifest.json")
        old_incomplete = False
        if os.path.exists(manifest_path):
            old_incomplete = not load_metrics(manifest_path).get("completed", False)
        if timings or not os.path.exists(manifest_path) or (row["completed"] and old_incomplete):
            man = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "adapter": adapter, "condition": condition,
                   "timings_sec": dict(timings), "completed": row["completed"]}
            man.update(extra_manifest or {})
            save_metrics(manifest_path, man)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--condition", default="sft")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="override train.seed so manifests match the checkpoint being evaluated")
    args = ap.parse_args()

    cfg = apply_smoke(load_config(args.config))
    if args.seed is not None:
        cfg.train.seed = args.seed
    cfg.eval.skip_generative = False   # the whole point here is the generative eval
    device = resolve_device(cfg.env)

    from eval.harness import run_eval, run_logprob_eval
    from obs import append_ledger, gpu_snapshot, _git_info, _versions
    from train.model_utils import resolve_dtype

    ev, lp = run_adapter_eval(
        cfg, args.adapter, args.condition, args.out,
        gen_eval=lambda: run_eval(cfg, device, adapter_path=args.adapter,
                                  condition=args.condition, out_dir=args.out),
        probe_eval=lambda: run_logprob_eval(cfg, device, adapter_path=args.adapter,
                                            condition=args.condition, out_dir=args.out),
        append_row=lambda row: append_ledger(ledger_path_for(args.out), {**row, "gpu": gpu_snapshot()}),
        extra_manifest={"config": args.config, "seed_override": args.seed,
                        "git": _git_info(), "versions": _versions(),
                        "resolved_dtype": resolve_dtype(cfg), "gpu": gpu_snapshot()},
    )
    print("gen per-tier:", ev["per_tier_acc"])
    print("probe per-tier:", lp["per_tier_acc"])


if __name__ == "__main__":
    main()
