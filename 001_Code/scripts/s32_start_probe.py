# run the source-adapter probe in a disposable process
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint import save_metrics
from config import load_config
from scripts.env_switch import apply_smoke, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = apply_smoke(load_config(args.config))
    device = resolve_device(cfg.env)
    from eval.harness import run_logprob_eval
    metrics = run_logprob_eval(cfg, device, condition="source_adapter_gate", out_dir=args.out,
                               adapter_path=args.adapter)
    save_metrics(os.path.join(args.out, "source_adapter_gate_logprob.json"), metrics)
    print("source adapter gate per-tier:", metrics["per_tier_acc"])


if __name__ == "__main__":
    main()
