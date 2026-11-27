# Materialise the three splits and assert no leakage. This is a local step (runs on
# a local machine, not colab). Colab is only for the heavy gpu runs. Use --limit for a smoke build
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.checks import norm_text
from data.gsm8k import load_sft
from data.splits import build_grpo_split, build_eval_split, check_leakage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap rows per subject for a smoke build")
    args = ap.parse_args()

    sft = load_sft(limit=args.limit)
    raw_grpo = build_grpo_split(limit_per_subject=args.limit, dedup_against_eval=False)
    grpo = build_grpo_split(limit_per_subject=args.limit)
    ev = build_eval_split(limit_per_subject=args.limit)

    # show what the dedup threw away, these are mmlu items duplicated into test
    ev_keys = {norm_text(r["prompt"]) for r in ev}
    dropped = [r for r in raw_grpo if norm_text(r["prompt"]) in ev_keys]
    if dropped:
        print(f"dropped {len(dropped)} grpo prompt(s) that also sit in the test split:")
        for r in dropped:
            print(f"  [{r['subject']}] {r['prompt'][:100]!r}...")

    leak = check_leakage(sft, grpo, ev)
    if leak:
        raise SystemExit(f"leakage found: {len(leak)} shared items, stopping")

    print(f"sft rows:  {len(sft)}")
    print(f"grpo rows: {len(grpo)}  by tier: {dict(Counter(r['tier'] for r in grpo))}")
    print(f"eval rows: {len(ev)}  by tier: {dict(Counter(r['tier'] for r in ev))}")
    print("no leakage, three splits ready")


if __name__ == "__main__":
    main()
