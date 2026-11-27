# One-off audit of the data splits, run locally like build_data. Writes a json
# artifact with the split counts,
# duplicates inside the test set, near duplicates between the grpo pool and test,
# the (subject, gold) signature that all preds files are checked against, and the
# current hub revision of both datasets. The near duplicate pass over the whole
# test set is quadratic, takes a minute or two, fine for a one-time local run
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.checks import norm_text, near_duplicates
from data.prompts import MMLU_INSTRUCTION
from data.splits import build_grpo_split, build_eval_split


def core_text(prompt):
    # question + options only, the shared instruction and the <think> suffix would
    # inflate every jaccard score since they appear in all prompts
    s = prompt
    if s.startswith(MMLU_INSTRUCTION):
        s = s[len(MMLU_INSTRUCTION):]
    return s.removesuffix("\n<think>").strip()


def hub_revision(repo_id):
    # Record which dataset revision the hub serves right now. Best effort, the
    # audit numbers themselves come from whatever the local cache loaded
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo_id).sha
    except Exception:
        return None


def main():
    test = build_eval_split()
    pool_raw = build_grpo_split(dedup_against_eval=False)
    pool = build_grpo_split()

    report = {"date": date.today().isoformat()}
    report["counts"] = {
        "test": len(test),
        "pool_raw": len(pool_raw),
        "pool_deduped": len(pool),
        "test_by_subject": dict(Counter(r["subject"] for r in test)),
        "pool_by_subject": dict(Counter(r["subject"] for r in pool)),
    }

    # exact duplicates inside the test split, grouped by normalised prompt
    seen = {}
    groups = []
    for i, r in enumerate(test):
        k = norm_text(r["prompt"])
        if k in seen:
            groups.append({"subject": r["subject"], "rows": [seen[k], i],
                           "golds": [test[seen[k]]["gold"], r["gold"]],
                           "question": core_text(r["prompt"])[:120]})
        else:
            seen[k] = i
    report["test_internal_exact_dups"] = groups

    # Near duplicates, instruction stripped. Pool vs test at two thresholds, and
    # inside the test set itself (self pairs filtered out)
    pool_q = [core_text(r["prompt"]) for r in pool]
    test_q = [core_text(r["prompt"]) for r in test]
    for th in (0.9, 0.8):
        pairs = near_duplicates(pool_q, test_q, threshold=th)
        report[f"pool_vs_test_near_dups_{th}"] = [
            {"pool_subject": pool[i]["subject"], "test_subject": test[j]["subject"],
             "score": s, "pool_q": pool_q[i][:100], "test_q": test_q[j][:100]}
            for i, j, s in pairs]
    within = [(i, j, s) for i, j, s in near_duplicates(test_q, test_q, threshold=0.9) if i < j]
    report["test_internal_near_dups_0.9"] = [
        {"subject_a": test[i]["subject"], "subject_b": test[j]["subject"],
         "rows": [i, j], "score": s} for i, j, s in within]

    # The ordering signature, same construction the preds cross-check uses. Note
    # this depends on the subject order in TIERS, so it changed at the tier swap
    sig = hashlib.sha256(json.dumps([(r["subject"], r["gold"]) for r in test]).encode()).hexdigest()
    report["test_subject_gold_sha256"] = sig
    report["gold_balance"] = dict(Counter(r["gold"] for r in test))
    report["hub_revisions"] = {"cais/mmlu": hub_revision("cais/mmlu"),
                               "openai/gsm8k": hub_revision("openai/gsm8k")}

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", f"split_audit_{date.today().strftime('%d%b%Y')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"test {len(test)}, pool {len(pool_raw)} -> {len(pool)} after dedup")
    print(f"exact dup pairs inside test: {len(report['test_internal_exact_dups'])}")
    print(f"pool vs test near dups: {len(report['pool_vs_test_near_dups_0.9'])} at 0.9, "
          f"{len(report['pool_vs_test_near_dups_0.8'])} at 0.8")
    print(f"near dup pairs inside test at 0.9: {len(report['test_internal_near_dups_0.9'])}")
    print(f"signature {sig[:10]}, saved to {out}")


if __name__ == "__main__":
    main()
