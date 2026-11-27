# Throwaway investigation for the aux-train memo. Run: python analysis/aux_train_check.py
# answers: does mmlu auxiliary_train carry per-subject labels, how big is the current grpo
# split, and is there any contamination with the eight mmlu test splits
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from data.subjects import all_subjects
from data.splits import build_grpo_split
from data.checks import norm_text, near_duplicates

SUBJECTS = all_subjects()


def q1_aux_labels():
    print("== Q1: auxiliary_train per-subject labels ==")
    ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    print("total aux_train rows:", len(ds))
    print("columns:", ds.column_names)
    if "subject" not in ds.column_names:
        # No subject column at all, so per-subject filtering is impossible. Thats the answer
        print("  no 'subject' column in auxiliary_train, per-subject counts are all 0 by definition")
        return ds
    subj = Counter(ds["subject"])
    for s in SUBJECTS:
        print(f"  {s}: {subj.get(s, 0)}")
    print("  rows with empty subject:", subj.get("", 0))
    print("  number of distinct subject values:", len(subj))
    return ds


def q2_grpo_counts():
    print("== Q2: current grpo split (dev+val) counts ==")
    rows = build_grpo_split()
    per_subj = Counter(r["subject"] for r in rows)
    per_tier = Counter(r["tier"] for r in rows)
    for s in SUBJECTS:
        print(f"  {s}: {per_subj.get(s, 0)}")
    print("  per tier:", dict(per_tier))
    print("  total grpo rows:", len(rows))


def _aux_questions(aux_ds, sample):
    # the cais/mmlu aux build nests each row under a 'train' field, handle both shapes
    take = aux_ds.select(range(min(sample, len(aux_ds))))
    if "question" in take.column_names:
        return list(take["question"])
    return [r["train"]["question"] for r in take]


def q3_contamination(aux_ds, sample=1500):
    print("== Q3: contamination between aux pool and the eight test splits ==")
    test_q = []
    for s in SUBJECTS:
        test_q += list(load_dataset("cais/mmlu", s, split="test")["question"])
    print("  test questions:", len(test_q))
    aux_q = _aux_questions(aux_ds, sample)
    print("  aux questions sampled:", len(aux_q))
    print("  exact overlap:", len(set(test_q) & set(aux_q)))
    tn = set(norm_text(q) for q in test_q)
    an = set(norm_text(q) for q in aux_q)
    print("  normalised overlap:", len(tn & an))
    nd = near_duplicates(aux_q, test_q, threshold=0.9)
    print("  near-duplicate pairs (jaccard >= 0.9):", len(nd))


if __name__ == "__main__":
    aux = q1_aux_labels()
    q2_grpo_counts()
    q3_contamination(aux)
