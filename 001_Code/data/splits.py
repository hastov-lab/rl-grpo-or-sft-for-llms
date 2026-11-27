# builds the three non overlapping splits and checks for leakage between them
# sft = gsm8k, grpo = mmlu dev+val, eval = mmlu test
from data.checks import find_overlap, norm_text
from data.mmlu import load_subject
from data.subjects import all_subjects, tier_of


def drop_overlap(rows, other_keys):
    # Kick out any row whose normalised prompt also appears in the other split.
    # MMLU has a few questions duplicated across its own splits, this catches them
    return [r for r in rows if norm_text(r["prompt"]) not in other_keys]


def build_grpo_split(limit_per_subject=None, dedup_against_eval=True):
    rows = []
    for subj in all_subjects():
        for split in ("dev", "validation"):
            exs = load_subject(subj, split)
            if limit_per_subject:
                exs = exs[:limit_per_subject]
            for ex in exs:
                ex["tier"] = tier_of(subj)
                rows.append(ex)
    if dedup_against_eval:
        eval_keys = {norm_text(r["prompt"]) for r in build_eval_split()}
        rows = drop_overlap(rows, eval_keys)
    return rows


def build_eval_split(limit_per_subject=None):
    rows = []
    for subj in all_subjects():
        exs = load_subject(subj, "test")
        if limit_per_subject:
            exs = exs[:limit_per_subject]
        for ex in exs:
            ex["tier"] = tier_of(subj)
            rows.append(ex)
    return rows


def check_leakage(sft_rows, grpo_rows, eval_rows):
    # normalise the prompts so case/whitespace differences cant hide an overlap
    sft_keys = [norm_text(r["prompt"]) for r in sft_rows]
    grpo_keys = [norm_text(r["prompt"]) for r in grpo_rows]
    eval_keys = [norm_text(r["prompt"]) for r in eval_rows]
    return find_overlap(sft_keys, grpo_keys, eval_keys)
