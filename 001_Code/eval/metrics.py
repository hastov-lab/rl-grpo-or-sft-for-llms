# scoring for the held out mmlu test: exact match accuracy, per tier/subject, bootstrap ci
import numpy as np


def accuracy(preds, golds):
    if not preds:
        return 0.0
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    return correct / len(preds)


def _group(rows, key):
    by = {}
    for r in rows:
        by.setdefault(r[key], []).append(r)
    return by


def per_tier_accuracy(rows):
    return {t: accuracy([x["pred"] for x in rs], [x["gold"] for x in rs])
            for t, rs in _group(rows, "tier").items()}


def per_subject_accuracy(rows):
    return {s: accuracy([x["pred"] for x in rs], [x["gold"] for x in rs])
            for s, rs in _group(rows, "subject").items()}


def bootstrap_ci(preds, golds, n_boot=1000, alpha=0.05, seed=0):
    n = len(preds)
    if n == 0:
        return (0.0, 0.0)
    correct = np.array([1.0 if p == g else 0.0 for p, g in zip(preds, golds)])
    rng = np.random.default_rng(seed)
    stats = [correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def per_tier_ci(rows, n_boot=1000, seed=0):
    return {t: bootstrap_ci([x["pred"] for x in rs], [x["gold"] for x in rs], n_boot=n_boot, seed=seed)
            for t, rs in _group(rows, "tier").items()}


def grpo_delta(sft_acc, grpo_acc):
    # GRPO accuracy minus SFT accuracy within each tier
    tiers = set(sft_acc) | set(grpo_acc)
    return {t: grpo_acc.get(t, 0.0) - sft_acc.get(t, 0.0) for t in tiers}


def build_pred_row(output, gold, subject, tier, n_tokens, *, extractor=None):
    # One row per test item so any metric or error analysis can be redone offline.
    # Scoring happens with the opening <think> put back (the prompt supplies it),
    # the raw generated text is what gets stored
    from data.mmlu import extract_letter
    from data.tagging import is_format_ok, with_open_tag
    scored_text = with_open_tag(output)
    reader = extract_letter if extractor is None else extractor
    pred = reader(scored_text)
    return {
        "subject": subject, "tier": tier, "gold": gold, "pred": pred,
        "correct": pred == gold, "n_tokens": n_tokens,
        "format_ok": is_format_ok(scored_text), "output": output,
    }


def extraction_failure_rate(rows):
    # fraction of eval rows where no letter could be extracted
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["pred"] is None) / len(rows)
