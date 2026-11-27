# statistics for the paired eval comparisons: mcnemar, paired bootstrap ci, holm
import json

import numpy as np
from scipy.stats import binom


def mcnemar_exact(n01, n10):
    # two sided exact test on the discordant pairs: p = 2 * P(X <= min) under binom(n, .5)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(2.0 * float(binom.cdf(k, n, 0.5)), 1.0)


def paired_delta_ci(correct_a, correct_b, n_boot=10000, alpha=0.05, seed=0):
    # Bootstrap ci for mean(b) - mean(a). Resample item indices jointly so the pairing stays
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired lists must be the same length")
    n = len(a)
    if n == 0:
        return (0.0, 0.0, 0.0)
    diffs = b - a
    delta = float(diffs.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diffs[idx].mean(axis=1)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return (delta, lo, hi)


def holm(pvals):
    # holm-bonferroni adjusted p values, kept in the input order, monotone and capped at 1
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def _read_preds(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_pred_pairs(path_a, path_b):
    # Read two aligned preds jsonl files and return their correct lists. Mismatch is an error
    ra, rb = _read_preds(path_a), _read_preds(path_b)
    if len(ra) != len(rb):
        raise ValueError("pred files have different lengths")
    for x, y in zip(ra, rb):
        if (x["subject"], x["gold"]) != (y["subject"], y["gold"]):
            raise ValueError("pred files are not aligned (subject/gold differ)")
    return [bool(x["correct"]) for x in ra], [bool(y["correct"]) for y in rb]


def align_pred_rows(rows_a, rows_b):
    # The tier swap on 15 jul changed the subject order in build_eval_split, so preds
    # written before and after it store the same items in a different row order.
    # Within a subject the order never changed (raw hf order), so regrouping both
    # files by subject and gluing the blocks back in one fixed subject order makes
    # them item-aligned again. Anything that does not line up exactly is an error
    def by_subject(rows):
        d = {}
        for r in rows:
            d.setdefault(r["subject"], []).append(r)
        return d

    da, db = by_subject(rows_a), by_subject(rows_b)
    if sorted(da) != sorted(db):
        raise ValueError("pred files cover different subjects")
    out_a, out_b = [], []
    for s in sorted(da):
        if len(da[s]) != len(db[s]):
            raise ValueError(f"pred files disagree on item count for {s}")
        for x, y in zip(da[s], db[s]):
            if x["gold"] != y["gold"]:
                raise ValueError(f"pred files are not alignable within {s} (gold differs)")
        out_a.extend(da[s])
        out_b.extend(db[s])
    return out_a, out_b


def load_pred_pairs_any_order(path_a, path_b):
    # Like load_pred_pairs but for files from different subject-order eras, for
    # example 007 preds against 005/006 preds. Realigns per subject first instead
    # of weakening the strict check
    ra, rb = align_pred_rows(_read_preds(path_a), _read_preds(path_b))
    return [bool(x["correct"]) for x in ra], [bool(y["correct"]) for y in rb]
