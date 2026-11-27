# leakage guard, runs before every training run
import re


def norm_text(s):
    # normalise so case/whitespace differences dont hide an overlap
    return re.sub(r"\s+", " ", s).strip().lower()


def find_overlap(*splits):
    # returns the items that show up in more than one split
    seen = {}
    overlap = set()
    for i, split in enumerate(splits):
        for item in set(split):
            if item in seen and seen[item] != i:
                overlap.add(item)
            seen[item] = i
    return overlap


def assert_no_leakage(*splits):
    bad = find_overlap(*splits)
    if bad:
        raise ValueError(f"leakage found: {len(bad)} shared items")


def near_duplicates(a_texts, b_texts, threshold=0.9):
    # jaccard on lowercased whitespace token sets, return the (i, j, score) pairs at/above threshold
    a_sets = [set(t.lower().split()) for t in a_texts]
    b_sets = [set(t.lower().split()) for t in b_texts]
    pairs = []
    for i, sa in enumerate(a_sets):
        for j, sb in enumerate(b_sets):
            union = sa | sb
            if not union:
                continue
            score = len(sa & sb) / len(union)
            if score >= threshold:
                pairs.append((i, j, round(score, 3)))
    return pairs
