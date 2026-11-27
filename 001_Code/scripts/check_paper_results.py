"""Reproduce Qwen2.5 accuracy comparisons from the saved MMLU predictions."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

RESULTS = Path(__file__).resolve().parents[1] / "results"
HARD_COUNTS = {"high_school_mathematics": 270, "formal_logic": 126}
N_HARD = sum(HARD_COUNTS.values())
TEST_COUNTS = {
    **HARD_COUNTS, "high_school_psychology": 545, "marketing": 234,
    "us_foreign_policy": 100, "high_school_geography": 198,
    "philosophy": 311, "business_ethics": 100,
}
N_TEST = sum(TEST_COUNTS.values())
BELOW_7B = [
    ("0.5B", "B05", "006", "007_p8", 10.6),
    ("1.5B", "B15", "009", "010_p8", 3.5),
    ("3B", "B3", "003", "007_p8", 0.8),
]
SEED_SPREAD = [
    ("0.5B", "B05", "006", [871, 594, 953], 19.1),
    ("1.5B", "B15", "009", [1270, 1196, 1248], 3.9),
    ("3B", "B3", "003", [1349, 1369, 1370], 1.1),
    ("7B", "B7", "013", [1511, 1517, 1520], 0.5),
]
# Reference counts by seed: Warm and Cold overall, then Warm and Cold on the hard tier.
WARM_START = [
    ("1.5B", "B15", "009", "012", [
        (1270, 1147, 177, 166), (1196, 1153, 157, 161), (1248, 1192, 168, 181),
    ]),
    ("3B", "B3", "003", "004", [
        (1349, 1275, 206, 215), (1369, 1283, 218, 221), (1370, 1270, 205, 213),
    ]),
]
SUBJECT_REFERENCES = {
    "high_school_mathematics": ("High-school mathematics", 556, 448),
    "formal_logic": ("Formal logic", 212, 213),
}

# Run prefixes are in seed order. Run 032 repeated warm p8 with matched starts.
SUPERVISED = [
    ("Plain p2", ("025_p2",) * 3, 11.027),
    ("Plain p8", ("014_p8",) * 3, 9.512),
    ("Plain p16", ("024_p16",) * 3, 9.428),
    ("Plain p32", ("025_p32",) * 3, 9.428),
    ("Warm p8", ("015_wsft", "032_wsft", "032_wsft"), 11.027),
    ("Warm p16", ("024_wsft_p16",) * 3, 10.438),
]


def answer_matches(row, path, line_number):
    gold, pred = row["gold"], row["pred"]
    if gold not in ("A", "B", "C", "D"):
        raise ValueError(f"{path}:{line_number}: invalid gold letter {gold!r}")
    if pred not in ("A", "B", "C", "D", None):
        raise ValueError(f"{path}:{line_number}: invalid saved prediction")
    return pred == gold


def hard_score(path, subject_totals=None):
    labels = []
    subjects = Counter()
    correct = Counter()
    with path.open(encoding="utf-8") as predictions:
        for line_number, line in enumerate(predictions, 1):
            row = json.loads(line)
            subject, tier, gold = row["subject"], row["tier"], row["gold"]
            labels.append((subject, tier, gold))
            if (subject in HARD_COUNTS) != (tier == "hard"):
                raise ValueError(f"{path}:{line_number}: unexpected tier for {subject}")
            if tier == "hard":
                subjects[subject] += 1
                correct[subject] += answer_matches(row, path, line_number)

    if len(labels) != N_TEST or subjects != HARD_COUNTS:
        raise ValueError(f"{path}: unexpected test set ({len(labels)} rows, hard: {subjects})")
    if subject_totals is not None:
        subject_totals.update(correct)
    return sum(correct.values()), labels


def print_subject_results(warm, supervised):
    print("\n7B subjects: mean generated accuracy across seeds 0, 1, 2 (p = supervised passes)")
    print("Subject | Warm GRPO | Best MC-SFT | MC-SFT accuracy | GRPO difference (pp)")
    for subject, (label, expected_warm, expected_best) in SUBJECT_REFERENCES.items():
        # Each seed has the same questions, so total correct ranks the three-seed means exactly.
        total = 3 * HARD_COUNTS[subject]
        best = max(counts[subject] for counts in supervised.values())
        winners = [name for name, counts in supervised.items() if counts[subject] == best]
        hits = warm[subject]
        gain = 100 * (hits - best) / total
        print(f"{label} | {100 * hits / total:.2f}% ({hits}/{total}) | {', '.join(winners)} | "
              f"{100 * best / total:.2f}% ({best}/{total}) | {gain:+.2f}")
        if (hits, best) != (expected_warm, expected_best):
            raise SystemExit(f"{label}: correct counts differ from the reference counts "
                             f"({expected_warm}, {expected_best})")


def endpoint_scores(path):
    subjects = Counter()
    correct = Counter()
    rows = []
    with path.open(encoding="utf-8") as predictions:
        for line_number, line in enumerate(predictions, 1):
            row = json.loads(line)
            subject, tier = row["subject"], row["tier"]
            if (subject in HARD_COUNTS) != (tier == "hard"):
                raise ValueError(f"{path}:{line_number}: unexpected tier for {subject}")
            rows.append((subject, tier, row["gold"]))
            subjects[subject] += 1
            correct[subject] += answer_matches(row, path, line_number)
    if subjects != TEST_COUNTS:
        raise ValueError(f"{path}: unexpected subject counts in the test set")
    return sum(correct.values()), sum(correct[s] for s in HARD_COUNTS), rows


def overall_score(path):
    # Two early runs group the subjects differently. Compare gold letters in order within each subject.
    correct, _, rows = endpoint_scores(path)
    labels = {}
    for subject, _, gold in rows:
        labels.setdefault(subject, []).append(gold)
    return correct, labels


def print_seed_spread():
    print(f"Warm GRPO seed spread: overall generated accuracy on {N_TEST:,} questions")
    print("Model | Run | Correct (seeds 0/1/2) | Accuracy % (seeds 0/1/2) | Max minus min (pp) | Paper (pp)")
    reference_labels = None
    for size, tag, prefix, expected_counts, expected_spread in SEED_SPREAD:
        counts = []
        for seed in range(3):
            run = f"{prefix}_COLAB_FULL_{tag}_sft_grpo_seed{seed}"
            correct, labels = overall_score(RESULTS / run / "grpo_preds.jsonl")
            if reference_labels is None:
                reference_labels = labels
            elif labels != reference_labels:
                raise ValueError(f"{run}: gold letters differ within a subject")
            counts.append(correct)

        spread = 100 * (max(counts) - min(counts)) / N_TEST
        if counts != expected_counts or round(spread, 1) != expected_spread:
            raise SystemExit(f"{size}: seed counts {counts} or spread {spread:.1f} differ from "
                             f"the references {expected_counts} and {expected_spread:.1f}")
        accuracies = ", ".join(f"{100 * count / N_TEST:.2f}" for count in counts)
        print(f"{size} | {prefix} | {counts} | {accuracies} | {spread:.1f} | {expected_spread:.1f}")


def print_warm_start():
    print(f"Warm minus Cold GRPO: {N_TEST:,} test questions per seed, including {N_HARD} hard questions")
    print("Model | Seed | Warm/Cold runs | Overall correct (Warm/Cold) | Difference (pp) | "
          "Hard correct (Warm/Cold) | Difference (pp)")
    for size, tag, warm_prefix, cold_prefix, references in WARM_START:
        for seed, expected in enumerate(references):
            warm_run = f"{warm_prefix}_COLAB_FULL_{tag}_sft_grpo_seed{seed}"
            cold_run = f"{cold_prefix}_COLAB_FULL_{tag}_base_grpo_seed{seed}"
            warm, warm_hard, warm_rows = endpoint_scores(RESULTS / warm_run / "grpo_preds.jsonl")
            cold, cold_hard, cold_rows = endpoint_scores(RESULTS / cold_run / "grpo_preds.jsonl")
            if warm_rows != cold_rows:
                raise ValueError(f"{size} seed {seed}: test subjects, tiers or gold letters differ between runs")
            if (warm, cold, warm_hard, cold_hard) != expected:
                raise SystemExit(f"{size} seed {seed}: Warm/Cold correct counts differ from {expected}")
            overall_gain = 100 * (warm - cold) / N_TEST
            hard_gain = 100 * (warm_hard - cold_hard) / N_HARD
            print(f"{size} | {seed} | {warm_prefix}/{cold_prefix} | {warm}/{cold} | "
                  f"{overall_gain:+.2f} | {warm_hard}/{cold_hard} | {hard_gain:+.2f}")


def print_below_7b():
    print(f"Overall generated accuracy below 7B: {N_TEST:,} questions per seed (0, 1, 2)")
    print("Model | Warm GRPO (run, correct, mean) | Plain MC-SFT, 8 passes (run, correct, mean) | SFT lead (pp) | Paper (pp)")
    reference_labels = None
    for size, tag, warm_prefix, plain_prefix, expected in BELOW_7B:
        warm_counts, plain_counts = [], []
        for seed in range(3):
            runs = (
                (f"{warm_prefix}_COLAB_FULL_{tag}_sft_grpo_seed{seed}", "grpo_preds.jsonl", warm_counts),
                (f"{plain_prefix}_COLAB_FULL_{tag}_sft_only_seed{seed}", "sft_preds.jsonl", plain_counts),
            )
            for run, filename, counts in runs:
                correct, labels = overall_score(RESULTS / run / filename)
                if reference_labels is None:
                    reference_labels = labels
                elif labels != reference_labels:
                    raise ValueError(f"{run}: gold letters differ within a subject")
                counts.append(correct)

        warm_accuracy = 100 * mean(warm_counts) / N_TEST
        plain_accuracy = 100 * mean(plain_counts) / N_TEST
        lead = 100 * (sum(plain_counts) - sum(warm_counts)) / (3 * N_TEST)
        print(f"{size} | {warm_prefix} {warm_counts} {warm_accuracy:.2f}% | "
              f"{plain_prefix} {plain_counts} {plain_accuracy:.2f}% | {lead:+.1f} | {expected:+.1f}")
        if round(lead, 1) != expected:
            raise SystemExit(f"{size}: calculated SFT lead differs from the reference value")


def print_7b_results(show_subjects):
    warm_counts = []
    warm_subjects = Counter()
    supervised_subjects = {}
    reference_labels = None
    for seed in range(3):
        run = f"013_COLAB_FULL_B7_sft_grpo_seed{seed}"
        count, labels = hard_score(RESULTS / run / "grpo_preds.jsonl", warm_subjects)
        if reference_labels is None:
            reference_labels = labels
        elif labels != reference_labels:
            raise ValueError(f"{run}: test subjects, tiers or gold letters differ across seeds")
        warm_counts.append(count)

    print(f"7B hard tier: {N_HARD} questions per seed (seeds 0, 1, 2)")
    print(f"Warm GRPO (013): {warm_counts} correct, mean accuracy {100 * mean(warm_counts) / N_HARD:.2f}%")
    print("MC-SFT      Run prefixes (seeds 0/1/2)            Correct (0/1/2)     Gain (pp)   Reference (pp)")
    gains = []
    for variant, prefixes, reference in SUPERVISED:
        sft_counts = []
        supervised_subjects[variant] = Counter()
        for seed, prefix in enumerate(prefixes):
            run = f"{prefix}_COLAB_FULL_B7_sft_only_seed{seed}"
            count, labels = hard_score(RESULTS / run / "sft_preds.jsonl", supervised_subjects[variant])
            if labels != reference_labels:
                raise ValueError(f"{run}: test subjects, tiers or gold letters differ from Warm GRPO")
            sft_counts.append(count)

        # Equal tier sizes in every run allow the paired gains to be averaged in counts.
        gain = 100 * (mean(warm_counts) - mean(sft_counts)) / N_HARD
        print(f"{variant:<11} {'/'.join(prefixes):<37} {str(sft_counts):<19} {gain:>9.3f} {reference:>16.3f}")
        if round(gain, 3) != reference:
            raise SystemExit(f"{variant}: calculated gain differs from the reference value")
        gains.append(gain)

    print(f"Warm GRPO gain over MC-SFT: {min(gains):.1f} to {max(gains):.1f} percentage points")
    if show_subjects:
        print_subject_results(warm_subjects, supervised_subjects)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", action="store_true",
                        help="also compare each 7B hard subject with the best MC-SFT mean")
    parser.add_argument("--below-7b", action="store_true",
                        help="compare overall Plain MC-SFT at eight passes with Warm GRPO below 7B")
    parser.add_argument("--warm-start", action="store_true",
                        help="compare Warm and Cold GRPO at 1.5B and 3B for each seed")
    parser.add_argument("--seed-spread", action="store_true",
                        help="show the observed range of Warm GRPO endpoint accuracies over three seeds")
    args = parser.parse_args()
    if args.subjects or not (args.below_7b or args.warm_start or args.seed_spread):
        print_7b_results(args.subjects)
    if args.below_7b:
        print_below_7b()
    if args.warm_start:
        print_warm_start()
    if args.seed_spread:
        print_seed_spread()


if __name__ == "__main__":
    main()
