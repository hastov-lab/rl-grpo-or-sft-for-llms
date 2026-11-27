import json

import pytest

from analysis.stats import (mcnemar_exact, paired_delta_ci, holm, load_pred_pairs,
                            align_pred_rows, load_pred_pairs_any_order)


def test_mcnemar_values():
    assert round(mcnemar_exact(10, 2), 4) == 0.0386
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0


def test_paired_delta_ci_identical():
    assert paired_delta_ci([1, 0, 1], [1, 0, 1]) == (0.0, 0.0, 0.0)


def test_paired_delta_ci_length_mismatch():
    with pytest.raises(ValueError):
        paired_delta_ci([1, 0], [1, 0, 1])


def test_holm_values():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_load_pred_pairs_roundtrip(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    rows_a = [{"subject": "marketing", "gold": "A", "correct": True},
              {"subject": "philosophy", "gold": "C", "correct": False}]
    rows_b = [{"subject": "marketing", "gold": "A", "correct": False},
              {"subject": "philosophy", "gold": "C", "correct": True}]
    a.write_text("\n".join(json.dumps(r) for r in rows_a))
    b.write_text("\n".join(json.dumps(r) for r in rows_b))
    ca, cb = load_pred_pairs(str(a), str(b))
    assert ca == [True, False]
    assert cb == [False, True]


def test_load_pred_pairs_mismatch(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps({"subject": "marketing", "gold": "A", "correct": True}))
    b.write_text(json.dumps({"subject": "philosophy", "gold": "A", "correct": True}))
    with pytest.raises(ValueError):
        load_pred_pairs(str(a), str(b))


# old-order file has usfp after philosophy, new-order file has it after marketing,
# same shape as real pre and post tier-swap preds
OLD_ORDER = [{"subject": "marketing", "gold": "A", "correct": True},
             {"subject": "philosophy", "gold": "B", "correct": False},
             {"subject": "us_foreign_policy", "gold": "C", "correct": True},
             {"subject": "us_foreign_policy", "gold": "D", "correct": False}]
NEW_ORDER = [{"subject": "marketing", "gold": "A", "correct": False},
             {"subject": "us_foreign_policy", "gold": "C", "correct": False},
             {"subject": "us_foreign_policy", "gold": "D", "correct": True},
             {"subject": "philosophy", "gold": "B", "correct": True}]


def test_align_pred_rows_reorders_subject_blocks():
    ra, rb = align_pred_rows(OLD_ORDER, NEW_ORDER)
    # after alignment both sides carry the items in the same order
    assert [(r["subject"], r["gold"]) for r in ra] == [(r["subject"], r["gold"]) for r in rb]
    # order stays as stored
    usfp = [r["gold"] for r in ra if r["subject"] == "us_foreign_policy"]
    assert usfp == ["C", "D"]


def test_align_pred_rows_rejects_gold_mismatch():
    bad = [dict(r) for r in NEW_ORDER]
    bad[1]["gold"] = "A"
    with pytest.raises(ValueError):
        align_pred_rows(OLD_ORDER, bad)


def test_align_pred_rows_rejects_missing_subject():
    with pytest.raises(ValueError):
        align_pred_rows(OLD_ORDER, NEW_ORDER[:1])


def test_load_pred_pairs_any_order_cross_era(tmp_path):
    a = tmp_path / "old.jsonl"
    b = tmp_path / "new.jsonl"
    a.write_text("\n".join(json.dumps(r) for r in OLD_ORDER))
    b.write_text("\n".join(json.dumps(r) for r in NEW_ORDER))
    # the strict loader must refuse these, that check stays as is
    with pytest.raises(ValueError):
        load_pred_pairs(str(a), str(b))
    ca, cb = load_pred_pairs_any_order(str(a), str(b))
    assert ca == [True, False, True, False]
    assert cb == [False, True, False, True]
