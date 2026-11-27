from eval.metrics import accuracy, per_tier_accuracy, bootstrap_ci


def test_accuracy():
    assert accuracy(["A", "B", "C"], ["A", "X", "C"]) == 2 / 3


def test_per_tier():
    rows = [
        {"pred": "A", "gold": "A", "tier": "easy"},
        {"pred": "B", "gold": "C", "tier": "easy"},
        {"pred": "D", "gold": "D", "tier": "hard"},
    ]
    out = per_tier_accuracy(rows)
    assert out["easy"] == 0.5
    assert out["hard"] == 1.0


def test_bootstrap_ci_all_correct():
    preds = ["A"] * 10
    golds = ["A"] * 10
    lo, hi = bootstrap_ci(preds, golds, n_boot=200, seed=0)
    assert lo == 1.0 and hi == 1.0


def test_grpo_delta():
    from eval.metrics import grpo_delta
    d = grpo_delta({"easy": 0.5, "hard": 0.2}, {"easy": 0.6, "hard": 0.5})
    assert abs(d["easy"] - 0.1) < 1e-9
    assert abs(d["hard"] - 0.3) < 1e-9


def test_per_subject_accuracy():
    from eval.metrics import per_subject_accuracy
    rows = [
        {"pred": "A", "gold": "A", "subject": "marketing"},
        {"pred": "B", "gold": "C", "subject": "marketing"},
        {"pred": "D", "gold": "D", "subject": "formal_logic"},
    ]
    out = per_subject_accuracy(rows)
    assert out["marketing"] == 0.5
    assert out["formal_logic"] == 1.0


def test_per_tier_ci_all_correct():
    from eval.metrics import per_tier_ci
    rows = [{"pred": "A", "gold": "A", "tier": "easy"},
            {"pred": "A", "gold": "A", "tier": "easy"}]
    ci = per_tier_ci(rows, n_boot=100, seed=0)
    assert ci["easy"] == (1.0, 1.0)


def test_build_pred_row():
    from eval.metrics import build_pred_row
    row = build_pred_row("<think>x</think>\n<answer>B</answer>", "B", "marketing", "easy", 12)
    assert row["pred"] == "B"
    assert row["correct"] is True
    assert row["format_ok"] is True
    assert row["n_tokens"] == 12


def test_extraction_failure_rate():
    from eval.metrics import extraction_failure_rate
    rows = [{"pred": "A"}, {"pred": None}, {"pred": "C"}, {"pred": None}]
    assert extraction_failure_rate(rows) == 0.5
    assert extraction_failure_rate([]) == 0.0
