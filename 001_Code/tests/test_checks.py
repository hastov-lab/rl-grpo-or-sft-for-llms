import pytest

from data.checks import find_overlap, assert_no_leakage


def test_no_overlap():
    assert find_overlap(["a", "b"], ["c", "d"]) == set()
    assert_no_leakage(["a"], ["b"])


def test_overlap_found():
    assert find_overlap(["a", "b"], ["b", "c"]) == {"b"}
    with pytest.raises(ValueError):
        assert_no_leakage(["a", "b"], ["b", "c"])


def test_norm_text():
    from data.checks import norm_text
    assert norm_text("  The   Answer ") == "the answer"


def test_near_duplicates_exact_match():
    from data.checks import near_duplicates
    assert near_duplicates(["x y z w v"], ["x y z w v"], 0.9) == [(0, 0, 1.0)]


def test_near_duplicates_below_threshold():
    from data.checks import near_duplicates
    assert near_duplicates(["x y z w v"], ["x y z w q"], 0.9) == []


def test_near_duplicates_lower_threshold():
    from data.checks import near_duplicates
    assert near_duplicates(["x y z w v"], ["x y z w q"], 0.6) == [(0, 0, 0.667)]
