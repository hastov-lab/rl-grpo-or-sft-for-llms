from data.splits import drop_overlap
from data.checks import norm_text


def test_drop_overlap_removes_duplicates():
    rows = [{"prompt": "What is X?\nA. 1"}, {"prompt": "keep me\nA. 2"}]
    eval_keys = {norm_text("what is  x?\na. 1")}   # same item, different case/spacing
    kept = drop_overlap(rows, eval_keys)
    assert len(kept) == 1
    assert kept[0]["prompt"].startswith("keep me")


def test_drop_overlap_keeps_all_when_clean():
    rows = [{"prompt": "a"}, {"prompt": "b"}]
    assert drop_overlap(rows, {norm_text("c")}) == rows
