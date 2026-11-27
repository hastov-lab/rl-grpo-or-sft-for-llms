from data.tagging import wrap, parse, is_format_ok


def test_roundtrip():
    t = wrap("some thinking", "C")
    assert parse(t) == ("some thinking", "C")
    assert is_format_ok(t)


def test_missing_tag():
    assert parse("<answer>C</answer>") is None
    assert not is_format_ok("just text")


def test_format_ok_min_think():
    from data.tagging import is_format_ok
    empty = "<think></think>\n<answer>B</answer>"
    assert is_format_ok(empty) is True
    assert is_format_ok(empty, min_think_chars=5) is False
    full = "<think>because</think>\n<answer>B</answer>"
    assert is_format_ok(full, min_think_chars=5) is True


def test_with_open_tag_prepends_only_when_missing():
    from data.tagging import with_open_tag
    assert with_open_tag("reasoning</think>\n<answer>B</answer>") == "<think>reasoning</think>\n<answer>B</answer>"
    assert with_open_tag("<think>x</think><answer>A</answer>") == "<think>x</think><answer>A</answer>"


def test_tagless_completion_scores_end_to_end():
    # what a rollout looks like when the prompt already opened the think block
    from config import RewardConfig
    from train.reward import make_reward_funcs
    acc, fmt = make_reward_funcs(RewardConfig())
    comp = ["some proper reasoning happens here first</think>\n<answer>B</answer>"]
    assert acc(completions=comp, gold=["B"]) == [1.0]
    assert fmt(completions=comp) == [0.2]


def test_tagless_completion_in_pred_row():
    from eval.metrics import build_pred_row
    row = build_pred_row("thinking about it a little bit</think>\n<answer>C</answer>",
                         "C", "marketing", "easy", 20)
    assert row["pred"] == "C" and row["correct"] and row["format_ok"]
    assert not row["output"].startswith("<think>")  # raw text stays raw
