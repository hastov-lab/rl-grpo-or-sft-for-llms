from config import RewardConfig
from train.reward import make_reward_funcs

REAL_THINK = "<think>this one takes some actual reasoning first</think>\n<answer>B</answer>"


def test_reward_funcs_score_correctly():
    acc, fmt = make_reward_funcs(RewardConfig())
    comps = [REAL_THINK, "the answer is A", "junk"]
    gold = ["B", "B", "C"]
    assert acc(completions=comps, gold=gold) == [1.0, 0.0, 0.0]
    assert fmt(completions=comps) == [0.2, 0.0, 0.0]


def test_empty_think_earns_no_format_reward():
    # Protocol decision of 10 jul: min_think_chars 20, so a token think block
    # cant collect the format reward. Accuracy is unaffected
    acc, fmt = make_reward_funcs(RewardConfig())
    comps = ["<think>x</think>\n<answer>B</answer>"]
    assert fmt(completions=comps) == [0.0]
    assert acc(completions=comps, gold=["B"]) == [1.0]


def test_gate_off_when_zero():
    _, fmt = make_reward_funcs(RewardConfig(min_think_chars=0))
    assert fmt(completions=["<think>x</think>\n<answer>B</answer>"]) == [0.2]
