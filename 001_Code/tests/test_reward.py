from config import RewardConfig
from train.reward import total_reward


def test_correct_and_formatted():
    cfg = RewardConfig()
    text = "<think>reasoning</think>\n<answer>B</answer>"
    assert total_reward(text, "B", "B", cfg) == 1.2


def test_correct_but_no_format():
    cfg = RewardConfig()
    assert total_reward("the answer is B", "B", "B", cfg) == 1.0


def test_wrong_but_formatted():
    cfg = RewardConfig()
    text = "<think>x</think>\n<answer>A</answer>"
    assert total_reward(text, "A", "B", cfg) == 0.2


def test_wrong_and_no_format():
    cfg = RewardConfig()
    assert total_reward("nope", "A", "B", cfg) == 0.0
