import pytest

from data.mmlu import build_probe_prompt
from eval.harness import argmax_letter, letter_token_ids


def test_probe_prompt_exact():
    out = build_probe_prompt("Q?", ["w", "x", "y", "z"])
    assert out == "Q?\nA. w\nB. x\nC. y\nD. z\nAnswer:"


def test_argmax_letter():
    assert argmax_letter([0.1, 0.9, 0.2, 0.3]) == "B"


def test_argmax_letter_tie_goes_first():
    assert argmax_letter([1.0, 1.0, 0.0, 0.0]) == "A"


class _StubTok:
    # duck typed tokenizer, just enough for letter_token_ids
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, s, add_special_tokens=False):
        return [self.mapping[s]]


def test_letter_ids_distinct_ok():
    tok = _StubTok({" A": 1, " B": 2, " C": 3, " D": 4})
    assert letter_token_ids(tok) == [1, 2, 3, 4]


def test_letter_ids_collision_raises():
    tok = _StubTok({" A": 1, " B": 1, " C": 3, " D": 4})
    with pytest.raises(ValueError):
        letter_token_ids(tok)
