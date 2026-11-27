from data.mmlu import build_prompt, gold_letter, format_example, extract_letter
from data.prompts import MMLU_INSTRUCTION


def test_build_prompt():
    p = build_prompt("q?", ["aa", "bb", "cc", "dd"])
    assert MMLU_INSTRUCTION in p
    assert "A. aa" in p
    assert "D. dd" in p
    assert "Answer:" not in p
    assert p.endswith("\n<think>")


def test_gold_letter():
    assert gold_letter(0) == "A"
    assert gold_letter(3) == "D"


def test_format_example():
    ex = {"question": "q?", "choices": ["a", "b", "c", "d"], "answer": 2, "subject": "marketing"}
    out = format_example(ex)
    assert out["gold"] == "C"
    assert out["subject"] == "marketing"


def test_extract_round_trips_gold():
    # a well formatted answer should score against the gold letter
    ans = "<think>reasoning</think>\n<answer>C</answer>"
    assert extract_letter(ans) == "C"
