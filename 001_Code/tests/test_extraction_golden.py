import pytest

from data.mmlu import extract_letter

CASES = [
    ("<think>x</think>\n<answer>D</answer>", "D"),
    ("<think>reasoning</think>\n<answer>The answer is B.</answer>", "B"),
    ("<think></think><answer>42</answer>", None),
    ("<think>x</think><answer>b</answer>", "B"),
    ("<think>hm</think><answer>A or maybe B</answer>", "B"),
    ("<think>the answer is C</think><answer>D</answer>", "D"),
    ("<think>x</think>\n<answer>B</answer> and then a trailing sentence with a word", "B"),
    ("The answer is B because a model like this is small.", "B"),
    ("I think the correct option is C. It is a good choice.", "C"),
    ("A dog walked past. The answer is clearly D", "D"),
    ("Answer: C", "C"),
    ("b", "B"),
    ("As a matter of fact, all options are wrong", None),
    ("the answer is a.", "A"),
    ("", None),
    ("The plan (B) seems best", "B"),
    ("Not A. Not B. It must be C.", "C"),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_extract_letter_golden(text, expected):
    assert extract_letter(text) == expected
