from data.mmlu import extract_letter


def test_from_answer_tag():
    assert extract_letter("<think>..</think>\n<answer>D</answer>") == "D"


def test_from_plain_text():
    assert extract_letter("i think the answer is B") == "B"


def test_none_when_missing():
    assert extract_letter("no letters here") is None
