from eval.diagnostics import format_compliance_rate, mean_response_length


def test_compliance():
    texts = ["<think>a</think>\n<answer>B</answer>", "bad"]
    assert format_compliance_rate(texts) == 0.5


def test_length():
    assert mean_response_length([10, 20]) == 15
