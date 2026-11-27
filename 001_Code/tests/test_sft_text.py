from train.sft import to_sft_text


def test_to_sft_text():
    ex = {"prompt": "what is 2+2?", "target": "<think>add</think>\n<answer>4</answer>"}
    out = to_sft_text(ex)
    assert out.startswith("what is 2+2?")
    assert "<answer>4</answer>" in out
