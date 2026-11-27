from data.gsm8k import parse_answer, format_example
from data.prompts import GSM8K_INSTRUCTION


def test_parse_answer_splits_final():
    reasoning, final = parse_answer("step one\nstep two\n#### 42")
    assert final == "42"
    assert "step one" in reasoning


def test_format_example_wraps_tags():
    ex = {"question": "what is 2+2?", "answer": "add them up\n#### 4"}
    out = format_example(ex)
    assert GSM8K_INSTRUCTION in out["prompt"]
    assert out["prompt"].endswith("what is 2+2?\n<think>")
    assert out["target"] == "add them up</think>\n<answer>4</answer>"
