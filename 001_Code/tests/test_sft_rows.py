from train.sft import build_sft_rows

ROWS = [{"prompt": "what is 2+2", "target": "<think>easy</think>\n<answer>4</answer>"}]


def test_text_mode_default():
    out = build_sft_rows(ROWS, "<eos>", completion_only=False)
    assert list(out[0].keys()) == ["text"]
    assert out[0]["text"].startswith("what is 2+2\n<think>")
    assert out[0]["text"].endswith("</answer><eos>")


def test_completion_only_mode():
    out = build_sft_rows(ROWS, "<eos>", completion_only=True)
    assert sorted(out[0].keys()) == ["completion", "prompt"]
    assert out[0]["prompt"].endswith("\n")
    assert out[0]["completion"].endswith("</answer><eos>")


def test_no_double_eos():
    rows = [{"prompt": "q", "target": "t<eos>"}]
    out = build_sft_rows(rows, "<eos>", completion_only=False)
    assert not out[0]["text"].endswith("<eos><eos>")


def test_mc_sft_rows_shape():
    from train.sft import mc_sft_rows
    grpo_rows = [{"prompt": "q1\n<think>", "gold": "B", "tier": "easy"},
                 {"prompt": "q2\n<think>", "gold": "D", "tier": "hard"}]
    out = mc_sft_rows(grpo_rows)
    assert out[0]["prompt"] == "q1\n<think>"
    assert out[0]["target"].endswith("<answer>B</answer>")
    assert "</think>" in out[0]["target"] and not out[0]["target"].startswith("<think>")
    # the fixed think passes the 20 char reward gate, checked once here on purpose
    think = out[0]["target"].split("</think>")[0]
    assert len(think.strip()) >= 20
    assert out[1]["target"].endswith("<answer>D</answer>")
