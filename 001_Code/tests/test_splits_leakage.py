from data.splits import check_leakage


def test_no_leakage_clean():
    sft = [{"prompt": "s1"}]
    grpo = [{"prompt": "g1"}]
    ev = [{"prompt": "e1"}]
    assert check_leakage(sft, grpo, ev) == set()


def test_leakage_detected():
    sft = [{"prompt": "x"}]
    grpo = [{"prompt": "x"}]  # same prompt leaked into grpo
    ev = [{"prompt": "e1"}]
    assert "x" in check_leakage(sft, grpo, ev)


def test_leakage_normalises_case_and_ws():
    from data.splits import check_leakage
    sft = [{"prompt": "The Answer"}]
    grpo = [{"prompt": "the   answer"}]
    ev = [{"prompt": "other"}]
    assert check_leakage(sft, grpo, ev)  # normalised prompts collide
