import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_batch_equivalence():
    # Tiny-GPT2, greedy: a batch of 1 vs a batch of 4 must give the same letters.
    # Only runs where torch and transformers are installed
    from config import RunConfig
    from enums import Mode
    from eval.harness import _generate_batch
    from data.mmlu import extract_letter
    from train.model_utils import load_model_and_tokenizer

    cfg = RunConfig()
    cfg.mode = Mode.SMOKE
    model, tok = load_model_and_tokenizer(cfg, "cpu")
    model.eval()
    prompts = [f"question {i}? A. a B. b C. c D. d" for i in range(6)]

    one = []
    for p in prompts:
        one += _generate_batch(model, tok, [p], "cpu", 16)
    four = []
    for i in range(0, 6, 4):
        four += _generate_batch(model, tok, prompts[i:i + 4], "cpu", 16)

    assert [extract_letter(t) for t in one] == [extract_letter(t) for t in four]
