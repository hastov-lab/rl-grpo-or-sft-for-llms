import random
from types import SimpleNamespace

import pytest

from config import RewardConfig, RunConfig
from data.mmlu import extract_letter as historical
from data.mmlu_b1 import extract_letter as b1
from data.tagging import with_open_tag
from eval.metrics import build_pred_row
from train.reward import make_reward_funcs


@pytest.mark.parametrize("letter", list("ABCDabcd"))
@pytest.mark.parametrize("punctuation", [".", ",", ")", "]", ":", ";", "!", "?", ""])
def test_standalone_answer_letters(letter, punctuation):
    assert b1(with_open_tag(f"The answer is {letter}{punctuation}")) == letter.upper()


@pytest.mark.parametrize("letter", list("ABCDabcd"))
@pytest.mark.parametrize("neighbour", ["x", "5", "_", "é", "\u0416", "中"])
def test_letters_inside_words_are_not_answers(letter, neighbour):
    assert b1(with_open_tag(f"answer {neighbour}{letter}.")) is None
    assert b1(with_open_tag(f"answer {letter}{neighbour}.")) is None


@pytest.mark.parametrize("letter", list("ABCDabcd"))
def test_complete_answer_tags(letter):
    text = f"reasoning</think><answer> ({letter}) </answer>"
    assert b1(with_open_tag(text)) == letter.upper()


@pytest.mark.parametrize("text,old,new", [
    ("The answer is unspecified.", "D", None),
    ("answer choices are listed.", "D", None),
    ("answer: b.", "B", "B"),
    ("final_answer: b.", "B", None),
    ("answer_choice: b.", "B", None),
    ("answer: éb.", "B", None),
    ("answer: 2b.", "B", None),
    ("not the answer: B.", "B", "B"),
    ("answer: B or C.", "B", "B"),
    ("done</think><answer>B or C</answer>", "C", "C"),
    ("done</think><answer>not B</answer>", "B", "B"),
    ("done</think><answer>E</answer>", None, None),
    ("", None, None),
    ("b", None, None),
    ("B", "B", "B"),
])
def test_historical_differences_and_shared_limits(text, old, new):
    # these pin a structural reader, not real language understanding
    assert historical(with_open_tag(text)) == old
    assert b1(with_open_tag(text)) == new


def test_keyword_window_does_not_cut_a_word():
    text = "answer" + " " * 29 + "choice"
    assert historical(with_open_tag(text)) == "C"
    assert b1(with_open_tag(text)) is None
    assert b1(with_open_tag("answer" + " " * 29 + "b.")) == "B"
    assert b1(with_open_tag("answer" + " " * 30 + "b.")) is None


def test_whole_answer_does_not_delete_unicode_or_underscore():
    for text in ("_b", "éb", "b中", "b_", "2b"):
        assert b1(f"<think>done</think><answer>{text}</answer>") is None


def test_reward_and_evaluation_opt_in_leaves_historical_defaults():
    text = "The answer is unspecified."
    old_acc, old_fmt = make_reward_funcs(RewardConfig())
    new_acc, new_fmt = make_reward_funcs(RewardConfig(), extractor=b1)
    for reader, expected in ((None, "D"), (b1, None), (None, "D")):
        assert build_pred_row(text, "D", "marketing", "easy", 7, extractor=reader)["pred"] == expected
    assert old_acc([text], gold=["D"]) == [1.0]
    assert new_acc([text], gold=["D"]) == [0.0]
    assert old_acc([text], gold=["D"]) == [1.0]
    assert old_fmt([text]) == new_fmt([text])
    assert [f.__name__ for f in (new_acc, new_fmt)] == ["accuracy_reward", "format_reward"]


def test_observer_changes_neither_rewards_nor_rng(tmp_path):
    import numpy as np
    import torch
    texts = ["answer: b.", "The answer is unspecified.",
             "long enough reasoning here</think><answer>B</answer>"]
    gold = ["B", "D", "B"]
    from train.parser_diagnostics import ParserDiagnostics
    diagnostic = ParserDiagnostics(str(tmp_path), str(tmp_path / "cp"), "test", 2, 3)
    observed = []
    before = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    bare = make_reward_funcs(RewardConfig(), extractor=b1)
    watched = make_reward_funcs(RewardConfig(), extractor=b1,
                                observer=diagnostic.observe)
    assert bare[0](texts, gold=gold) == watched[0](texts, gold=gold,
        trainer_state=SimpleNamespace(global_step=0), log_metric=lambda key, value: observed.append((key, value)))
    assert bare[1](texts) == watched[1](texts)
    assert observed == [("parser/responses_checked", 3), ("parser/h_only_accuracy_reward", 1),
                        ("parser/b1_only_accuracy_reward", 0)]
    diagnostic.on_log(None, SimpleNamespace(global_step=1), None, logs={"loss": 1.0})
    assert before[0] == random.getstate()
    assert np.array_equal(before[1][1], np.random.get_state()[1])
    assert before[1][2:] == np.random.get_state()[2:]
    assert torch.equal(before[2], torch.random.get_rng_state())


def test_actual_generated_evaluator_uses_requested_reader(monkeypatch, tmp_path):
    import json
    import eval.harness as harness
    model = SimpleNamespace(eval=lambda: None)
    tokenizer = SimpleNamespace(tokenize=lambda text: text.split())
    monkeypatch.setattr(harness, "load_model_and_tokenizer", lambda *a, **k: (model, tokenizer))
    monkeypatch.setattr(harness, "_generate_resilient", lambda *a, **k: ["The answer is unspecified."])
    rows = [{"prompt": "question", "gold": "D", "subject": "marketing", "tier": "easy"}]    
    # only model generation is replaced; scoring runs through the real evaluator.
    for condition, reader, expected in (("old", None, "D"), ("new", b1, None)):
        harness.run_eval_rows(RunConfig(), "cpu", rows, condition=condition,
                              out_dir=str(tmp_path), extractor=reader)
        row = json.loads((tmp_path / f"{condition}_preds.jsonl").read_text())
        assert row["pred"] == expected
