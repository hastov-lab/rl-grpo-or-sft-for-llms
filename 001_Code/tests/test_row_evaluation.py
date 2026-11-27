# The row-driven evaluators keep the historical prediction schema and only add
# fact identity and letter scores when a caller asks for them.
import json
import sys
import types
from types import SimpleNamespace

import pytest

from config import RunConfig
from enums import Mode


class FakeEncoding(dict):
    def to(self, device):
        return self


class FakeScalar(float):
    def item(self):
        return float(self)


class FakeLastLogits:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


class FakeLogits:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        assert key == (slice(None), -1, slice(None))
        return FakeLastLogits(self.rows)


class FakeTokenizer:
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, prompts, return_tensors=None, padding=None):
        return FakeEncoding(input_ids=[[1] for _ in prompts])

    def encode(self, text, add_special_tokens=False):
        return [{" A": 1, " B": 2, " C": 3, " D": 4}[text]]

    def tokenize(self, text):
        return text.split()


class FakeModel:
    def __init__(self, scores):
        self.scores = scores

    def eval(self):
        return self

    def __call__(self, **kwargs):
        rows = []
        for scores in self.scores:
            row = [FakeScalar(0.0) for _ in range(5)]
            for token_id, score in enumerate(scores, start=1):
                row[token_id] = FakeScalar(score)
            rows.append(row)
        return SimpleNamespace(logits=FakeLogits(rows))


class NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


ROWS = [
    {"prompt": "q1\n<think>", "probe": "q1\nAnswer:", "gold": "B",
     "subject": "currency", "tier": "synthetic", "fact_id": "f1",
     "relation": "currency", "group": "injected", "variant": "canonical",
     "choice_order": ["a", "b", "c", "d"]},
    {"prompt": "q2\n<think>", "probe": "q2\nAnswer:", "gold": "D",
     "subject": "capital", "tier": "synthetic", "fact_id": "f2",
     "relation": "capital", "group": "control", "variant": "canonical",
     "choice_order": ["w", "x", "y", "z"]},
]


def test_logprob_rows_keep_identity_and_letter_scores(monkeypatch, tmp_path):
    import eval.harness as harness

    torch = types.ModuleType("torch")
    torch.no_grad = NoGrad
    monkeypatch.setitem(sys.modules, "torch", torch)
    model = FakeModel([[0.1, 0.8, 0.2, 0.3], [0.1, 0.2, 0.3, 0.9]])
    monkeypatch.setattr(harness, "load_model_and_tokenizer", lambda *args, **kwargs: (
        model, FakeTokenizer()))
    monkeypatch.setattr(harness, "set_all_seeds", lambda seed: None)

    metrics = harness.run_logprob_rows(
        RunConfig(), "cpu", ROWS, condition="probe", out_dir=str(tmp_path),
        identity_fields=("fact_id", "relation", "group", "variant", "choice_order"),
        include_scores=True)
    saved = [json.loads(line) for line in
             (tmp_path / "probe_logprob_preds.jsonl").read_text().splitlines()]
    assert metrics["n_items"] == 2
    assert [row["pred"] for row in saved] == ["B", "D"]
    assert saved[0]["fact_id"] == "f1"
    assert saved[0]["letter_logits"] == {"A": 0.1, "B": 0.8, "C": 0.2, "D": 0.3}
    assert saved[0]["correct_margin"] == pytest.approx(0.5)
    assert 0.0 < saved[0]["correct_probability"] < 1.0


def test_generated_rows_keep_identity_fields(monkeypatch, tmp_path):
    import eval.harness as harness

    monkeypatch.setattr(harness, "load_model_and_tokenizer", lambda *args, **kwargs: (
        FakeModel([]), FakeTokenizer()))
    monkeypatch.setattr(harness, "set_all_seeds", lambda seed: None)
    monkeypatch.setattr(harness, "_generate_resilient", lambda *args, **kwargs: [
        "done</think>\n<answer>B</answer>",
        "done</think>\n<answer>D</answer>",
    ])
    metrics = harness.run_eval_rows(
        RunConfig(), "cpu", ROWS, condition="gen", out_dir=str(tmp_path),
        identity_fields=("fact_id", "relation", "group", "variant", "choice_order"))
    saved = [json.loads(line) for line in
             (tmp_path / "gen_preds.jsonl").read_text().splitlines()]
    assert metrics["n_items"] == 2
    assert metrics["format_compliance"] == 1.0
    assert [row["fact_id"] for row in saved] == ["f1", "f2"]


def test_historical_eval_wrapper_keeps_rows_and_metrics(monkeypatch, tmp_path):
    import eval.harness as harness

    calls = []
    model = FakeModel([])
    monkeypatch.setattr(harness, "set_all_seeds", lambda seed: calls.append(("seed", seed)))
    monkeypatch.setattr(harness, "build_eval_split", lambda limit_per_subject=None: (
        calls.append(("rows", limit_per_subject)) or ROWS))

    def load_model(*args, **kwargs):
        calls.append(("model", kwargs.get("adapter_path"), kwargs.get("model_path")))
        return model, FakeTokenizer()

    monkeypatch.setattr(harness, "load_model_and_tokenizer", load_model)
    monkeypatch.setattr(harness, "_generate_resilient", lambda *args, **kwargs: [
        "done</think>\n<answer>B</answer>",
        "done</think>\n<answer>A</answer>",
    ])
    cfg = RunConfig()
    cfg.mode = Mode.FULL
    metrics = harness.run_eval(
        cfg, "cpu", adapter_path="adapter", condition="legacy",
        out_dir=str(tmp_path))
    saved = [json.loads(line) for line in
             (tmp_path / "legacy_preds.jsonl").read_text().splitlines()]

    assert calls == [("seed", 0), ("rows", None), ("model", "adapter", None)]
    assert set(saved[0]) == {
        "subject", "tier", "gold", "pred", "correct", "n_tokens",
        "format_ok", "output",
    }
    assert [row["pred"] for row in saved] == ["B", "A"]
    assert [row["correct"] for row in saved] == [True, False]
    assert set(metrics) == {
        "condition", "n_items", "per_tier_acc", "per_subject_acc", "per_tier_ci",
        "overall_ci", "format_compliance", "mean_len", "extraction_failure_rate",
    }
    assert metrics["condition"] == "legacy" and metrics["n_items"] == 2


def test_historical_logprob_wrapper_keeps_rows_and_metrics(monkeypatch, tmp_path):
    import eval.harness as harness

    torch = types.ModuleType("torch")
    torch.no_grad = NoGrad
    monkeypatch.setitem(sys.modules, "torch", torch)
    calls = []
    model = FakeModel([[0.1, 0.8, 0.2, 0.3], [0.1, 0.2, 0.3, 0.9]])
    monkeypatch.setattr(harness, "set_all_seeds", lambda seed: calls.append(("seed", seed)))
    monkeypatch.setattr(harness, "build_eval_split", lambda limit_per_subject=None: (
        calls.append(("rows", limit_per_subject)) or ROWS))

    def load_model(*args, **kwargs):
        calls.append(("model", kwargs.get("adapter_path"), kwargs.get("model_path")))
        return model, FakeTokenizer()

    monkeypatch.setattr(harness, "load_model_and_tokenizer", load_model)
    cfg = RunConfig()
    cfg.mode = Mode.FULL
    metrics = harness.run_logprob_eval(
        cfg, "cpu", adapter_path="adapter", condition="legacy_probe",
        out_dir=str(tmp_path), model_path=None)
    saved = [json.loads(line) for line in
             (tmp_path / "legacy_probe_logprob_preds.jsonl").read_text().splitlines()]

    assert calls == [("seed", 0), ("rows", None), ("model", "adapter", None)]
    assert set(saved[0]) == {"subject", "tier", "gold", "pred", "correct"}
    assert [row["pred"] for row in saved] == ["B", "D"]
    assert [row["correct"] for row in saved] == [True, True]
    assert set(metrics) == {
        "condition", "metric", "n_items", "per_tier_acc", "per_subject_acc",
        "per_tier_ci", "overall_ci",
    }
    assert metrics["condition"] == "legacy_probe"
    assert metrics["metric"] == "logprob" and metrics["n_items"] == 2
