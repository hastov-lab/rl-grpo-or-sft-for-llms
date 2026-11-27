# Event lists were captured from the wrappers before the row-driven split, so a
# reordered side effect or changed trainer argument fails here.
import sys
import types
from types import SimpleNamespace

import pytest

from config import RunConfig
from enums import Condition, Mode


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def test_sft_wrapper_keeps_rows_arguments_and_side_effect_order(monkeypatch, tmp_path):
    import checkpoint
    import obs
    import train.sft as sft

    events = []
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs
            events.append("trainer")
            self.state = SimpleNamespace(log_history=[])

        def train(self, resume_from_checkpoint=None):
            captured["resume"] = resume_from_checkpoint
            events.append("train")

        def save_model(self, path):
            captured["saved_to"] = path
            events.append("save_model")

    class FakeTokenizer:
        eos_token = "<eos>"

    def from_list(rows):
        captured.setdefault("datasets", []).append(rows)
        events.append("dataset")
        return rows

    def tokenizer_from_pretrained(source):
        captured["tokenizer_source"] = source
        events.append("tokenizer")
        return FakeTokenizer()

    def sft_config(**kwargs):
        captured["config"] = kwargs
        events.append("sft_config")
        return SimpleNamespace(**kwargs)

    monkeypatch.setitem(sys.modules, "datasets", _module(
        "datasets", Dataset=SimpleNamespace(from_list=from_list)))
    monkeypatch.setitem(sys.modules, "transformers", _module(
        "transformers", AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_from_pretrained)))
    monkeypatch.setitem(sys.modules, "trl", _module(
        "trl", SFTConfig=sft_config, SFTTrainer=lambda **kwargs: FakeTrainer(**kwargs)))

    train_rows = [{"prompt": "fact prefix", "target": " fact value"}]
    monkeypatch.setattr(sft, "set_all_seeds", lambda seed: events.append("seed"))
    monkeypatch.setattr(sft, "resolve_sft_start", lambda cfg: (
        events.append("resolve_start") or ("/fixed/start", "local", "fixed/start")))
    monkeypatch.setattr(sft, "load_sft", lambda split="train", limit=None: (
        events.append("load_rows") or train_rows))
    monkeypatch.setattr(sft, "resolve_dtype", lambda cfg: "float32")
    monkeypatch.setattr(sft, "make_lora", lambda cfg: events.append("make_lora") or "lora")
    monkeypatch.setattr(sft, "write_adapter_base", lambda path, kind, value: (
        captured.update(base=(path, kind, value)), events.append("adapter_base")))
    monkeypatch.setattr(checkpoint, "write_config_hash", lambda path, cfg: events.append("write_hash"))
    monkeypatch.setattr(checkpoint, "check_config_hash", lambda path, cfg: events.append("check_hash"))
    monkeypatch.setattr(obs, "prepare_wandb", lambda cfg, report_to: (
        events.append("prepare_wandb") or {"enabled": False}))
    monkeypatch.setattr(obs, "stamp_trainer_logs", lambda trainer: (
        events.append("stamp_logs") or trainer))

    cfg = RunConfig()
    cfg.mode = Mode.FULL
    cfg.condition = Condition.SFT_ONLY
    cfg.train.sft_val_examples = 0
    out_dir = str(tmp_path / "sft")
    sft.run_sft(cfg, "cpu", out_dir=out_dir, report_to=[], allow_oom_fallback=False)

    assert events == [
        "seed", "write_hash", "resolve_start", "load_rows", "tokenizer", "dataset",
        "prepare_wandb", "check_hash", "sft_config", "make_lora", "trainer",
        "stamp_logs", "train", "save_model", "adapter_base",
    ]
    assert captured["trainer"]["model"] == "/fixed/start"
    assert captured["trainer"]["train_dataset"] == [
        {"prompt": "fact prefix\n", "completion": " fact value<eos>"}]
    assert captured["trainer"]["eval_dataset"] is None
    assert captured["config"]["model_init_kwargs"] == {"dtype": "float32"}
    assert captured["resume"] is None
    assert captured["base"] == (out_dir, "local", "fixed/start")


def test_grpo_wrapper_keeps_rows_arguments_and_side_effect_order(monkeypatch, tmp_path):
    import checkpoint
    import obs
    import train.grpo as grpo

    events = []
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs
            events.append("trainer")
            self.state = SimpleNamespace(log_history=[])

        def add_callback(self, callback):
            captured["callback"] = callback
            events.append("add_callback")

        def train(self, resume_from_checkpoint=None):
            captured["resume"] = resume_from_checkpoint
            events.append("train")

        def save_model(self, path):
            captured["saved_to"] = path
            events.append("save_model")

    def from_list(rows):
        captured["dataset"] = rows
        events.append("dataset")
        return rows

    def grpo_config(**kwargs):
        captured["config"] = kwargs
        events.append("grpo_config")
        return SimpleNamespace(**kwargs)

    monkeypatch.setitem(sys.modules, "datasets", _module(
        "datasets", Dataset=SimpleNamespace(from_list=from_list)))
    monkeypatch.setitem(sys.modules, "trl", _module(
        "trl", GRPOConfig=grpo_config, GRPOTrainer=lambda **kwargs: FakeTrainer(**kwargs)))

    source_rows = [{"prompt": "question\n<think>", "gold": "C", "tier": "hard",
                    "subject": "formal_logic"}]
    monkeypatch.setattr(grpo, "set_all_seeds", lambda seed: events.append("seed"))
    monkeypatch.setattr(grpo, "build_grpo_split", lambda limit_per_subject=None: (
        events.append("load_rows") or source_rows))
    monkeypatch.setattr(grpo, "resolve_model_id", lambda cfg: (
        events.append("resolve_start") or "model-id"))
    monkeypatch.setattr(grpo, "resolve_dtype", lambda cfg: "float32")
    monkeypatch.setattr(grpo, "make_reward_funcs", lambda cfg: (
        events.append("reward_funcs") or ["accuracy", "format"]))
    monkeypatch.setattr(grpo, "make_lora", lambda cfg: events.append("make_lora") or "lora")
    monkeypatch.setattr(grpo, "write_adapter_base", lambda path, kind, value: (
        captured.update(base=(path, kind, value)), events.append("adapter_base")))
    monkeypatch.setattr(checkpoint, "write_config_hash", lambda path, cfg: events.append("write_hash"))
    monkeypatch.setattr(checkpoint, "check_config_hash", lambda path, cfg: events.append("check_hash"))
    monkeypatch.setattr(obs, "prepare_wandb", lambda cfg, report_to: (
        events.append("prepare_wandb") or {"enabled": False}))
    monkeypatch.setattr(obs, "stamp_trainer_logs", lambda trainer: (
        events.append("stamp_logs") or trainer))
    monkeypatch.setattr(obs, "make_grpo_sentinel", lambda cfg: (
        events.append("sentinel") or "callback"))

    cfg = RunConfig()
    cfg.mode = Mode.FULL
    cfg.condition = Condition.BASE_GRPO
    out_dir = str(tmp_path / "grpo")
    grpo.run_grpo(cfg, "cpu", out_dir=out_dir, report_to=[], allow_oom_fallback=False)

    assert events == [
        "seed", "write_hash", "load_rows", "dataset", "resolve_start",
        "prepare_wandb", "check_hash", "grpo_config", "reward_funcs", "make_lora",
        "trainer", "stamp_logs", "sentinel", "add_callback", "train", "save_model",
        "adapter_base",
    ]
    assert captured["dataset"] == [
        {"prompt": "question\n<think>", "gold": "C", "tier": "hard"}]
    assert captured["trainer"]["model"] == "model-id"
    assert captured["trainer"]["reward_funcs"] == ["accuracy", "format"]
    assert captured["config"]["num_generations"] == 8
    assert captured["resume"] is None
    assert captured["base"] == (out_dir, "hf_id", "model-id")


def test_standard_wrappers_reject_synthetic_rows_before_side_effects(monkeypatch):
    # Guard against a W2 config reaching the wrappers and quietly training on
    # gsm8k or the 245-prompt mmlu pool.
    import train.grpo as grpo
    import train.sft as sft

    cfg = RunConfig()
    cfg.experiment = "034_smoke"
    cfg.train.sft_data = "synthetic_facts"
    monkeypatch.setattr(sft, "set_all_seeds", lambda seed: pytest.fail("sft touched the run"))
    monkeypatch.setattr(grpo, "set_all_seeds", lambda seed: pytest.fail("grpo touched the run"))
    with pytest.raises(ValueError, match="run_sft_rows"):
        sft.run_sft(cfg, "cpu")
    with pytest.raises(ValueError, match="run_grpo_rows"):
        grpo.run_grpo(cfg, "cpu")
