from types import SimpleNamespace

import pytest

from config import RewardConfig
from data.mmlu_b1 import extract_letter
from file_utils import read_json
from scripts.s12_eval import IntegrityError
from train.parser_diagnostics import ParserDiagnostics, CHECKPOINT_FILE
from train.reward import make_reward_funcs


WINDOW_TEXT = "answer b. " + "x" * 24 + " Cat"
TEXTS = ["The answer is unspecified.", WINDOW_TEXT, WINDOW_TEXT, "answer: b."]
GOLD = ["D", "B", "A", "B"]


def score_step(diagnostic, step):
    state = SimpleNamespace(global_step=step - 1)
    reward = make_reward_funcs(RewardConfig(), extractor=extract_letter,
                               observer=diagnostic.observe)[0]
    result = reward(TEXTS, gold=GOLD, trainer_state=state, prompts=["question"] * 4)
    state.global_step = step
    diagnostic.on_log(None, state, None, logs={"loss": 1.0})
    return result


def test_counters_bound_samples_and_ignore_same_reward_letter_change(tmp_path):
    diagnostic = ParserDiagnostics(str(tmp_path), str(tmp_path / "cp"), "binding", 2, 4)
    assert score_step(diagnostic, 1) == [0.0, 1.0, 0.0, 1.0]
    row = read_json(diagnostic.path)["steps"][0]
    assert diagnostic.summary() == {"completed_updates": 1, "responses_checked": 4,
                                    "h_only_accuracy_reward": 1, "b1_only_accuracy_reward": 1}
    assert len(row["examples"]) == 2
    assert row["examples"][1]["historical"] == "C"
    assert row["examples"][1]["b1"] == "B"


def test_repeated_scoring_is_not_silently_double_counted(tmp_path):
    diagnostic = ParserDiagnostics(str(tmp_path), str(tmp_path / "cp"), "binding", 2, 4)
    kwargs = dict(completions=TEXTS, predictions=[None, "C", "C", "B"], gold=GOLD,
                  trainer_state=SimpleNamespace(global_step=0))
    diagnostic.observe(**kwargs)
    with pytest.raises(IntegrityError, match="repeated"):
        diagnostic.observe(**kwargs)
    diagnostic.close_attempt()
    assert diagnostic.summary()["responses_checked"] == 0
    assert "uncommitted_scoring" in read_json(diagnostic.path)["discarded_attempts"][0]


def test_resume_uses_checkpoint_prefix_not_discarded_tail(tmp_path):
    # the resumed run and the abandoned tail need separate denominators
    cp = tmp_path / "cp"
    diagnostic = ParserDiagnostics(str(tmp_path), str(cp), "binding", 2, 4)
    score_step(diagnostic, 1)
    (cp / "checkpoint-1").mkdir(parents=True)
    diagnostic.on_save(None, SimpleNamespace(global_step=1), None)
    score_step(diagnostic, 2)
    restored = ParserDiagnostics(str(tmp_path), str(cp), "binding", 2, 4)
    restored.on_train_begin(None, SimpleNamespace(global_step=1), None)
    assert restored.summary()["responses_checked"] == 4
    score_step(restored, 2)
    assert restored.summary()["responses_checked"] == 8
    assert len(restored.record["discarded_attempts"]) == 1


def test_checkpoint_binding_and_missing_counts_are_not_zero(tmp_path):
    diagnostic = ParserDiagnostics(str(tmp_path), str(tmp_path / "cp"), "binding", 2, 4)
    with pytest.raises(IntegrityError, match="ahead"):
        diagnostic.on_train_begin(None, SimpleNamespace(global_step=1), None)
    score_step(diagnostic, 1)
    with pytest.raises(IntegrityError, match="missing or changed"):
        diagnostic.on_train_begin(None, SimpleNamespace(global_step=1), None)
    with pytest.raises(IntegrityError, match="different run"):
        ParserDiagnostics(str(tmp_path), str(tmp_path / "cp"), "other", 2, 4)


def test_a1_flush_precedes_sentinel_exception(tmp_path, monkeypatch):
    # update 50 must stay readable even when no final model gets saved
    import obs
    from config import RunConfig
    from transformers.trainer_callback import CallbackHandler, TrainerControl, TrainerState
    monkeypatch.setattr(obs, "_alert", lambda *args: None)
    cfg = RunConfig()
    diagnostic = ParserDiagnostics(str(tmp_path), str(tmp_path / "cp"), "binding", 2, 4)
    handler = CallbackHandler([diagnostic, obs.make_grpo_sentinel(cfg)], None, None, None, None)
    state, control = TrainerState(), TrainerControl()
    for step in range(1, 51):
        state.global_step = step - 1
        diagnostic.observe(TEXTS, [None, "C", "C", "B"], GOLD, state)
        state.global_step = step
        logs = {"loss": 1.0, "completions/clipped_ratio": 0.5,
                "rewards/accuracy_reward/mean": 0.1, "kl": 0.1}
        if step < 50:
            handler.on_log(None, state, control, logs)
        else:
            with pytest.raises(RuntimeError, match="sentinel abort: a1"):
                handler.on_log(None, state, control, logs)
    assert read_json(diagnostic.path)["steps"][-1]["step"] == 50


def test_real_trl_scoring_checkpoint_and_resume(tmp_path):
    import torch
    from peft import LoraConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    from config import RunConfig
    from train.parser_diagnostics import attach_parser_diagnostics

    # A tiny random model exercises the installed trainer with no weight download
    backend = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2,
                                   "question": 3, "answer": 4, "A": 5, "B": 6, "C": 7, "D": 8}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                        eos_token="<eos>", unk_token="<unk>", padding_side="left")
    dataset = Dataset.from_list([{"prompt": "question", "gold": "B"}] * 4)
    cfg = RunConfig()
    cfg.train.num_generations = 2
    cp = tmp_path / "cp"

    class stop_at_one(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_training_stop = True
            return control

    def trainer():
        torch.manual_seed(19)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=9, hidden_size=16, intermediate_size=32,
                                            num_hidden_layers=1, num_attention_heads=2,
                                            num_key_value_heads=2, pad_token_id=0, eos_token_id=1))
        args = GRPOConfig(output_dir=str(cp), max_steps=2, num_generations=2,
                          per_device_train_batch_size=2, gradient_accumulation_steps=1,
                          max_completion_length=4, logging_steps=1, save_steps=1,
                          gradient_checkpointing=False, beta=0.04, use_cpu=True,
                          bf16=False, fp16=False, report_to=[], disable_tqdm=True,
                          log_completions=False, seed=19)
        instance = GRPOTrainer(model=model, args=args, train_dataset=dataset,
                               processing_class=tokenizer, reward_funcs=make_reward_funcs(cfg.reward),
                               peft_config=LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                                      task_type="CAUSAL_LM"))
        diagnostic = attach_parser_diagnostics(instance, cfg, str(tmp_path / "result"), str(cp), "tiny")
        return instance, diagnostic

    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first, diagnostic = trainer()
        first.add_callback(stop_at_one())
        first.train()
        assert diagnostic.summary()["responses_checked"] == 2
        assert (cp / "checkpoint-1" / CHECKPOINT_FILE).is_file()
        second, diagnostic = trainer()
        second.train(resume_from_checkpoint=str(cp / "checkpoint-1"))
        assert diagnostic.summary()["completed_updates"] == 2
        assert diagnostic.summary()["responses_checked"] == 4
    finally:
        torch.set_num_threads(before)
