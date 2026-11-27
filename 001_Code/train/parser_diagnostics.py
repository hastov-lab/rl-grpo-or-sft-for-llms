# Record H/B1 reward comparisons without changing the returned rewards

import hashlib
import json
import os

from transformers import TrainerCallback

from data.mmlu import extract_letter as historical_letter
from data.tagging import with_open_tag
from file_utils import read_json
from scripts.s12_eval import IntegrityError, atomic_write_json


DIAGNOSTICS_FILE = "parser_diagnostics.json"
CHECKPOINT_FILE = "parser_diagnostics_binding.json"
COUNTERS = ("responses_checked", "h_only_accuracy_reward", "b1_only_accuracy_reward")


def prefix_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


class ParserDiagnostics(TrainerCallback):
    def __init__(self, result_dir, checkpoint_dir, binding, sample_limit=20, batch_size=8):
        self.path = os.path.join(result_dir, DIAGNOSTICS_FILE)
        self.checkpoint_dir = checkpoint_dir
        self.binding = binding
        self.sample_limit = int(sample_limit)
        self.batch_size = int(batch_size)
        self.pending = None
        if self.sample_limit < 0 or self.batch_size < 1:
            raise ValueError("invalid parser diagnostic limits")
        os.makedirs(result_dir, exist_ok=True)
        self.record = (read_json(self.path) if os.path.exists(self.path) else
                       {"binding": binding, "steps": [], "discarded_attempts": []})
        if self.record.get("binding") != binding:
            raise IntegrityError("parser diagnostics belong to a different run")
        for step, row in enumerate(self.record["steps"], 1):
            if (row["step"] != step or row["responses_checked"] != self.batch_size
                    or row["scoring_calls"] != 1 or len(row["examples"]) > self.sample_limit
                    or any(not isinstance(row[k], int) or row[k] < 0 for k in COUNTERS)
                    or row["h_only_accuracy_reward"] + row["b1_only_accuracy_reward"] > self.batch_size):
                raise IntegrityError("parser diagnostic history is inconsistent")

    def save(self):
        atomic_write_json(self.path, self.record)

    def on_train_begin(self, args, state, control, **kwargs):
        step = int(state.global_step)
        rows = self.record["steps"]
        if len(rows) < step:
            raise IntegrityError("checkpoint is ahead of parser diagnostics")
        if step:
            path = os.path.join(self.checkpoint_dir, f"checkpoint-{step}", CHECKPOINT_FILE)
            expected = {"binding": self.binding, "step": step,
                        "prefix_sha256": prefix_hash(rows[:step])}
            if not os.path.exists(path) or read_json(path) != expected:
                raise IntegrityError("checkpoint parser binding is missing or changed")
        # a rollback leaves orphan updates behind, keep them out of the resumed totals
        if len(rows) > step:
            self.record["discarded_attempts"].append(
                {"resumed_from_step": step, "steps": rows[step:]})
            self.record["steps"] = rows[:step]
        self.pending = None
        self.save()
        return control

    def observe(self, completions, predictions, gold, trainer_state,
                prompts=None, log_metric=None, **kwargs):
        step = int(trainer_state.global_step) + 1
        if self.pending is not None or step != len(self.record["steps"]) + 1:
            raise IntegrityError("unexpected repeated or out-of-order reward scoring")
        if not len(completions) == len(predictions) == len(gold) == self.batch_size:
            raise IntegrityError("parser diagnostic reward batch size changed")
        if prompts is not None and len(prompts) != len(completions):
            raise IntegrityError("parser diagnostic prompts are not aligned")
        old = [historical_letter(with_open_tag(text)) for text in completions]
        row = {"step": step, "responses_checked": len(completions), "scoring_calls": 1,
               "h_only_accuracy_reward": 0, "b1_only_accuracy_reward": 0, "examples": []}
        for index, (text, h, b1, answer) in enumerate(zip(completions, old, predictions, gold)):
            if answer not in ("A", "B", "C", "D"):
                raise IntegrityError("parser diagnostic gold is outside A-D")
            h_match, b1_match = h == answer, b1 == answer
            row["h_only_accuracy_reward"] += int(h_match and not b1_match)
            row["b1_only_accuracy_reward"] += int(b1_match and not h_match)
            if h != b1 and len(row["examples"]) < self.sample_limit:
                row["examples"].append({
                    "response_index": index, "output": text, "gold": answer,
                    "historical": h, "b1": b1, "h_matches_gold": h_match,
                    "b1_matches_gold": b1_match,
                    "prompt_sha256": (hashlib.sha256(prompts[index].encode()).hexdigest()
                                      if prompts is not None else None)})
        self.pending = row
        if log_metric is not None:
            for key in COUNTERS:
                log_metric(f"parser/{key}", row[key])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" not in (logs or {}):
            return control
        step = int(state.global_step)
        if self.pending is None or self.pending["step"] != step:
            raise IntegrityError("training update has no parser comparison")
        row = self.pending
        row["training_log"] = dict(logs, step=step)
        self.record["steps"].append(row)
        self.pending = None
        # this fires before A1/A2, so a sentinel stop cannot swallow the last update
        self.save()
        return control

    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if len(self.record["steps"]) != step or self.pending is not None:
            raise IntegrityError("checkpoint and parser diagnostics are not aligned")
        directory = os.path.join(self.checkpoint_dir, f"checkpoint-{step}")
        if not os.path.isdir(directory):
            raise IntegrityError("parser checkpoint directory is missing")
        atomic_write_json(os.path.join(directory, CHECKPOINT_FILE), {
            "binding": self.binding, "step": step,
            "prefix_sha256": prefix_hash(self.record["steps"])})
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.pending is not None or len(self.record["steps"]) != int(state.global_step):
            raise IntegrityError("training ended with incomplete parser diagnostics")
        return control

    def close_attempt(self):
        if self.pending is not None:
            self.record["discarded_attempts"].append({"uncommitted_scoring": self.pending})
            self.pending = None
            self.save()

    def summary(self):
        rows = self.record["steps"]
        return {"completed_updates": len(rows),
                **{key: sum(row[key] for row in rows) for key in COUNTERS}}


def attach_parser_diagnostics(trainer, cfg, result_dir, checkpoint_dir, binding):
    from data.mmlu_b1 import extract_letter
    from train.reward import make_reward_funcs

    args = trainer.args
    if (trainer.accelerator.num_processes != 1 or args.num_iterations != 1
            or args.steps_per_generation != args.gradient_accumulation_steps
            or args.generation_batch_size != cfg.train.num_generations):
        raise IntegrityError("parser diagnostics require one reward batch per update")
    if trainer.reward_func_names != ["accuracy_reward", "format_reward"]:
        raise IntegrityError("GRPO reward names changed")
    diagnostic = ParserDiagnostics(result_dir, checkpoint_dir, binding,
                                    cfg.log.num_sample_completions, cfg.train.num_generations)
    # swap the reward funcs on this trainer only, the module defaults stay historical
    trainer.reward_funcs = make_reward_funcs(cfg.reward, extractor=extract_letter,
                                            observer=diagnostic.observe)
    trainer.add_callback(diagnostic)
    return diagnostic
