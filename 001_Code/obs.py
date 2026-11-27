# observability: file logging, run manifest, gpu + timing, tracker selection, compute ledger
import importlib.util
import hashlib
import json
import os
import math
import subprocess
import sys
import time
from datetime import datetime


def get_logger(out_dir, name="run"):
    import logging
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(os.path.join(out_dir, "run.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class Timer:
    # time each stage to track wall time and the gpu budget
    def __enter__(self):
        self._t = time.time()
        self.seconds = None
        return self

    def __exit__(self, *a):
        self.seconds = round(time.time() - self._t, 3)


def _has(mod):
    return importlib.util.find_spec(mod) is not None


def resolve_report_to(cfg):
    # only keep trackers that are actually installed so a missing one cant crash the run
    ok = []
    for t in cfg.log.trackers:
        if t == "tensorboard" and not (_has("tensorboard") or _has("tensorboardX")):
            continue
        if t == "wandb" and not _has("wandb"):
            continue
        ok.append(t)
    return ok


def _git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _versions():
    out = {}
    for mod in ["trl", "transformers", "peft", "accelerate", "datasets", "torch", "numpy"]:
        if importlib.util.find_spec(mod) is None:
            out[mod] = None
            continue
        try:
            out[mod] = getattr(__import__(mod), "__version__", None)
        except Exception:
            out[mod] = None
    # the pinned dataset revisions belong next to the library versions, so every
    # manifest records exactly which data a run loaded
    from data.mmlu import MMLU_REVISION
    from data.gsm8k import GSM8K_REVISION
    out["cais/mmlu@"] = MMLU_REVISION
    out["openai/gsm8k@"] = GSM8K_REVISION
    return out


versions = _versions


def gpu_snapshot():
    try:
        import torch
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            return {"device": torch.cuda.get_device_name(i),
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated(i) / 1e9, 3)}
    except Exception:
        pass
    return {"device": "cpu", "peak_vram_gb": None}


def write_manifest(cfg, out_dir, extra=None):
    # everything needed to reproduce the run: config, seed, git, library versions
    from config import run_tag
    from train.model_utils import resolve_dtype
    os.makedirs(out_dir, exist_ok=True)
    man = {
        "run_tag": run_tag(cfg),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": cfg.experiment, "condition": cfg.condition.name,
        "env": cfg.env.name, "mode": cfg.mode.name, "scale": cfg.scale.name,
        "seed": cfg.train.seed,
        "exploratory": cfg.exploratory,
        "resolved_dtype": resolve_dtype(cfg),  # t4 gives fp16, a100 gives bf16, worth recording
        # every sub config, the 003 manifest missing grpo/eval taught us that lesson
        "config": {"lora": vars(cfg.lora), "reward": vars(cfg.reward), "train": vars(cfg.train),
                   "grpo": vars(cfg.grpo), "eval": vars(cfg.eval), "exec": vars(cfg.exec)},
        "git": _git_info(),
        "versions": _versions(),
    }
    if extra:
        man.update(extra)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    return man


def append_ledger(path, row):
    # append one timing record per run
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def read_key_file(path):
    # pull an api key from a local gitignored file, none if its missing
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def configure_wandb(cfg, report_to):
    # make the config authoritative when a runtime still has older values
    enabled = "wandb" in (report_to or [])
    state = {
        "enabled": enabled,
        "key_source": None,
    }
    if not enabled:
        return state

    os.environ["WANDB_PROJECT"] = cfg.log.wandb_project
    os.environ["WANDB_MODE"] = cfg.log.wandb_mode

    if os.environ.get("WANDB_API_KEY"):
        state["key_source"] = "env"
    else:
        key = read_key_file(cfg.log.wandb_key_file)
        if key:
            os.environ["WANDB_API_KEY"] = key
            state["key_source"] = "file"
        else:
            state["key_source"] = "login_or_offline"
    return state


def prepare_wandb(cfg, report_to):
    # close any active run before a new trainer creates its own stage record
    state = configure_wandb(cfg, report_to)
    if state["enabled"]:
        wandb_finish()
    return state


def stamp_trainer_logs(trainer):
    # put a wall clock time at the front of every per step log line the trainer
    # prints, so the console history shows when each step happened
    try:
        from transformers.trainer_callback import ProgressCallback

        if getattr(trainer.args, "disable_tqdm", False):
            return trainer  # different printer path, not worth special casing

        class stamped(ProgressCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is not None:
                    logs = {"time": datetime.now().strftime("%H:%M:%S"), **logs}
                super().on_log(args, state, control, logs=logs, **kwargs)

        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(stamped)
    except Exception:
        pass  # if the internals ever move, plain logs beat a crashed run
    return trainer


STARVATION_LOG_FIELDS = {
    "mean_accuracy_reward": "rewards/accuracy_reward/mean",
    "mean_format_reward": "rewards/format_reward/mean",
    "share_zero_reward_spread": "frac_reward_zero_std",
}


def normalise_starvation_metrics(logs):
    missing = [source for source in STARVATION_LOG_FIELDS.values()
               if source not in logs]
    if missing:
        raise RuntimeError(f"GRPO log is missing starvation fields: {missing}")
    record = {name: float(logs[source])
              for name, source in STARVATION_LOG_FIELDS.items()}
    if any(not math.isfinite(value) for value in record.values()):
        raise RuntimeError("GRPO starvation fields contain a non-finite value")
    if not 0.0 <= record["share_zero_reward_spread"] <= 1.0:
        raise RuntimeError("GRPO zero-spread share is outside [0, 1]")
    return record


def make_starvation_guard(prior_history=None):
    from transformers import TrainerCallback

    class starvation_guard(TrainerCallback):
        def __init__(self):
            self.first_update = None
            self.last_step = 0
            rows = [row for row in (prior_history or [])
                    if "loss" in row and row.get("step") is not None]
            for expected, row in enumerate(rows, 1):
                if int(row["step"]) != expected:
                    raise RuntimeError("GRPO starvation history is not contiguous")
                current = normalise_starvation_metrics(row)
                if self.first_update is None:
                    self.first_update = current
                self.last_step = expected

        def on_log(self, args, state, control, logs=None, **kwargs):
            if "loss" not in (logs or {}):
                return control
            current = normalise_starvation_metrics(logs)
            step = int(state.global_step)
            if step != self.last_step + 1:
                raise RuntimeError("GRPO starvation history is not contiguous")
            if self.first_update is None:
                self.first_update = current
            self.last_step = step
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if state.global_step and self.first_update is None:
                raise RuntimeError("GRPO starvation fields were not recorded at update 1")
            return control

    return starvation_guard()


# registered abort criteria for the first training window

def a1_truncation(clipped_ratios, threshold=0.3):
    # too many rollouts hitting the cap early on means the cap is wrong
    return len(clipped_ratios) >= 50 and sum(clipped_ratios[:50]) / 50 > threshold


def a2_immobility(acc_early, acc_mid, kl_mid):
    # reward flat AND the policy not moving = lr/beta suspect, dont burn the rest
    return (acc_mid - acc_early) < 0.02 and kl_mid < 0.005


A2_VERDICT_FILE = "a2_verdict.json"
A2_MIN_IMPROVEMENT = 0.02
A2_MAX_MID_KL = 0.005


def normalise_a2_history(history):
    rows = []
    seen = set()
    for row in history or []:
        step = row.get("step")
        acc = row.get("rewards/accuracy_reward/mean", row.get("accuracy_reward"))
        kl = row.get("kl")
        if step is None or acc is None or kl is None:
            continue
        if int(step) in seen:
            raise RuntimeError("a2 history steps are duplicated")
        seen.add(int(step))
        rows.append({"step": int(step), "accuracy_reward": float(acc), "kl": float(kl)})
        if not math.isfinite(rows[-1]["accuracy_reward"]) or not math.isfinite(rows[-1]["kl"]):
            raise RuntimeError("a2 history contains a non-finite value")
    rows.sort(key=lambda row: row["step"])
    if rows:
        steps = [row["step"] for row in rows]
        if steps != list(range(1, steps[-1] + 1)):
            raise RuntimeError("a2 history steps are missing or duplicated")
    return rows


_a2_rows = normalise_a2_history


def a2_history_sha256(rows):
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def a2_verdict_payload(cfg, rows):
    from checkpoint import config_hash

    rows = normalise_a2_history(rows)
    if [row["step"] for row in rows] != list(range(1, 151)):
        raise RuntimeError("a2 verdict requires exactly steps 1-150")
    early = sum(row["accuracy_reward"] for row in rows[:50]) / 50
    mid = sum(row["accuracy_reward"] for row in rows[100:150]) / 50
    kl_mid = sum(row["kl"] for row in rows[100:150]) / 50
    return {
        "experiment": cfg.experiment,
        "seed": cfg.train.seed,
        "step": 150,
        "early_accuracy_reward_mean": early,
        "mid_accuracy_reward_mean": mid,
        "signed_improvement": mid - early,
        "mid_kl_mean": kl_mid,
        "min_improvement": A2_MIN_IMPROVEMENT,
        "max_mid_kl": A2_MAX_MID_KL,
        "would_fire": a2_immobility(early, mid, kl_mid),
        "config_hash": config_hash(cfg),
        "prefix_history_sha256": a2_history_sha256(rows[:100]),
        "verdict_history_sha256": a2_history_sha256(rows),
    }


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    part = path + ".part"
    with open(part, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(part, path)


def verify_a2_verdict(path, cfg, rows=None):
    from checkpoint import config_hash
    from file_utils import read_json

    try:
        record = read_json(path)
    except Exception as exc:
        raise RuntimeError(f"a2 verdict does not parse: {path}") from exc
    required = {
        "experiment", "seed", "step", "early_accuracy_reward_mean",
        "mid_accuracy_reward_mean", "signed_improvement", "mid_kl_mean",
        "min_improvement", "max_mid_kl", "would_fire", "config_hash",
        "prefix_history_sha256", "verdict_history_sha256",
    }
    if set(record) != required:
        raise RuntimeError("a2 verdict fields do not match the registered schema")
    expected = a2_verdict_payload(cfg, rows) if rows is not None else None
    if expected is not None and record != expected:
        raise RuntimeError("a2 verdict does not match the restored training history")
    if (record["experiment"] != cfg.experiment or record["seed"] != cfg.train.seed
            or record["config_hash"] != config_hash(cfg)):
        raise RuntimeError("a2 verdict is bound to a different run")
    if (record["step"] != 150 or record["min_improvement"] != A2_MIN_IMPROVEMENT
            or record["max_mid_kl"] != A2_MAX_MID_KL):
        raise RuntimeError("a2 verdict thresholds changed")
    calculated = a2_immobility(record["early_accuracy_reward_mean"],
                                record["mid_accuracy_reward_mean"],
                                record["mid_kl_mean"])
    if record["would_fire"] is not calculated:
        raise RuntimeError("a2 verdict classification is inconsistent")
    return record


def projected_hours(step_times, max_steps):
    # project total train time from the recent step times
    recent = step_times[-10:]
    return (sum(recent) / len(recent)) * max_steps / 3600 if recent else 0.0


def _alert(title, text):
    # best effort wandb email so a stopped overnight run doesnt wait until morning
    try:
        import wandb
        if wandb.run is not None:
            wandb.alert(title=title, text=text)
    except Exception:
        pass


def make_grpo_sentinel(cfg, out_dir=None, prior_history=None):
    # raise on sentinel failure so the run cannot be marked complete
    from transformers import TrainerCallback

    class sentinel(TrainerCallback):
        def __init__(self):
            self.clipped = [float(row["completions/clipped_ratio"])
                            for row in (prior_history or [])
                            if row.get("completions/clipped_ratio") is not None]
            self.times = [float(row["step_time"])
                          for row in (prior_history or [])
                          if row.get("step_time") is not None]
            self.rows = normalise_a2_history(prior_history)
            self.acc = [row["accuracy_reward"] for row in self.rows]
            self.kl = [row["kl"] for row in self.rows]
            self.steps = [row["step"] for row in self.rows]
            self.verdict_path = os.path.join(out_dir, A2_VERDICT_FILE) if out_dir else None
            if cfg.train.a2_action == "record":
                if not self.verdict_path:
                    raise RuntimeError("a2 record mode requires an output directory")
                if len(self.rows) < 100:
                    raise RuntimeError("a2 record mode requires the restored first 100 steps")
                if len(self.rows) >= 150:
                    record = verify_a2_verdict(self.verdict_path, cfg, self.rows[:150])
                    if record["would_fire"] is not True:
                        raise RuntimeError("a2 non-reproducing trajectory is already concluded")
            if cfg.train.a2_action == "observe" and len(self.rows) >= 150:
                if not self.verdict_path:
                    raise RuntimeError("a2 observe mode requires an output directory")
                verify_a2_verdict(self.verdict_path, cfg, self.rows[:150])

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            if "completions/clipped_ratio" in logs:
                self.clipped.append(float(logs["completions/clipped_ratio"]))
            if "rewards/accuracy_reward/mean" in logs:
                self.acc.append(float(logs["rewards/accuracy_reward/mean"]))
                if cfg.train.a2_action in ("record", "observe"):
                    step = int(state.global_step)
                    if self.steps and step != self.steps[-1] + 1:
                        raise RuntimeError("a2 callback history is not contiguous")
                    if not self.steps and step != 1:
                        raise RuntimeError("a2 callback history was not restored")
                    if "kl" not in logs:
                        raise RuntimeError("a2 callback log is missing kl")
                    self.steps.append(step)
                    self.rows.append({
                        "step": step,
                        "accuracy_reward": float(logs["rewards/accuracy_reward/mean"]),
                        "kl": float(logs["kl"]),
                    })
            if "kl" in logs:
                self.kl.append(float(logs["kl"]))
            if "step_time" in logs:
                self.times.append(float(logs["step_time"]))
            return self._check(state, control)

        def _fire(self, msg):
            _alert("grpo sentinel abort", msg)
            raise RuntimeError(f"sentinel abort: {msg}")

        def _check(self, state, control):
            s = state.global_step
            if s == 20 and cfg.exec.max_train_hours:
                proj = projected_hours(self.times, cfg.train.max_steps)
                if proj > cfg.exec.max_train_hours:
                    self._fire(f"a4: projected {proj:.1f}h grpo train time exceeds the "
                               f"{cfg.exec.max_train_hours}h limit. rethink before burning budget")
            if s == 50 and a1_truncation(self.clipped):
                self._fire("a1: over 30% of rollouts truncated in the first 50 steps. "
                           "raise the completion cap or mask truncated completions")
            if cfg.train.a2_action == "record" and s > 150:
                record = verify_a2_verdict(self.verdict_path, cfg, self.rows[:150])
                if record["would_fire"] is False:
                    raise RuntimeError("a2 non-reproducing trajectory continued past step 150")
            if cfg.train.a2_action == "observe" and s > 150:
                verify_a2_verdict(self.verdict_path, cfg, self.rows[:150])
            if s == 150:
                if cfg.train.a2_action == "record" and self.steps != list(range(1, 151)):
                    raise RuntimeError("a2 record mode reached step 150 without complete history")
                if cfg.train.a2_action == "record":
                    payload = a2_verdict_payload(cfg, self.rows)
                    if os.path.exists(self.verdict_path):
                        verify_a2_verdict(self.verdict_path, cfg, self.rows)
                    else:
                        _atomic_json(self.verdict_path, payload)
                        verify_a2_verdict(self.verdict_path, cfg, self.rows)
                    if payload["would_fire"] is False:
                        control.should_training_stop = True
                    return control
                if cfg.train.a2_action == "observe":
                    if self.steps != list(range(1, 151)):
                        raise RuntimeError("a2 observe mode reached step 150 without complete history")
                    if not self.verdict_path:
                        raise RuntimeError("a2 observe mode requires an output directory")
                    payload = a2_verdict_payload(cfg, self.rows)
                    if os.path.exists(self.verdict_path):
                        verify_a2_verdict(self.verdict_path, cfg, self.rows)
                    else:
                        _atomic_json(self.verdict_path, payload)
                        verify_a2_verdict(self.verdict_path, cfg, self.rows)
                    return control
                if len(self.acc) >= 150:
                    early = sum(self.acc[:50]) / 50
                    mid = sum(self.acc[100:150]) / 50
                    kl_mid = sum(self.kl[100:150]) / 50 if len(self.kl) >= 150 else 1.0
                    if a2_immobility(early, mid, kl_mid):
                        self._fire("a2: reward flat and kl near zero by step 150, "
                                   "policy is not moving. suspect lr or beta")
            return control

    return sentinel()


def next_fallback(per_device, accum, checkpointing):
    # The oom ladder: halve the batch while doubling accum, then turn on
    # gradient checkpointing. Returns none when there is nothing left to try
    if per_device > 1:
        return per_device // 2, accum * 2, checkpointing
    if not checkpointing:
        return per_device, accum, True
    return None


def oom_ladder(attempt, per_device, accum, checkpointing):
    # Run attempt(), stepping down the ladder on cuda oom. Anything already
    # checkpointed is kept because attempts resume from the last checkpoint
    while True:
        try:
            return attempt(per_device, accum, checkpointing)
        except Exception as e:
            if "out of memory" not in str(e).lower():
                raise
            nxt = next_fallback(per_device, accum, checkpointing)
            if nxt is None:
                raise
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            per_device, accum, checkpointing = nxt
            _alert("grpo oom fallback",
                   f"retrying with batch {per_device} x accum {accum}, checkpointing {checkpointing}")
            print(f"oom: retrying with batch {per_device} x accum {accum}, checkpointing {checkpointing}")


def wandb_finish():
    # close the current run so the next trainer starts a separate record
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
