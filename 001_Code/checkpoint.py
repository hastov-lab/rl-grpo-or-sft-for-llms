# save everything so plots never require retraining
import glob
import hashlib
import json
import os
import pickle
from dataclasses import asdict

from file_utils import read_text


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def save_state(path, state):
    # generic checkpoint dump, real adapters also go through peft save_pretrained
    _ensure_dir(path)
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_state(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_metrics(path, metrics):
    _ensure_dir(path)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def has_checkpoint(out_dir):
    # true if theres a periodic checkpoint to resume from after a crash
    return bool(glob.glob(os.path.join(out_dir, "checkpoint-*")))


def _config_core(cfg, omit_train_fields=()):
    core = {
        "experiment": cfg.experiment,
        "env": cfg.env.name, "mode": cfg.mode.name, "scale": cfg.scale.name,
        "condition": cfg.condition.name,
        "train": asdict(cfg.train), "grpo": asdict(cfg.grpo),
        "lora": asdict(cfg.lora), "reward": asdict(cfg.reward),
    }
    # neutral fields drop out so old hashes still match
    if core["train"].get("a2_action") == "abort":
        core["train"].pop("a2_action")
    if core["train"].get("sft_parameter_mode") == "lora":
        core["train"].pop("sft_parameter_mode")
    if core["train"].get("sft_precision_mode") == "legacy":
        core["train"].pop("sft_precision_mode")
    if core["train"].get("grpo_parameter_mode") == "lora":
        core["train"].pop("grpo_parameter_mode")
    if core["train"].get("grpo_precision_mode") == "legacy":
        core["train"].pop("grpo_precision_mode")
    for f in omit_train_fields:
        core["train"].pop(f, None)
    return core


def config_hash(cfg, omit_train_fields=()):
    # Only hash what actually shapes training. Eval/log tweaks must not block a resume
    core = _config_core(cfg, omit_train_fields)
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()


_config_hash = config_hash


def acceptable_hashes(cfg):
    # hashes this config may legitimately match. Older checkpoints were hashed
    # before the neutral chain fields existed, so a stored hash may omit exactly
    # those fields, and only while they are empty in the current config too; any
    # real training difference still fails
    hashes = [config_hash(cfg)]
    if getattr(cfg.train, "grpo_from_adapter", "") == "":
        hashes.append(config_hash(cfg, ("grpo_from_adapter",)))
        if getattr(cfg.train, "sft_from_adapter", "") == "":
            hashes.append(config_hash(cfg, ("grpo_from_adapter", "sft_from_adapter")))
    return hashes


def write_config_hash(out_dir, cfg):
    # write once, so a later resume can be compared against the config that made the checkpoint
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "config_hash.txt")
    if not os.path.exists(p):
        with open(p, "w") as f:
            f.write(config_hash(cfg))


def check_config_hash(out_dir, cfg):
    # stop a resume from silently mixing a different config into the same checkpoint dir
    p = os.path.join(out_dir, "config_hash.txt")
    if os.path.exists(p) and read_text(p).strip() not in acceptable_hashes(cfg):
        raise ValueError(
            f"config hash mismatch in {out_dir}: this run's config differs from the one that "
            f"created the checkpoint. delete the dir to start fresh or reuse the original config.")


ARTIFACT_TYPE_FILE = "artifact_type.json"
ADAPTER_ARTIFACT_FILES = ("adapter_model.safetensors", "adapter_config.json",
                          "adapter_base.json")
FULL_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def write_artifact_type(out_dir, kind, family=None, scale=None):
    if kind not in ("adapter", "full"):
        raise ValueError(f"bad artifact type: {kind!r}")
    payload = {"kind": kind}
    if family:
        payload["family"] = family
    if scale:
        payload["scale"] = scale
    path = os.path.join(out_dir, ARTIFACT_TYPE_FILE)
    part = path + ".part"
    with open(part, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(part, path)
    return payload


def _full_weight_files(out_dir):
    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path) as f:
                shards = sorted(set(json.load(f).get("weight_map", {}).values()))
        except Exception as exc:
            raise ValueError(f"{out_dir}: full-model shard index does not parse") from exc
        if not shards:
            raise ValueError(f"{out_dir}: full-model shard index has no weights")
        return ["model.safetensors.index.json", *shards]
    if os.path.exists(os.path.join(out_dir, "model.safetensors")):
        return ["model.safetensors"]
    raise ValueError(f"{out_dir}: full-model weights are missing")


def _check_base_record(out_dir, name):
    path = os.path.join(out_dir, name)
    try:
        with open(path) as f:
            record = json.load(f)
    except Exception as exc:
        raise ValueError(f"{out_dir}: {name} does not parse") from exc
    if (record.get("kind") not in ("hf_id", "local")
            or not isinstance(record.get("value"), str)
            or not record["value"]):
        raise ValueError(f"{out_dir}: {name} has a bad base reference")


def check_artifact_complete(out_dir, expected=None):
    path = os.path.join(out_dir, ARTIFACT_TYPE_FILE)
    if not os.path.exists(path):
        raise ValueError(f"{out_dir}: artifact type record is missing")
    try:
        with open(path) as f:
            record = json.load(f)
    except Exception as exc:
        raise ValueError(f"{out_dir}: artifact type record does not parse") from exc
    kind = record.get("kind")
    if kind not in ("adapter", "full"):
        raise ValueError(f"{out_dir}: bad artifact type {kind!r}")
    if expected and kind != expected:
        raise ValueError(f"{out_dir}: expected {expected} artifact, found {kind}")
    names = set(os.listdir(out_dir))
    if kind == "adapter":
        missing = sorted(set(ADAPTER_ARTIFACT_FILES) - names)
        if missing:
            raise ValueError(f"{out_dir}: adapter artifact is missing {missing}")
        if "model.safetensors" in names or "model.safetensors.index.json" in names:
            raise ValueError(f"{out_dir}: adapter artifact contains full-model weights")
        _check_base_record(out_dir, "adapter_base.json")
    else:
        required = {"config.json", "model_base.json", *FULL_TOKENIZER_FILES,
                    *_full_weight_files(out_dir)}
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"{out_dir}: full-model artifact is missing {missing}")
        if "adapter_model.safetensors" in names or "adapter_config.json" in names:
            raise ValueError(f"{out_dir}: full-model artifact contains adapter files")
        _check_base_record(out_dir, "model_base.json")
    return record
