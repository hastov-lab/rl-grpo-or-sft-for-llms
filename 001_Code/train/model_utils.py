# Shared model helpers. Heavy imports stay inside the functions so the module
# imports fine without torch (keeps the offline tests light)
import json
import os

from config import MODEL_IDS, SCRATCH_GPT_SCALES, SMOKE_MODEL_ID
from enums import Mode


SCRATCH_LORA_TARGET_MODULES = (
    "W_query", "W_key", "W_value", "out_proj", "layers.0", "layers.2",
)


def is_scratch_gpt(cfg):
    return cfg.scale in SCRATCH_GPT_SCALES


def resolve_model_id(cfg):
    # Qwen smoke uses the tiny stand in. Scratch GPT keeps its selected scale so
    # the custom architecture and weight mapping are exercised.
    if is_scratch_gpt(cfg):
        return MODEL_IDS[cfg.scale]
    if cfg.mode == Mode.SMOKE:
        return SMOKE_MODEL_ID
    return MODEL_IDS[cfg.scale]


def resolve_dtype(cfg):
    # auto picks bf16 on capable gpus, else fp16 on gpu, else fp32 on cpu
    want = cfg.train.dtype
    if want != "auto":
        return want
    try:
        import torch
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return "bfloat16"
            return "float16"
    except Exception:
        pass
    return "float32"


def make_lora(cfg):
    from peft import LoraConfig as PeftLora
    kwargs = dict(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        task_type="CAUSAL_LM",
    )
    if cfg.lora.target_modules is not None:
        kwargs["target_modules"] = cfg.lora.target_modules
    elif is_scratch_gpt(cfg):
        kwargs["target_modules"] = list(SCRATCH_LORA_TARGET_MODULES)
    return PeftLora(**kwargs)


def _torch_dtype(name):
    import torch
    if isinstance(name, torch.dtype):
        return name
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"bad torch dtype: {name!r}") from exc


def _scratch_source(cfg, source):
    from gpt2_scratch.weights import reference_checkpoint
    alias = MODEL_IDS[cfg.scale]
    model_id, revision = reference_checkpoint(alias)
    if source not in (alias, model_id):
        raise ValueError(f"scratch GPT source {source!r} does not match scale {cfg.scale.name}")
    return alias, model_id, revision


def _check_scratch_artifact(cfg, source):
    from checkpoint import check_artifact_complete
    record = check_artifact_complete(source, expected="full")
    if record.get("family") != "scratch_gpt":
        raise ValueError(f"{source}: full artifact is not from the scratch GPT family")
    if record.get("scale") != cfg.scale.name:
        raise ValueError(f"{source}: full artifact is from a different scale")


def load_scratch_tokenizer(cfg, source):
    from transformers import GPT2TokenizerFast

    if os.path.isdir(source):
        _check_scratch_artifact(cfg, source)
        tok = GPT2TokenizerFast.from_pretrained(source)
    else:
        alias, _, revision = _scratch_source(cfg, source)
        tok = GPT2TokenizerFast.from_pretrained(alias, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_scratch_model(cfg, source, dtype=None, device=None):
    from gpt2_scratch import (ScratchGPT2Config, ScratchGPT2ForCausalLM,
                              load_weights_into_gpt)

    dtype = _torch_dtype(dtype or resolve_dtype(cfg))
    if os.path.isdir(source):
        _check_scratch_artifact(cfg, source)
        model = ScratchGPT2ForCausalLM.from_pretrained(source, dtype=dtype)
    else:
        from gpt2_scratch.weights import get_pretrained_gpt2
        alias, _, revision = _scratch_source(cfg, source)
        model = ScratchGPT2ForCausalLM(
            ScratchGPT2Config.from_scale(alias)).to(dtype=dtype)
        load_weights_into_gpt(
            model, get_pretrained_gpt2(alias, revision=revision))
    if device is not None:
        model = model.to(device)
    return model


def load_scratch_model_and_tokenizer(cfg, source, dtype=None, device=None):
    model = load_scratch_model(cfg, source, dtype=dtype, device=device)
    tok = load_scratch_tokenizer(cfg, source)
    return model, tok


def training_model_and_tokenizer(cfg, model_src, dtype):
    if not is_scratch_gpt(cfg):
        return model_src, None
    return load_scratch_model_and_tokenizer(cfg, model_src, dtype=dtype)


def parameter_mode(cfg, stage):
    if stage == "sft":
        return cfg.train.sft_parameter_mode
    if stage == "grpo":
        return cfg.train.grpo_parameter_mode
    raise ValueError(f"bad training stage: {stage!r}")


def precision_settings(cfg, stage):
    mode = (cfg.train.sft_precision_mode if stage == "sft"
            else cfg.train.grpo_precision_mode)
    matched = mode == "fp32_master_bf16"
    return {
        "model_dtype": "float32" if matched else resolve_dtype(cfg),
        "bf16": matched,
        "parameter_mode": parameter_mode(cfg, stage),
    }


def merge_model_dtype(cfg):
    if is_scratch_gpt(cfg):
        return precision_settings(cfg, "sft")["model_dtype"]
    return resolve_dtype(cfg)


def record_scratch_artifact(cfg, tokenizer, out_dir, stage, base_kind, base_val):
    from checkpoint import write_artifact_type
    mode = parameter_mode(cfg, stage)
    tokenizer.save_pretrained(out_dir)
    if mode == "full":
        write_model_base(out_dir, base_kind, base_val)
        write_artifact_type(
            out_dir, "full", family="scratch_gpt", scale=cfg.scale.name)
    else:
        write_adapter_base(out_dir, base_kind, base_val)
        write_artifact_type(
            out_dir, "adapter", family="scratch_gpt", scale=cfg.scale.name)


def write_adapter_base(adapter_dir, kind, value):
    # Remember which base this adapter sits on. Eval must never have to guess
    os.makedirs(adapter_dir, exist_ok=True)
    with open(os.path.join(adapter_dir, "adapter_base.json"), "w") as f:
        json.dump({"kind": kind, "value": value}, f)


def write_model_base(model_dir, kind, value):
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model_base.json"), "w") as f:
        json.dump({"kind": kind, "value": value}, f)


def resolve_adapter_base(adapter_dir, cfg):
    # Find the base model an adapter was trained on. Missing file is a hard error,
    # loading an adapter onto the wrong base would give silently wrong numbers
    p = os.path.join(adapter_dir, "adapter_base.json")
    if not os.path.exists(p):
        raise ValueError(f"no adapter_base.json in {adapter_dir}, cant tell which base this adapter belongs to")
    with open(p) as f:
        info = json.load(f)
    if info["kind"] == "hf_id":
        return info["value"]
    joined = os.path.join(cfg.output_root, info["value"])
    if os.path.isdir(joined):
        return joined
    if os.path.isdir(info["value"]):
        return info["value"]
    raise ValueError(f"merged base for {adapter_dir} not found, tried {joined} and {info['value']}")


def merge_adapter(cfg, adapter_dir, merged_dir):
    # bake an sft adapter into its base so grpo can start from the sft'd weights
    if os.path.exists(os.path.join(merged_dir, "config.json")):
        if is_scratch_gpt(cfg):
            from checkpoint import check_artifact_complete
            check_artifact_complete(merged_dir, expected="full")
        return merged_dir  # already merged, resume friendly
    from peft import PeftModel
    src = resolve_adapter_base(adapter_dir, cfg)
    if is_scratch_gpt(cfg):
        from checkpoint import check_artifact_complete, write_artifact_type
        record = check_artifact_complete(adapter_dir, expected="adapter")
        if record.get("family") != "scratch_gpt":
            raise ValueError(f"{adapter_dir}: adapter is not from the scratch GPT family")
        if record.get("scale") != cfg.scale.name:
            raise ValueError(f"{adapter_dir}: adapter is from a different scale")
        model, tok = load_scratch_model_and_tokenizer(
            cfg, src, dtype=merge_model_dtype(cfg))
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(src)
        model = AutoModelForCausalLM.from_pretrained(
            src, dtype=merge_model_dtype(cfg))
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    if is_scratch_gpt(cfg):
        with open(os.path.join(adapter_dir, "adapter_base.json")) as f:
            base = json.load(f)
        write_model_base(merged_dir, base["kind"], base["value"])
        write_artifact_type(
            merged_dir, "full", family="scratch_gpt", scale=cfg.scale.name)
    return merged_dir


def load_model_and_tokenizer(cfg, device, adapter_path=None, model_path=None,
                             dtype=None):
    if adapter_path and model_path:
        raise ValueError("adapter_path and model_path cannot be used together")
    # with an adapter the base comes from its provenance file, not from the config
    if model_path:
        src = model_path
    else:
        src = resolve_adapter_base(adapter_path, cfg) if adapter_path else resolve_model_id(cfg)
    if is_scratch_gpt(cfg):
        if adapter_path:
            from checkpoint import check_artifact_complete
            record = check_artifact_complete(adapter_path, expected="adapter")
            if record.get("family") != "scratch_gpt":
                raise ValueError(f"{adapter_path}: adapter is not from the scratch GPT family")
            if record.get("scale") != cfg.scale.name:
                raise ValueError(f"{adapter_path}: adapter is from a different scale")
        model, tok = load_scratch_model_and_tokenizer(
            cfg, src, dtype=dtype or resolve_dtype(cfg), device=device)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(src)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            src, dtype=dtype or resolve_dtype(cfg)).to(device)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path).to(device)
    return model, tok


def set_all_seeds(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
