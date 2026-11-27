# Loading pretrained GPT-2 weights into my from-scratch GPTModel.
#
# Attribution:
# The weight-name mapping pattern (which source tensor goes into which
# GPTModel parameter, with what transpose and which shape) is ADAPTED from
# Sebastian Raschka's "Build a Large Language Model (From Scratch)"
# (Manning, 2024), chapter 5 section 5.5.
# (github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/)
#
# I kept the mapping structure close to the earlier version because a silent
# mismapping here gives a runnable model that outputs garbage. This
# version adds complete source key accounting and verifies the tied head alias.
#
# One deliberate change from Raschka: I download weights from Hugging Face
# instead of the original TensorFlow chekpoint. Reasons:
#   (a) transformers is already part of the training environment,
#   (b) the state-dict keys are readable and debuggable,
#   (c) avoiding tensorflow saves a large dependency,
#   (d) the checkpoint revision and file hashes can be pinned directly.

import torch


REFERENCE_ALIAS = "gpt2"
REFERENCE_CHECKPOINTS = {
    "gpt2": (
        "openai-community/gpt2",
        "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    ),
    "gpt2-medium": (
        "openai-community/gpt2-medium",
        "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
    ),
    "gpt2-large": (
        "openai-community/gpt2-large",
        "32b71b12589c2f8d625668d2335a01cac3249519",
    ),
    "gpt2-xl": (
        "openai-community/gpt2-xl",
        "15ea56dee5df4983c59b2538573817e1667135e2",
    ),
}
REFERENCE_MODEL_ID, REFERENCE_REVISION = REFERENCE_CHECKPOINTS[REFERENCE_ALIAS]


def reference_checkpoint(model_name):
    if model_name not in REFERENCE_CHECKPOINTS:
        raise ValueError(
            f"unknown model_name {model_name!r}. "
            f"supported: {sorted(REFERENCE_CHECKPOINTS)}"
        )
    return REFERENCE_CHECKPOINTS[model_name]


def _assign(target, source, name):
    # Copy `source` into `target` inplace, but only if shapes agree.
    # Explicit check beats torch's default message, which can be hard to
    # track down when a transpose is missing.
    if target.shape != source.shape:
        raise ValueError(
            f"shape mismatch for {name}: "
            f"target {tuple(target.shape)}, source {tuple(source.shape)}"
        )
    with torch.no_grad():
        target.copy_(source.detach())


def _source(state_dict, key, accounted):
    # record every source key at the point where the mapping consumes it
    if key not in state_dict:
        raise KeyError(f"missing source tensor: {key}")
    accounted.add(key)
    return state_dict[key]


def load_weights_into_gpt(model, hf_state_dict):
    # Hugging Face key naming (for reference):
    #   transformer.wte.weight           -> tok_emb.weight
    #   transformer.wpe.weight           -> pos_emb.weight
    #   transformer.h.{i}.ln_1.*         -> trf_blocks[i].norm1.*
    #   transformer.h.{i}.attn.c_attn.*  -> trf_blocks[i].att.W_{q,k,v}.*
    #   transformer.h.{i}.attn.c_proj.*  -> trf_blocks[i].att.out_proj.*
    #   transformer.h.{i}.ln_2.*         -> trf_blocks[i].norm2.*
    #   transformer.h.{i}.mlp.c_fc.*     -> trf_blocks[i].ff.layers[0].*
    #   transformer.h.{i}.mlp.c_proj.*   -> trf_blocks[i].ff.layers[2].*
    #   transformer.ln_f.*               -> final_norm.*
    accounted = set()

    # ### embeddings ##########################################################
    source_wte = _source(hf_state_dict, "transformer.wte.weight", accounted)
    _assign(model.tok_emb.weight, source_wte, "tok_emb.weight")
    _assign(
        model.pos_emb.weight,
        _source(hf_state_dict, "transformer.wpe.weight", accounted),
        "pos_emb.weight",
    )

    if "lm_head.weight" in hf_state_dict:
        # GPT-2 ties lm_head.weight to wte.weight. Source head is recorded
        # as an alias rather than copied into a second param.
        source_head = _source(hf_state_dict, "lm_head.weight", accounted)
        if not torch.equal(source_head, source_wte):
            raise ValueError("source language-model head is not tied to the token embedding")

    # ### per block ###########################################################
    for block_index, block in enumerate(model.trf_blocks):
        prefix = f"transformer.h.{block_index}"

        # LayerNorm 1 (before attention)
        _assign(
            block.norm1.scale,
            _source(hf_state_dict, f"{prefix}.ln_1.weight", accounted),
            f"block {block_index} norm1 scale",
        )
        _assign(
            block.norm1.shift,
            _source(hf_state_dict, f"{prefix}.ln_1.bias", accounted),
            f"block {block_index} norm1 shift",
        )

        combined_weight = _source(
            hf_state_dict, f"{prefix}.attn.c_attn.weight", accounted
        )
        combined_bias = _source(
            hf_state_dict, f"{prefix}.attn.c_attn.bias", accounted
        )

        # Attention Q, K, V: Hugging Face stores a combined c_attn of shape
        # (d, 3d) using the Conv1D convention (in, out). This model has three
        # separate nn.Linear layers with the torch convention (out, in), so the
        # tensor is split along dim=1 and transposed.
        query_weight, key_weight, value_weight = torch.split(
            combined_weight, block.att.d_out, dim=1
        )
        query_bias, key_bias, value_bias = torch.split(
            combined_bias, block.att.d_out, dim=0
        )
        _assign(
            block.att.W_query.weight,
            query_weight.T,
            f"block {block_index} query weight",
        )
        _assign(
            block.att.W_key.weight,
            key_weight.T,
            f"block {block_index} key weight",
        )
        _assign(
            block.att.W_value.weight,
            value_weight.T,
            f"block {block_index} value weight",
        )
        _assign(
            block.att.W_query.bias,
            query_bias,
            f"block {block_index} query bias",
        )
        _assign(
            block.att.W_key.bias,
            key_bias,
            f"block {block_index} key bias",
        )
        _assign(
            block.att.W_value.bias,
            value_bias,
            f"block {block_index} value bias",
        )

        # Attention output projection: same Conv1D -> Linear transpose story
        _assign(
            block.att.out_proj.weight,
            _source(hf_state_dict, f"{prefix}.attn.c_proj.weight", accounted).T,
            f"block {block_index} attention output weight",
        )
        _assign(
            block.att.out_proj.bias,
            _source(hf_state_dict, f"{prefix}.attn.c_proj.bias", accounted),
            f"block {block_index} attention output bias",
        )

        # LayerNorm 2 (before FFN)
        _assign(
            block.norm2.scale,
            _source(hf_state_dict, f"{prefix}.ln_2.weight", accounted),
            f"block {block_index} norm2 scale",
        )
        _assign(
            block.norm2.shift,
            _source(hf_state_dict, f"{prefix}.ln_2.bias", accounted),
            f"block {block_index} norm2 shift",
        )

        # FeedForward uses Sequential(Linear, GELU, Linear, Dropout), so the
        # two weighted layers are still at index 0 and index 2.
        _assign(
            block.ff.layers[0].weight,
            _source(hf_state_dict, f"{prefix}.mlp.c_fc.weight", accounted).T,
            f"block {block_index} feed-forward input weight",
        )
        _assign(
            block.ff.layers[0].bias,
            _source(hf_state_dict, f"{prefix}.mlp.c_fc.bias", accounted),
            f"block {block_index} feed-forward input bias",
        )
        _assign(
            block.ff.layers[2].weight,
            _source(hf_state_dict, f"{prefix}.mlp.c_proj.weight", accounted).T,
            f"block {block_index} feed-forward output weight",
        )
        _assign(
            block.ff.layers[2].bias,
            _source(hf_state_dict, f"{prefix}.mlp.c_proj.bias", accounted),
            f"block {block_index} feed-forward output bias",
        )

    # ### final LayerNorm #####################################################
    _assign(
        model.final_norm.scale,
        _source(hf_state_dict, "transformer.ln_f.weight", accounted),
        "final_norm.scale",
    )
    _assign(
        model.final_norm.shift,
        _source(hf_state_dict, "transformer.ln_f.bias", accounted),
        "final_norm.shift",
    )

    # ### output head: GPT-2 ties lm_head to wte ##############################
    model.tie_weights()

    unaccounted = sorted(set(hf_state_dict) - accounted)
    # A new or misspelled source key must fail instead of being silently dropped.
    if unaccounted:
        raise ValueError(f"unaccounted source tensors: {unaccounted}")
    if model.out_head.weight is not model.tok_emb.weight:
        raise ValueError("output and token-embedding weights are not tied")
    return {
        "accounted_source_keys": sorted(accounted),
        "alias": {"lm_head.weight": "transformer.wte.weight"},
    }


def get_pretrained_gpt2(
    model_name=REFERENCE_ALIAS,
    revision=None,
    local_files_only=False,
):
    # Download GPT-2 from Hugging Face and return its state_dict.
    # model_name: "gpt2" (=124M), "gpt2-medium" (=355M), etc.
    # Importing transformers here keeps the mapping helpers independently testable.
    from transformers import GPT2LMHeadModel

    model_id, pinned_revision = reference_checkpoint(model_name)
    if revision is None:
        revision = pinned_revision
    model = GPT2LMHeadModel.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
        dtype=torch.float32,
    )
    state_dict = dict(model.state_dict())

    # I only need the state_dict; discard the reference model to free memory
    del model
    return state_dict
