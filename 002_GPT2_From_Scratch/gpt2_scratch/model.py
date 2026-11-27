# The GPT-2 architecture in one file: LayerNorm, GELU, FeedForward,
# TransformerBlock, and ScratchGPT2ForCausalLM. Following Raschka (2024) ch04.
#
# The layers and transformer block share one file to avoid a circular import.
#
# The transformer block adapts the layout in Raschka (2024), Chapter 4.
#
# This implementation keeps the language model head and adds the narrow
# Transformers interface needed for causal-LM training, caching and safe reload.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .attention import MultiHeadAttention
from .configuration import ScratchGPT2Config


# ########## 4.2 LayerNorm ##################################################
class LayerNorm(nn.Module):
    # The original GPT-2 uses learned scale and shift parameters. Their copied
    # names are kept because the pretrained-weight mapping depends on them.
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # normalise along the feature (last) dimension only.
        # F.layer_norm matches the official GPT-2 calculation used by the parity gate.
        return F.layer_norm(
            x,
            (self.scale.shape[0],),
            self.scale,
            self.shift,
            self.eps,
        )


# ########### 4.3 GELU activation #############################################
class GELU(nn.Module):
    # GPT-2 uses the tanh approximation of GELU (faster than the erf version
    # and what the original paper used). torch.nn.GELU(approximate="tanh")
    # would do the same thing, but having the formula written out makes it
    # easy to verify I got it right.
    def forward(self, x):
        return 0.5 * x * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            )
        )


# ########## 4.3 FeedForward ##################################################
class FeedForward(nn.Module):
    # Two-layer MLP with expansion factor 4 (GPT-2 standard).
    # For GPT-2 small: 768 -> 3072 -> 768.
    def __init__(self, config):
        super().__init__()
        inner_size = config.n_inner or 4 * config.n_embd
        self.layers = nn.Sequential(
            nn.Linear(config.n_embd, inner_size),
            GELU(),
            nn.Linear(inner_size, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        return self.layers(x)


# ########## 4.4-4.5 TransformerBlock #########################################
class TransformerBlock(nn.Module):
    # Pre-LayerNorm block (LN BEFORE attention/feedforward, not after).
    # Pre-LN is more stable to train than post-LN; GPT-2 uses pre-LN.
    # Two residual connections wrap the two sub-layers.
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config.n_embd,
            d_out=config.n_embd,
            context_length=config.n_positions,
            dropout=config.attn_pdrop,
            num_heads=config.n_head,
            qkv_bias=config.qkv_bias,
            layer_idx=layer_idx,
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config.n_embd, config.layer_norm_epsilon)
        self.norm2 = LayerNorm(config.n_embd, config.layer_norm_epsilon)
        self.drop_shortcut = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        x,
        attention_mask=None,
        past_key_values=None,
        past_length=0,
        use_cache=False,
    ):
        # sub-layer 1: attention wrapped by a residual
        shortcut = x
        x = self.norm1(x)
        x = self.att(
            x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            past_length=past_length,
            use_cache=use_cache,
        )
        x = shortcut + self.drop_shortcut(x)

        # sub-layer 2: feed-forward wrapped by a residual
        shortcut = x
        x = self.norm2(x)
        x = shortcut + self.ff(x)
        return x


# ########## 4.6 GPTModel #################################################
class ScratchGPT2ForCausalLM(PreTrainedModel, GenerationMixin):
    # Top-level architecture. Pipeline:
    #
    #   (batch, seq) token ids
    #     -> token embeddings + positional embeddings
    #     -> embedding dropout
    #     -> stack of n_layers TransformerBlocks
    #     -> final LayerNorm
    #     -> tied vocabulary head
    config_class = ScratchGPT2Config
    base_model_prefix = ""
    main_input_name = "input_ids"
    _tied_weights_keys = {"out_head.weight": "tok_emb.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.n_positions, config.n_embd)
        self.drop_emb = nn.Dropout(config.embd_pdrop)
        self.trf_blocks = nn.ModuleList(
            TransformerBlock(config, layer_idx)
            for layer_idx in range(config.n_layer)
        )
        self.final_norm = LayerNorm(config.n_embd, config.layer_norm_epsilon)

        # bias=False: classic GPT-2 design. The pretrained weights expect this.
        self.out_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

        # language model head and token embedding remain one param
        self.tie_weights()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            nn.init.ones_(module.scale)
            nn.init.zeros_(module.shift)

    def get_input_embeddings(self):
        return self.tok_emb

    def set_input_embeddings(self, value):
        self.tok_emb = value

    def get_output_embeddings(self):
        return self.out_head

    def set_output_embeddings(self, value):
        self.out_head = value

    def forward_hidden(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
    ):
        # Returns hidden state AFTER final_norm but BEFORE out_head.
        # Shape: (batch, seq_len, emb_dim).
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass input ids or input embeddings, but not both")

        use_cache = self.config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError("past key values must use the Transformers cache interface")
        if use_cache and past_key_values is None:
            # Start one shared cache so every block appends at its own layer index.
            past_key_values = DynamicCache(config=self.config)

        past_length = (
            past_key_values.get_seq_length()
            if past_key_values is not None
            else 0
        )
        if inputs_embeds is None:
            inputs_embeds = self.tok_emb(input_ids)
        batch_size, seq_length = inputs_embeds.shape[:2]
        total_length = past_length + seq_length
        if total_length > self.config.n_positions:
            raise ValueError("sequence length exceeds the configured context")

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, total_length):
                raise ValueError("attention mask has the wrong shape")
            if position_ids is None:
                # Left padding must not move the first real token away from position zero.
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids[:, -seq_length:]
        elif position_ids is None:
            # positional embedding: same rows for every item in the batch.
            # torch.arange gives the positions added after any cached prefix.
            position_ids = torch.arange(
                past_length,
                total_length,
                device=inputs_embeds.device,
            ).unsqueeze(0)

        position_embeds = self.pos_emb(position_ids)
        hidden_states = self.drop_emb(inputs_embeds + position_embeds)
        for block in self.trf_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                past_length=past_length,
                use_cache=use_cache,
            )
        return self.final_norm(hidden_states), past_key_values if use_cache else None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        num_items_in_batch=None,
        output_attentions=None,
        output_hidden_states=None,
        tokenizer=None,
        return_dict=None,
        **kwargs,
    ):
        if output_attentions or output_hidden_states:
            raise NotImplementedError("attention and hidden-state outputs are not exposed")
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported forward arguments: {names}")
        return_dict = self.config.return_dict if return_dict is None else return_dict
        hidden_states, next_cache = self.forward_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        # avoid the full vocabulary projection when the caller only needs a slice
        if labels is None:
            selected_hidden = hidden_states
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                selected_hidden = hidden_states[:, -logits_to_keep:, :]
            elif torch.is_tensor(logits_to_keep):
                selected_hidden = hidden_states[:, logits_to_keep, :]
            logits = self.out_head(selected_hidden)

        loss = None
        if labels is not None:
            # Full pipeline: hidden state + out_head projection.
            # Returns logits (batch, seq_len, vocab_size).
            full_logits = self.out_head(hidden_states)
            # Each position predicts the following token, while -100 labels stay ignored.
            shift_logits = full_logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            reduction = "sum" if num_items_in_batch is not None else "mean"
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction=reduction,
            )
            if num_items_in_batch is not None:
                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.to(loss.device)
                loss = loss / num_items_in_batch
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                logits = full_logits[:, -logits_to_keep:, :]
            elif torch.is_tensor(logits_to_keep):
                logits = full_logits[:, logits_to_keep, :]
            else:
                logits = full_logits

        if not return_dict:
            values = (logits, next_cache)
            return ((loss,) + values) if loss is not None else values
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        next_sequence_length=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self,
            input_ids,
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if attention_mask is not None and model_inputs.get("position_ids") is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            sequence_length = model_inputs["input_ids"].shape[-1]
            model_inputs["position_ids"] = position_ids[:, -sequence_length:]
        return model_inputs


GPTModel = ScratchGPT2ForCausalLM
