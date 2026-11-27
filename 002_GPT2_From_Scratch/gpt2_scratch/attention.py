# Multi-head causal self-attention from scratch.
# The parallel multi-head implementation follows Raschka (2024), Section 3.6.2.
#
# The algorithm is standard scaled-dot-product attention plus three extras:
#   1. multiple heads computed in parallel via a reshape,
#   2. a causal mask (each token can only attend to earlier tokens + itself),
#   3. dropout on the attention weights as regularisation.
#
# This implementation also handles padding and a shared generation cache.
# Causal masking cannot be turned off because this package is only for language modelling.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
        layer_idx=0,
    ):
        super().__init__()
        # d_out must be divisible by num_heads so every head gets an equal slice
        if d_out % num_heads != 0:
            raise ValueError("attention width must be divisible by the number of heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.layer_idx = layer_idx

        # Q, K, V projections. For GPT-2 I pass qkv_bias=True at the
        # top-level config to match the original Conv1D layers in the checkpoint.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # the output projection mixes information back across the concatenated heads
        self.out_proj = nn.Linear(d_out, d_out)

        # dropout on the attention weights (NOT on the final outputs)
        self.dropout = nn.Dropout(dropout)

        # the causal mask: upper triangle (excluding the diagonal) is 1, rest is 0.
        # register_buffer so it follows the module to GPU via .to(device)
        # but doesnt show up in model.parameters(). persistent=False also keeps
        # this fixed tensor out of checkpoints because it is rebuilt from the config.
        mask = torch.triu(
            torch.ones(context_length, context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x,
        attention_mask=None,
        past_key_values=None,
        past_length=0,
        use_cache=False,
    ):
        # x: (batch, num_tokens, d_in)
        batch_size, query_length, _ = x.shape

        # project to Q, K, V. each result: (batch, num_tokens, d_out)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # split the last dim into (num_heads, head_dim), then move heads to
        # the 2nd dim so the attention calculation broadcasts across them.
        queries = queries.view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # each layer appends its new K and V tensors to the shared cache
        if use_cache:
            keys, values = past_key_values.update(
                keys, values, self.layer_idx
            )

        key_length = keys.shape[-2]
        query_positions = torch.arange(
            past_length,
            past_length + query_length,
            device=x.device,
        )
        key_positions = torch.arange(key_length, device=x.device)
        causal_mask = key_positions[None, :] <= query_positions[:, None]
        padding_is_clear = attention_mask is None or bool(attention_mask.all())
        direct_causal = past_length == 0 and padding_is_clear

        if attention_mask is not None and (
            attention_mask.ndim != 2 or attention_mask.shape[-1] != key_length
        ):
            raise ValueError("attention mask must cover the full key sequence")

        if direct_causal:
            combined_mask = None
        else:
            combined_mask = causal_mask[None, None]
            if attention_mask is not None:
                combined_mask = combined_mask & attention_mask[:, None, None, :].bool()

        # Scaled dot product attention scores every query against the allowed keys.
        # Direct causal path lets PyTorch use its fast built-in mask, while the
        # combined mask is needed for left padding and cached decoding.
        context = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=combined_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=direct_causal,
        )

        # bring heads back next to each token position and merge them into d_out
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, query_length, self.d_out)

        # final projection so heads can actually mix, rather than just concatenate.
        return self.out_proj(context)
