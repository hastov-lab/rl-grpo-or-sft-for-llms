# Model configs. Each dict can still be passed into the earlier GPTModel interface.
# Numbers taken from Raschka (2024) ch04 and the original GPT-2 paper.
# ScratchGPT2Config converts the same values into the standard Transformers fields.

from transformers import PretrainedConfig


GPT_CONFIG_124M = {
    "vocab_size":      50257,   # matches tiktoken's "gpt2" encoder (50256 + <|endoftext|>)
    "context_length":  1024,    # max positions the pretrained weights support
    "emb_dim":         768,     # hidden size
    "n_heads":         12,      # attention heads
    "n_layers":        12,      # number of transformer blocks
    "drop_rate":       0.1,     # official GPT-2 architecture dropout
    "qkv_bias":        True,    # OpenAI checkpoint has biases on Q,K,V projections
}

# 355M variant (gpt2-medium).
# Same shape formula as the other sizes; only emb_dim, n_heads, n_layers change.
GPT_CONFIG_355M = {
    "vocab_size":      50257,
    "context_length":  1024,
    "emb_dim":         1024,
    "n_heads":         16,
    "n_layers":        24,
    "drop_rate":       0.1,
    "qkv_bias":        True,
}

# 774M variant (gpt2-large).
GPT_CONFIG_774M = {
    "vocab_size":      50257,
    "context_length":  1024,
    "emb_dim":         1280,
    "n_heads":         20,
    "n_layers":        36,
    "drop_rate":       0.1,
    "qkv_bias":        True,
}

# 1500M variant (gpt2-xl).
# This configuration requires its own memory pilot before training.
GPT_CONFIG_1500M = {
    "vocab_size":      50257,
    "context_length":  1024,
    "emb_dim":         1600,
    "n_heads":         25,
    "n_layers":        48,
    "drop_rate":       0.1,
    "qkv_bias":        True,
}


# convenience lookup: maps the model_name string used by get_pretrained_gpt2()
# to the matching architecture config. keeps config-picking logic in one place.
_CONFIG_BY_NAME = {
    "gpt2":        GPT_CONFIG_124M,
    "gpt2-medium": GPT_CONFIG_355M,
    "gpt2-large":  GPT_CONFIG_774M,
    "gpt2-xl":     GPT_CONFIG_1500M,
}


def get_config(model_name):
    # return a COPY so callers can mutate it without affecting the module-level dict.
    if model_name not in _CONFIG_BY_NAME:
        raise ValueError(
            f"unknown model_name {model_name!r}. "
            f"supported: {sorted(_CONFIG_BY_NAME)}"
        )
    return _CONFIG_BY_NAME[model_name].copy()


class ScratchGPT2Config(PretrainedConfig):
    model_type = "gpt2_scratch"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        qkv_bias=True,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=50256,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if "use_causal_mask" in kwargs:
            raise TypeError("causal attention cannot be disabled")
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.qkv_bias = qkv_bias
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache

    @classmethod
    def from_scale(cls, model_name, **overrides):
        # convert the earlier dict names into the standard Transformers names
        old = get_config(model_name)
        values = {
            "vocab_size": old["vocab_size"],
            "n_positions": old["context_length"],
            "n_embd": old["emb_dim"],
            "n_head": old["n_heads"],
            "n_layer": old["n_layers"],
            "resid_pdrop": old["drop_rate"],
            "embd_pdrop": old["drop_rate"],
            "attn_pdrop": old["drop_rate"],
            "qkv_bias": old["qkv_bias"],
        }
        values.update(overrides)
        return cls(**values)
