# Thin wrapper around tiktoken's GPT-2 BPE encoder.
# Following Raschka (2024) ch02 section 2.5.
#
# Why wrap it at all? Two reasons:
#   1. only one place in the code hardcodes "gpt2" as the encoder name,
#   2. PAD_TOKEN_ID and vocab_size live here as named attributes, so I
#      don't sprinkle magic numbers (50256, 50257) all over the codebase.
#
# The training pipeline uses GPT2TokenizerFast. This wrapper stays only
# as a parity reference for the tiktoken implementation.

import tiktoken


class Tokenizer:
    # GPT-2's tokenizer has NO dedicated pad token.
    # The wrapper uses token 50256 for padding.
    # Causal-LM batches must mask padded labels instead of treating them as targets.
    PAD_TOKEN_ID = 50256

    def __init__(self, encoding_name="gpt2"):
        # "gpt2" is the encoder that matches the pretrained weights. Dont change
        # this without also changing the weight loader and the tokenizer parity gate.
        self._enc = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self):
        # should be 50257 for gpt2 (50256 real tokens + 1 <|endoftext|>).
        return self._enc.n_vocab

    def encode(self, text):
        # Turn a string into a list of int token ids.
        # Special tokens still fail loudly so they cannot enter by accident.
        return self._enc.encode(text)

    def decode(self, token_ids):
        # Turn a list/tensor of int ids back into a string.
        # I cast to list() so numpy arrays and torch tensors both work.
        return self._enc.decode(list(token_ids))
