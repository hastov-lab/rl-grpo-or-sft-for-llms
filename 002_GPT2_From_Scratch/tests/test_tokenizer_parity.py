from transformers import GPT2TokenizerFast

from gpt2_scratch.tokenizer import Tokenizer
from gpt2_scratch.weights import REFERENCE_ALIAS, REFERENCE_REVISION


def test_legacy_and_standard_tokenizers_match_on_fixed_prompts():
    standard = GPT2TokenizerFast.from_pretrained(
        REFERENCE_ALIAS,
        revision=REFERENCE_REVISION,
        local_files_only=True,
    )
    legacy = Tokenizer()
    prompts = (
        "Hello, world.",
        "Question: Which option is correct?\nAnswer:",
        "A train travels 120 miles in 3 hours.",
        # the probe alphabet. the leading space matters, gpt-2 gives "A" and
        # " A" different ids and the probe scores the spaced form
        " A B C D",
    )
    for prompt in prompts:
        expected = standard.encode(prompt, add_special_tokens=False)
        assert legacy.encode(prompt) == expected
        assert legacy.decode(expected) == standard.decode(expected)


def test_standard_tokenizer_keeps_gpt2_vocabulary_and_eos():
    tokenizer = GPT2TokenizerFast.from_pretrained(
        REFERENCE_ALIAS,
        revision=REFERENCE_REVISION,
        local_files_only=True,
    )
    assert tokenizer.vocab_size == 50257
    assert tokenizer.eos_token_id == 50256
