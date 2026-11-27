# Shape and sanity tests for MultiHeadAttention.
# These don't check numerical equivalence with any reference, they check
# structural things that tend to break silently during a refactor.
#
# The most important test is the causality one. If the mask is broken,
# everything else still passes but the model cheats by seeing the future.

import pytest
import torch

from gpt2_scratch.attention import MultiHeadAttention


def make_attention(context_length=8, dropout=0.0):
    return MultiHeadAttention(
        d_in=16,
        d_out=16,
        context_length=context_length,
        dropout=dropout,
        num_heads=2,
    )


def test_output_shape_and_values():
    # (batch=2, num_tokens=8, d=16) in -> (2, 8, 16) out
    torch.manual_seed(123)
    attention = make_attention()
    output = attention(torch.randn(2, 8, 16))
    assert output.shape == (2, 8, 16)

    # no NaNs or Infs coming out of attention
    assert torch.isfinite(output).all()


def test_width_must_be_divisible_by_heads():
    # classic bug: num_heads that doesn't divide d_out.
    # should fail loudly at construction time, not silently at forward.
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(16, 17, 8, 0.0, 2)


def test_earlier_tokens_do_not_see_later_tokens():
    # THE important test for ch03: changing a later token's input
    # must NOT change the output at an earlier position.
    # if the causal mask is broken, this fails.
    torch.manual_seed(123)
    attention = make_attention()
    attention.eval()
    values = torch.randn(1, 8, 16)
    first = attention(values)
    changed = values.clone()

    # perturb ONLY the last token
    changed[:, -1] += torch.randn(16) * 5.0
    second = attention(changed)

    # output at every position EXCEPT the last must be unchanged.
    # the last position attends to itself, so it is allowed to change.
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-6)

    # and sanity: the last position SHOULD change (otherwise attention is dead)
    assert not torch.allclose(first[:, -1], second[:, -1], atol=1e-6)


def test_padding_mask_keeps_outputs_finite():
    # real batches contain left padding, so the combined padding and causal
    # mask must still leave a valid attention path for every real token
    torch.manual_seed(123)
    attention = make_attention()
    values = torch.randn(2, 8, 16)
    mask = torch.tensor([
        [0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1],
    ])
    output = attention(values, attention_mask=mask)
    assert torch.isfinite(output).all()


def test_causal_mask_is_not_serialized():
    # the mask is fixed by context length and can be rebuilt on every load
    attention = make_attention()
    assert "mask" not in attention.state_dict()


def test_eval_is_deterministic_and_train_dropout_varies():
    # in eval mode (dropout off), same input -> same output.
    # in train mode with dropout>0, the two outputs should differ.
    torch.manual_seed(0)
    attention = make_attention(dropout=0.5)
    values = torch.randn(1, 8, 16)
    attention.eval()
    assert torch.allclose(attention(values), attention(values))
    attention.train()

    # shouldn't be exactly equal (there's a tiny chance this fails by luck,
    # but with 16*8 dropout draws at p=0.5, the chance is astronomically small)
    assert not torch.allclose(attention(values), attention(values))
