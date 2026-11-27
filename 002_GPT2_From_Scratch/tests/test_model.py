# Shape and sanity tests for the ch04 building blocks.
# These check structural things that silently break during a refactor:
# shapes, normalisation stats, gradient flow, and agreement with torch
# built-ins where that is meaningful.

import torch

from gpt2_scratch.configuration import (
    GPT_CONFIG_124M,
    ScratchGPT2Config,
    get_config,
)
from gpt2_scratch.model import GELU, LayerNorm, ScratchGPT2ForCausalLM


def tiny_config(**values):
    settings = {
        "vocab_size": 101,
        "n_positions": 16,
        "n_embd": 32,
        "n_layer": 2,
        "n_head": 4,
    }
    settings.update(values)
    return ScratchGPT2Config(**settings)


def test_layernorm_output_stats():
    # after LN, mean should be ~0 and std ~1 along the last dim (since
    # scale is initialised to 1 and shift to 0).
    torch.manual_seed(0)
    layer = LayerNorm(16)
    values = torch.randn(4, 8, 16) * 100 + 50
    output = layer(values)
    assert torch.allclose(output.mean(dim=-1), torch.zeros(4, 8), atol=1e-5)

    # unbiased=False to match my LayerNorm implementation
    assert torch.allclose(
        output.std(dim=-1, unbiased=False),
        torch.ones(4, 8),
        atol=1e-4,
    )


def test_gelu_matches_torch_reference():
    # my GELU uses the tanh approximation, same as torch.nn.GELU(approximate="tanh").
    # they should be numerically identical within float32 tolerance.
    values = torch.randn(100) * 5
    expected = torch.nn.GELU(approximate="tanh")(values)
    assert torch.allclose(GELU()(values), expected, atol=1e-5)


def test_forward_returns_logits_loss_and_gradients():
    # the headline test: (batch=2, seq=8) ints in -> (2, 8, 101) logits out
    torch.manual_seed(0)
    model = ScratchGPT2ForCausalLM(tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    assert output.logits.shape == (2, 8, model.config.vocab_size)

    # no NaN/Inf coming out of a fresh random model
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.loss)

    # calling .backward() should populate .grad on at least one parameter.
    # catches accidental detach(), no_grad(), or a broken graph.
    output.loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )


def test_official_architecture_values_are_pinned():
    # the config gate catches a model that matches shapes but not GPT-2's
    # actual dropout, normalisation or activation settings
    config = ScratchGPT2Config.from_scale("gpt2")
    assert config.vocab_size == 50257
    assert config.n_positions == 1024
    assert config.n_embd == 768
    assert config.n_layer == 12
    assert config.n_head == 12
    assert config.activation_function == "gelu_new"
    assert config.layer_norm_epsilon == 1e-5
    assert config.embd_pdrop == 0.1
    assert config.attn_pdrop == 0.1
    assert config.resid_pdrop == 0.1
    assert config.qkv_bias is True


def test_earlier_config_dict_stays_available():
    assert GPT_CONFIG_124M["emb_dim"] == 768
    assert GPT_CONFIG_124M["n_heads"] == 12
    assert GPT_CONFIG_124M["n_layers"] == 12
    copied = get_config("gpt2")
    copied["emb_dim"] = 1
    assert GPT_CONFIG_124M["emb_dim"] == 768


def test_124m_parameter_count_and_tied_head():
    # equal values are not enough: these two paths must share one parameter
    model = ScratchGPT2ForCausalLM(ScratchGPT2Config.from_scale("gpt2"))
    assert sum(parameter.numel() for parameter in model.parameters()) == 124439808
    assert model.out_head.weight is model.tok_emb.weight
    assert model.out_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_shifted_loss_ignores_masked_labels():
    # causal LM predicts the next token, and -100 keeps padded or prompt labels
    # out of the loss without removing real EOS targets
    torch.manual_seed(0)
    model = ScratchGPT2ForCausalLM(tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    labels = input_ids.clone()
    labels[0, :3] = -100
    output = model(input_ids=input_ids, labels=labels, use_cache=False)
    expected = torch.nn.functional.cross_entropy(
        output.logits[:, :-1].reshape(-1, model.config.vocab_size),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(output.loss, expected)


def test_logits_to_keep_projects_only_requested_positions():
    torch.manual_seed(0)
    model = ScratchGPT2ForCausalLM(tiny_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    full = model(input_ids=input_ids, use_cache=False).logits

    projected_shapes = []
    hook = model.out_head.register_forward_pre_hook(
        lambda module, inputs: projected_shapes.append(inputs[0].shape)
    )
    kept = model(
        input_ids=input_ids,
        use_cache=False,
        logits_to_keep=2,
    ).logits
    hook.remove()

    assert projected_shapes == [torch.Size([2, 2, model.config.n_embd])]
    assert torch.allclose(kept, full[:, -2:], atol=1e-6)

    full[:, -2:].float().square().mean().backward()
    full_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    reduced = model(
        input_ids=input_ids,
        use_cache=False,
        logits_to_keep=2,
    ).logits
    reduced.float().square().mean().backward()
    for name, parameter in model.named_parameters():
        if name in full_gradients:
            assert torch.allclose(parameter.grad, full_gradients[name], atol=1e-6)


def test_logits_to_keep_keeps_full_projection_for_loss():
    torch.manual_seed(0)
    model = ScratchGPT2ForCausalLM(tiny_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    baseline = model(input_ids=input_ids, labels=input_ids, use_cache=False)

    projected_shapes = []
    hook = model.out_head.register_forward_pre_hook(
        lambda module, inputs: projected_shapes.append(inputs[0].shape)
    )
    kept = model(
        input_ids=input_ids,
        labels=input_ids,
        use_cache=False,
        logits_to_keep=2,
    )
    hook.remove()

    assert projected_shapes == [torch.Size([2, 8, model.config.n_embd])]
    assert torch.allclose(kept.loss, baseline.loss)
    assert torch.allclose(kept.logits, baseline.logits[:, -2:])
