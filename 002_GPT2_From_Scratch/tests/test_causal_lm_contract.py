import importlib.util

import pytest
import torch

from gpt2_scratch import ScratchGPT2Config, ScratchGPT2ForCausalLM


def test_causal_mask_cannot_be_disabled():
    config = ScratchGPT2Config(
        vocab_size=101,
        n_positions=16,
        n_embd=32,
        n_layer=2,
        n_head=4,
    )
    model = ScratchGPT2ForCausalLM(config)
    assert all(not hasattr(block.att, "use_causal_mask") for block in model.trf_blocks)
    with pytest.raises(TypeError):
        ScratchGPT2Config(use_causal_mask=False)


def test_input_embeddings_and_labels_are_mutually_well_defined():
    config = ScratchGPT2Config(
        vocab_size=101,
        n_positions=16,
        n_embd=32,
        n_layer=2,
        n_head=4,
    )
    model = ScratchGPT2ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    embeddings = model.get_input_embeddings()(input_ids)
    with pytest.raises(ValueError):
        model(input_ids=input_ids, inputs_embeds=embeddings)


def test_package_has_no_classification_path():
    config = ScratchGPT2Config(
        vocab_size=101,
        n_positions=16,
        n_embd=32,
        n_layer=2,
        n_head=4,
    )
    model = ScratchGPT2ForCausalLM(config)
    assert model.get_output_embeddings().out_features == config.vocab_size
    assert importlib.util.find_spec("gpt2_scratch.dataset") is None
    assert all(
        not isinstance(module, torch.nn.Linear) or module.out_features != 2
        for module in model.modules()
    )
