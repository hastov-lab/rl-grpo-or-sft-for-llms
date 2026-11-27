import pytest
import torch
from transformers import GPT2LMHeadModel

from gpt2_scratch import (
    ScratchGPT2Config,
    ScratchGPT2ForCausalLM,
    load_weights_into_gpt,
)
from gpt2_scratch.weights import REFERENCE_ALIAS, REFERENCE_REVISION


@pytest.fixture(scope="session")
def pretrained_pair():
    reference = GPT2LMHeadModel.from_pretrained(
        REFERENCE_ALIAS,
        revision=REFERENCE_REVISION,
        # pinned revision and no download, so a silent upstream change cant move
        # the numbers every parity gate is measured against
        local_files_only=True,
        # float32 both sides. the 1e-4 and 1e-5 tolerances assume it, bfloat16
        # would miss them by an order of magnitude
        dtype=torch.float32,
    ).eval()
    scratch = ScratchGPT2ForCausalLM(
        ScratchGPT2Config.from_scale("gpt2")
    # eval() on both, otherwise the 0.1 dropout fires and no two forward passes agree, let alone two diferent models
    ).eval()
    coverage = load_weights_into_gpt(scratch, reference.state_dict())
    return reference, scratch, coverage
