# Tests for the _assign helper used during weight loading.
# The full mapping now has a separate numerical parity test against the pinned
# Hugging Face checkpoint, while these tests keep the small failure cases local.

import pytest
import torch

from gpt2_scratch.weights import REFERENCE_CHECKPOINTS, _assign, reference_checkpoint


def test_reference_checkpoints_cover_each_model_size():
    assert set(REFERENCE_CHECKPOINTS) == {
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
    }
    for model_name, (model_id, revision) in REFERENCE_CHECKPOINTS.items():
        assert model_id.rsplit("/", 1)[-1] == model_name
        assert len(revision) == 40
        assert reference_checkpoint(model_name) == (model_id, revision)


def test_reference_checkpoint_rejects_an_unknown_size():
    with pytest.raises(ValueError, match="unknown model_name"):
        reference_checkpoint("gpt2-xxl")


def test_assign_copies_values_without_sharing_storage():
    # after _assign, modifying target should NOT modify source.
    # this catches the bug where we accidentally share storage.
    target = torch.zeros(3, 4)
    source = torch.randn(3, 4)
    original = source.clone()
    _assign(target, source, "test")
    assert torch.allclose(target, source)
    target.fill_(99.0)

    # source must be untouched
    assert torch.allclose(source, original)


def test_assign_names_a_shape_mismatch():
    # I want a LOUD failure on the offending name, not a cryptic torch error.
    # checking the exception message confirms the name survives.
    target = torch.zeros(3, 4)
    source = torch.randn(3, 5)
    with pytest.raises(ValueError, match="block 7 query weight"):
        _assign(target, source, "block 7 query weight")
