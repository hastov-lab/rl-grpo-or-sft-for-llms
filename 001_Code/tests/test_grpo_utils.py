import numpy as np

from train.grpo_utils import group_advantage


def test_zero_variance_skipped():
    adv = group_advantage([1.0, 1.0, 1.0])
    assert np.allclose(adv, 0.0)


def test_standardised():
    adv = group_advantage([0.0, 1.0])
    assert np.isclose(adv.mean(), 0.0)
    assert np.isclose(adv.std(), 1.0)
