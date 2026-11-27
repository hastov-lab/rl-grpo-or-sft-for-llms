import numpy as np


def group_advantage(rewards):
    # standardise inside the group, if everyone got the same reward theres no signal
    r = np.asarray(rewards, dtype=float)
    std = r.std()
    if std == 0:
        return np.zeros_like(r)
    return (r - r.mean()) / std
