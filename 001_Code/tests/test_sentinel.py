import pytest
from types import SimpleNamespace

from obs import (a1_truncation, a2_immobility, a2_verdict_payload,
                 make_grpo_sentinel, projected_hours, next_fallback, oom_ladder)


def test_a1_fires_on_heavy_truncation():
    assert a1_truncation([0.5] * 50) is True
    assert a1_truncation([0.1] * 50) is False
    assert a1_truncation([0.9] * 10) is False   # not enough steps yet


def test_a1_uses_the_restored_clipped_history():
    from config import RunConfig

    cfg = RunConfig()
    prior = [{"completions/clipped_ratio": 1.0, "step": step}
             for step in range(1, 50)]
    callback = make_grpo_sentinel(cfg, prior_history=prior)
    state = SimpleNamespace(global_step=50)
    control = SimpleNamespace()
    with pytest.raises(RuntimeError, match="sentinel abort: a1"):
        callback.on_log(
            None, state, control,
            logs={"completions/clipped_ratio": 1.0})


def test_a2_fires_only_when_flat_and_pinned():
    assert a2_immobility(0.25, 0.255, 0.001) is True
    assert a2_immobility(0.25, 0.35, 0.001) is False   # reward moved
    assert a2_immobility(0.25, 0.255, 0.05) is False   # policy moved


def test_a2_record_payload_uses_the_registered_windows():
    from config import RunConfig

    cfg = RunConfig()
    cfg.experiment = "031"
    cfg.train.a2_action = "record"
    rows = [{"step": step,
             "rewards/accuracy_reward/mean": 0.7 if step <= 50 else 0.69,
             "kl": 0.001} for step in range(1, 151)]
    record = a2_verdict_payload(cfg, rows)
    assert record["early_accuracy_reward_mean"] == pytest.approx(0.7)
    assert record["mid_accuracy_reward_mean"] == pytest.approx(0.69)
    assert record["would_fire"] is True


def test_projection():
    assert projected_hours([36.0] * 10, 500) == pytest.approx(5.0)


def test_fallback_ladder_order():
    assert next_fallback(8, 1, False) == (4, 2, False)
    assert next_fallback(4, 2, False) == (2, 4, False)
    assert next_fallback(1, 8, False) == (1, 8, True)
    assert next_fallback(1, 8, True) is None


def test_oom_ladder_walks_then_succeeds():
    calls = []

    def attempt(per_device, accum, checkpointing):
        calls.append((per_device, accum, checkpointing))
        if len(calls) < 3:
            raise RuntimeError("CUDA out of memory")
        return "trained"

    assert oom_ladder(attempt, 8, 1, False) == "trained"
    assert calls == [(8, 1, False), (4, 2, False), (2, 4, False)]


def test_oom_ladder_reraises_other_errors():
    def attempt(per_device, accum, checkpointing):
        raise ValueError("something else broke")

    with pytest.raises(ValueError):
        oom_ladder(attempt, 8, 1, False)


def test_oom_ladder_gives_up_at_the_bottom():
    def attempt(per_device, accum, checkpointing):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError):
        oom_ladder(attempt, 1, 8, True)
