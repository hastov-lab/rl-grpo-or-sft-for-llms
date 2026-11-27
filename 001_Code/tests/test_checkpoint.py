from checkpoint import save_state, load_state, save_metrics, load_metrics


def test_state_roundtrip(tmp_path):
    path = str(tmp_path / "state.pkl")
    state = {"w": [1, 2, 3], "step": 10}
    save_state(path, state)
    assert load_state(path) == state


def test_metrics_roundtrip(tmp_path):
    path = str(tmp_path / "m.json")
    m = {"acc": 0.42, "tier": "hard"}
    save_metrics(path, m)
    assert load_metrics(path) == m


def test_has_checkpoint(tmp_path):
    from checkpoint import has_checkpoint
    d = tmp_path / "run"
    d.mkdir()
    assert has_checkpoint(str(d)) is False
    (d / "checkpoint-50").mkdir()
    assert has_checkpoint(str(d)) is True


def test_config_hash_roundtrip(tmp_path):
    from config import RunConfig
    from checkpoint import write_config_hash, check_config_hash
    d = str(tmp_path)
    write_config_hash(d, RunConfig())
    check_config_hash(d, RunConfig())  # same config, no raise


def test_config_hash_mismatch(tmp_path):
    import pytest
    from config import RunConfig
    from enums import Scale
    from checkpoint import write_config_hash, check_config_hash
    d = str(tmp_path)
    write_config_hash(d, RunConfig())
    other = RunConfig()
    other.scale = Scale.B3
    with pytest.raises(ValueError):
        check_config_hash(d, other)


def test_config_hash_ignores_eval_changes(tmp_path):
    # eval and log tweaks must not block a training resume
    from config import RunConfig
    from checkpoint import write_config_hash, check_config_hash
    d = str(tmp_path)
    write_config_hash(d, RunConfig())
    other = RunConfig()
    other.eval.batch_size = 99
    other.log.save_steps = 7
    check_config_hash(d, other)  # no raise


def test_config_hash_catches_lr_change(tmp_path):
    import pytest
    from config import RunConfig
    from checkpoint import write_config_hash, check_config_hash
    d = str(tmp_path)
    write_config_hash(d, RunConfig())
    other = RunConfig()
    other.train.grpo_lr = 5.0e-5
    with pytest.raises(ValueError):
        check_config_hash(d, other)


def test_config_hash_accepts_the_historical_schemas(tmp_path):
    # checkpoints written before the neutral chain fields existed must still
    # resume, and only while those fields are empty now
    import pytest
    from config import RunConfig
    from checkpoint import check_config_hash, config_hash

    cfg = RunConfig()

    # a stored hash from the era before grpo_from_adapter
    d14 = str(tmp_path / "era14")
    import os
    os.makedirs(d14)
    with open(os.path.join(d14, "config_hash.txt"), "w") as f:
        f.write(config_hash(cfg, ("grpo_from_adapter",)))
    check_config_hash(d14, cfg)  # no raise

    # a stored hash from the era before both chain fields
    d07 = str(tmp_path / "era07")
    os.makedirs(d07)
    with open(os.path.join(d07, "config_hash.txt"), "w") as f:
        f.write(config_hash(cfg, ("grpo_from_adapter", "sft_from_adapter")))
    check_config_hash(d07, cfg)  # no raise

    # non-empty chain fields disable the allowance
    chained = RunConfig()
    chained.train.grpo_from_adapter = "checkpoints/x/sft"
    with pytest.raises(ValueError):
        check_config_hash(d14, chained)
    with pytest.raises(ValueError):
        check_config_hash(d07, chained)

    # any real training difference still fails in every era
    changed = RunConfig()
    changed.train.grpo_lr *= 2
    with pytest.raises(ValueError):
        check_config_hash(d14, changed)
    with pytest.raises(ValueError):
        check_config_hash(d07, changed)

    # a changed seed still fails
    reseeded = RunConfig()
    reseeded.train.seed = 7
    with pytest.raises(ValueError):
        check_config_hash(d07, reseeded)


def test_default_a2_action_preserves_hash_and_record_changes_it():
    from config import RunConfig
    from checkpoint import _config_hash, config_hash

    cfg = RunConfig()
    assert _config_hash is config_hash
    assert config_hash(cfg) == config_hash(cfg, ("a2_action",))
    cfg.train.a2_action = "record"
    assert config_hash(cfg) != config_hash(cfg, ("a2_action",))
