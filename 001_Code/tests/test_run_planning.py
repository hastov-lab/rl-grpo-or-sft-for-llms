from config import RunConfig
from enums import Condition, Env, Mode, Scale
from scripts.run import plan_stages, base_eval_dir, build_summary


def test_plan_stages_table():
    assert plan_stages(Condition.SFT_GRPO) == ["base_eval", "sft", "sft_eval", "grpo", "grpo_eval"]
    assert plan_stages(Condition.BASE_GRPO) == ["base_eval", "grpo", "grpo_eval"]
    assert plan_stages(Condition.EXTERNAL_SFT_GRPO) == ["base_eval", "grpo", "grpo_eval"]
    assert plan_stages(Condition.MATCHED_LORA_GRPO) == ["base_eval", "grpo", "grpo_eval"]
    assert plan_stages(Condition.MATCHED_FULL_GRPO) == ["base_eval", "grpo", "grpo_eval"]
    assert plan_stages(Condition.SFT_ONLY) == ["base_eval", "sft", "sft_eval"]


def test_base_eval_dir_ignores_seed_and_condition():
    a = RunConfig(); a.train.seed = 0; a.condition = Condition.SFT_GRPO
    b = RunConfig(); b.train.seed = 1; b.condition = Condition.BASE_GRPO
    assert base_eval_dir(a) == base_eval_dir(b)


def test_base_eval_dir_differs_by_scale():
    a = RunConfig(); a.scale = Scale.B05
    b = RunConfig(); b.scale = Scale.B3
    assert base_eval_dir(a) != base_eval_dir(b)


def _ev(acc):
    return {"per_tier_acc": acc}


def test_summary_sft_grpo_delta_vs_sft():
    cfg = RunConfig(); cfg.condition = Condition.SFT_GRPO
    s = build_summary(cfg,
                      _ev({"easy": 0.2}), _ev({"easy": 0.4}), _ev({"easy": 0.5}),
                      _ev({"easy": 0.25}), _ev({"easy": 0.45}), _ev({"easy": 0.5}))
    assert s["reference"] == "sft"
    assert abs(s["grpo_delta"]["easy"] - 0.1) < 1e-9
    assert abs(s["grpo_logprob_delta"]["easy"] - 0.05) < 1e-9


def test_summary_base_grpo_delta_vs_base():
    cfg = RunConfig(); cfg.condition = Condition.BASE_GRPO
    s = build_summary(cfg, _ev({"easy": 0.2}), None, _ev({"easy": 0.5}),
                      _ev({"easy": 0.25}), None, _ev({"easy": 0.45}))
    assert s["reference"] == "base"
    assert s["sft_acc"] is None
    assert abs(s["grpo_delta"]["easy"] - 0.3) < 1e-9
    assert abs(s["grpo_logprob_delta"]["easy"] - 0.2) < 1e-9


def test_summary_sft_only_grpo_fields_null():
    cfg = RunConfig(); cfg.condition = Condition.SFT_ONLY
    s = build_summary(cfg, _ev({"easy": 0.2}), _ev({"easy": 0.4}), None,
                      _ev({"easy": 0.25}), _ev({"easy": 0.45}), None)
    assert s["grpo_acc"] is None
    assert s["grpo_delta"] is None
    assert s["grpo_logprob_delta"] is None
    assert s["sft_acc"] == {"easy": 0.4}


def test_fmt_secs():
    from scripts.run import _fmt_secs
    assert _fmt_secs(5) == "5s"
    assert _fmt_secs(65) == "1m 05s"
    assert _fmt_secs(3725) == "1h 02m 05s"


def test_fresh_wipes_smoke_dirs_only(tmp_path):
    import os
    from config import run_tag
    from scripts.run import wipe_run_dirs, base_eval_dir
    cfg = RunConfig()
    cfg.output_root = str(tmp_path)
    tag = run_tag(cfg)
    smoke_dirs = [tmp_path / "results" / tag, tmp_path / "checkpoints" / tag]
    full_dir = tmp_path / "results" / "COLAB_FULL_B05_sft_grpo_seed0"
    for d in smoke_dirs + [full_dir]:
        d.mkdir(parents=True)
    os.makedirs(base_eval_dir(cfg))
    wipe_run_dirs(cfg)
    assert not any(d.exists() for d in smoke_dirs)
    assert not os.path.isdir(base_eval_dir(cfg))
    assert full_dir.exists()   # never touches anything outside its own tag


def test_fresh_refuses_full_mode():
    import pytest
    from scripts.run import wipe_run_dirs
    cfg = RunConfig()
    cfg.mode = Mode.FULL
    with pytest.raises(SystemExit):
        wipe_run_dirs(cfg)
