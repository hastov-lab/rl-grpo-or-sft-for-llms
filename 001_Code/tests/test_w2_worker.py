# The worker only dispatches, so a wrong branch or a missing start must fail
# before any training begins.
import json
import os

import pytest

from config import RunConfig
from scripts.w2_worker import _latest_checkpoint, train_branch, verify_adapter_start


def test_training_dispatches_fixed_rows_and_start(monkeypatch, tmp_path):
    import train.grpo
    import train.sft

    cfg = RunConfig()
    cfg.experiment = "034_smoke"
    cfg.output_root = str(tmp_path)
    cfg.train.sft_data = "synthetic_facts"
    bundle = {"sft_rows": [{"prompt": "p", "target": " t"}],
              "grpo_rows": [{"prompt": "q", "gold": "A", "tier": "synthetic"}]}
    start = {"merged_dir": os.path.join(str(tmp_path), "start")}
    calls = []
    monkeypatch.setattr(train.sft, "run_sft_rows", lambda *args, **kwargs: (
        calls.append(("sft", args, kwargs)) or kwargs["out_dir"]))
    monkeypatch.setattr(train.grpo, "run_grpo_rows", lambda *args, **kwargs: (
        calls.append(("grpo", args, kwargs)) or kwargs["out_dir"]))

    for branch in ("sft", "grpo"):
        train_branch(cfg, "cpu", branch, bundle, start, f"/{branch}", "/logs",
                     branch, ["wandb"])
    assert [call[0] for call in calls] == ["sft", "grpo"]
    assert calls[0][1][2] == bundle["sft_rows"]
    assert calls[1][1][2] == bundle["grpo_rows"]
    assert calls[0][1][3] == start["merged_dir"]
    assert calls[1][1][3] == start["merged_dir"]


def test_adapter_evaluation_requires_the_bound_start(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_base.json").write_text(json.dumps(
        {"kind": "local", "value": "different"}))
    with pytest.raises(RuntimeError, match="different W2 start"):
        verify_adapter_start(str(adapter), {"base_value": "expected"})


def test_latest_checkpoint_ignores_unrelated_names(tmp_path):
    (tmp_path / "checkpoint-25").mkdir()
    (tmp_path / "checkpoint-50").mkdir()
    (tmp_path / "checkpoint-old").mkdir()
    assert _latest_checkpoint(str(tmp_path)) == 50
