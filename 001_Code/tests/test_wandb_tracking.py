import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from config import RunConfig
from obs import configure_wandb, prepare_wandb, wandb_finish


def _fake_wandb(monkeypatch, run=None):
    module = types.ModuleType("wandb")
    module.run = run
    module.finished = 0

    def finish():
        module.finished += 1
        module.run = None

    module.finish = finish
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def test_config_overrides_a_stale_wandb_project(monkeypatch):
    cfg = RunConfig()
    monkeypatch.setenv("WANDB_PROJECT", "inm363")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    state = configure_wandb(cfg, ["wandb"])

    assert os.environ["WANDB_PROJECT"] == cfg.log.wandb_project
    assert os.environ["WANDB_MODE"] == cfg.log.wandb_mode
    assert state == {"enabled": True, "key_source": "env"}


def test_each_training_stage_closes_its_own_wandb_run(monkeypatch):
    module = _fake_wandb(monkeypatch, SimpleNamespace(id="stale"))
    cfg = RunConfig()
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    state = prepare_wandb(cfg, ["wandb"])
    assert state["enabled"] is True
    assert module.finished == 1

    module.run = SimpleNamespace(id="current")
    wandb_finish()
    assert module.finished == 2


def test_future_notebooks_cannot_restore_the_old_project_assignment():
    historical = {
        "018_colab_ext_seed1.ipynb",
        "019_colab_ext_seed2.ipynb",
        "020_colab_ext_seed0.ipynb",
        "021_colab_scratch1000.ipynb",
        "022_colab_b7_smooth1000.ipynb",
    }
    old_assignments = (
        "os.environ.setdefault('WANDB_PROJECT', 'inm363')",
        'os.environ.setdefault("WANDB_PROJECT", "inm363")',
    )
    for path in Path("notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        if any(value in source for value in old_assignments):
            assert path.name in historical, f"old wandb project assignment in {path.name}"
