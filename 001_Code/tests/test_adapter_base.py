import json
import os

import pytest

from config import RunConfig
from train.model_utils import write_adapter_base, resolve_adapter_base


def _cfg(root):
    cfg = RunConfig()
    cfg.output_root = str(root)
    return cfg


def test_hf_id_passes_through(tmp_path):
    d = str(tmp_path / "adapter")
    write_adapter_base(d, "hf_id", "Qwen/Qwen2.5-0.5B")
    assert resolve_adapter_base(d, _cfg(tmp_path)) == "Qwen/Qwen2.5-0.5B"


def test_local_joins_output_root(tmp_path):
    d = str(tmp_path / "adapter")
    merged = tmp_path / "checkpoints" / "x" / "sft_merged"
    merged.mkdir(parents=True)
    write_adapter_base(d, "local", os.path.join("checkpoints", "x", "sft_merged"))
    assert resolve_adapter_base(d, _cfg(tmp_path)) == str(merged)


def test_local_falls_back_to_raw_path(tmp_path):
    # joined path is missing but the stored value itself exists as a dir
    d = str(tmp_path / "adapter")
    raw = tmp_path / "elsewhere" / "sft_merged"
    raw.mkdir(parents=True)
    other_root = tmp_path / "not_the_root"
    other_root.mkdir()
    write_adapter_base(d, "local", str(raw))
    assert resolve_adapter_base(d, _cfg(other_root)) == str(raw)


def test_local_missing_everywhere_raises(tmp_path):
    d = str(tmp_path / "adapter")
    write_adapter_base(d, "local", os.path.join("checkpoints", "gone", "sft_merged"))
    with pytest.raises(ValueError):
        resolve_adapter_base(d, _cfg(tmp_path))


def test_missing_json_raises(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    with pytest.raises(ValueError):
        resolve_adapter_base(str(d), _cfg(tmp_path))


def test_write_round_trip(tmp_path):
    d = str(tmp_path / "adapter")
    write_adapter_base(d, "hf_id", "gpt2")
    with open(os.path.join(d, "adapter_base.json")) as f:
        info = json.load(f)
    assert info == {"kind": "hf_id", "value": "gpt2"}
