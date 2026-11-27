import os

import pytest

from config import load_config
from train.sft import resolve_sft_start


def test_no_chain_uses_hf_base(tmp_path):
    cfg = load_config("configs/smoke_local.yaml")
    cfg.output_root = str(tmp_path)
    src, kind, val = resolve_sft_start(cfg)
    assert kind == "hf_id"
    assert src == val  # base model id straight through, nothing merged


def test_chain_with_missing_adapter_fails_loudly(tmp_path):
    cfg = load_config("configs/008_wsft_colab_b05.yaml")
    cfg.output_root = str(tmp_path)  # empty dir, so the stage-1 adapter is absent
    with pytest.raises(FileNotFoundError):
        resolve_sft_start(cfg)


def test_chain_merges_into_shared_dir(tmp_path, monkeypatch):
    # The success path: adapter present, merge gets called once with the shared
    # per-scale dir and the provenance points there. Merge itself is stubbed so
    # the test stays offline
    import train.sft as sft_mod
    cfg = load_config("configs/008_wsft_colab_b05.yaml")
    cfg.output_root = str(tmp_path)
    adir = tmp_path / cfg.train.sft_from_adapter
    adir.mkdir(parents=True)
    (adir / "adapter_model.safetensors").touch()
    calls = []
    monkeypatch.setattr(sft_mod, "merge_adapter", lambda c, a, m: calls.append((a, m)) or m)
    src, kind, val = resolve_sft_start(cfg)
    assert kind == "local"
    source_tag = adir.parent.name
    assert val == f"checkpoints/{cfg.experiment}_{cfg.scale.name}_{source_tag}_merged_base"
    assert src.endswith(val)
    assert len(calls) == 1 and calls[0][0] == str(adir)


def test_chain_reuses_the_parent_merged_base(tmp_path, monkeypatch):
    import train.sft as sft_mod
    cfg = load_config("configs/008_wsft_colab_b05.yaml")
    cfg.output_root = str(tmp_path)
    adir = tmp_path / cfg.train.sft_from_adapter
    adir.mkdir(parents=True)
    (adir / "adapter_model.safetensors").touch()
    merged = adir.parent / "sft_merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    monkeypatch.setattr(sft_mod, "merge_adapter", lambda *args: pytest.fail("unexpected merge"))

    src, kind, val = resolve_sft_start(cfg)

    assert src == str(merged)
    assert kind == "local"
    assert val == os.path.relpath(merged, tmp_path)


def test_008_configs_parse_with_chain_field():
    for p in ("configs/008_wsft_colab_b05.yaml", "configs/008_wsft_colab_b3.yaml"):
        cfg = load_config(p)
        assert cfg.train.sft_from_adapter.startswith("checkpoints/")
        assert cfg.train.sft_data == "mmlu_mc"
