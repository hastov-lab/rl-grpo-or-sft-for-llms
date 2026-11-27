import os

from config import RunConfig
from obs import resolve_report_to, write_manifest, gpu_snapshot


def test_resolve_report_to_only_keeps_installed():
    # whatever survives must be a known tracker that is actually importable,
    # so this holds whether or not tensorboard/wandb are installed
    import importlib.util as u
    cfg = RunConfig()
    cfg.log.trackers = ("tensorboard", "wandb")
    out = resolve_report_to(cfg)
    assert set(out).issubset({"tensorboard", "wandb"})
    for t in out:
        if t == "tensorboard":
            assert u.find_spec("tensorboard") or u.find_spec("tensorboardX")
        if t == "wandb":
            assert u.find_spec("wandb")


def test_gpu_snapshot_cpu_fallback():
    snap = gpu_snapshot()
    assert "device" in snap and "peak_vram_gb" in snap


def test_write_manifest(tmp_path):
    man = write_manifest(RunConfig(), str(tmp_path))
    assert os.path.exists(os.path.join(str(tmp_path), "manifest.json"))
    assert man["run_tag"] == "000_LOCAL_SMOKE_B05_sft_grpo_seed0"
    assert "versions" in man and "git" in man


def test_read_key_file(tmp_path):
    from obs import read_key_file
    p = tmp_path / "k.txt"
    p.write_text("secret123\n")
    assert read_key_file(str(p)) == "secret123"
    assert read_key_file(str(tmp_path / "missing.txt")) is None
