import json
import sys


def test_worker_failure_classification_distinguishes_cuda_from_cpu_oom():
    from scripts.grpo_worker import classify_failure

    cuda = classify_failure(RuntimeError("CUDA out of memory while allocating tensor"))
    assert cuda["error_type"] == "RuntimeError"
    assert cuda["cuda_oom"] is True

    cpu = classify_failure(MemoryError("out of memory"))
    assert cpu["error_type"] == "MemoryError"
    assert cpu["cuda_oom"] is False


def test_resume_guard_blocks_worker_failure_but_allows_runtime_interruption():
    from scripts.grpo_worker import blocked_resume_reason

    interrupted = [{"interrupted": True, "end_step": 300}]
    assert blocked_resume_reason(interrupted) is None

    manual = [{"resume_blocked": True, "error_type": "KeyboardInterrupt", "end_step": 350}]
    assert blocked_resume_reason(manual) is None

    failed = [{"resume_blocked": True, "cuda_oom": True, "end_step": 400}]
    assert blocked_resume_reason(failed) == "previous cuda oom at step 400 requires review before resume"


def test_stream_subprocess_forwards_worker_output(capsys):
    from scripts.grpo_worker import stream_subprocess

    proc = stream_subprocess([sys.executable, "-u", "-c", "print('worker line', flush=True)"])
    assert proc.returncode == 0
    assert "worker line" in capsys.readouterr().out


def test_worker_status_replaces_an_existing_file(tmp_path):
    from scripts.grpo_worker import save_status

    path = tmp_path / "worker_status.json"
    path.write_text('{"completed": false}')
    save_status(str(path), {"completed": True, "peak_vram_gib": 76.4})
    assert json.loads(path.read_text()) == {"completed": True, "peak_vram_gib": 76.4}
    assert not (tmp_path / "worker_status.json.tmp").exists()
