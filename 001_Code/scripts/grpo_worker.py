# run one grpo stage in a disposable process so cuda memory is released on exit
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def save_status(path, payload):
    # replace the status atomically so an interrupted write cannot look complete
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def classify_failure(exc, torch_module=None):
    message = str(exc)
    cuda_oom_type = getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", None)
    is_cuda_oom = bool(cuda_oom_type and isinstance(exc, cuda_oom_type))
    low = message.lower()
    if "cuda" in low and "out of memory" in low:
        is_cuda_oom = True
    return {
        "error_type": type(exc).__name__,
        "error_message": message[:2000],
        "cuda_oom": is_cuda_oom,
    }


def blocked_resume_reason(attempts):
    if not attempts:
        return None
    last = attempts[-1]
    if not last.get("resume_blocked"):
        return None
    if last.get("error_type") == "KeyboardInterrupt":
        return None
    failure = "cuda oom" if last.get("cuda_oom") else "worker failure"
    return f"previous {failure} at step {last.get('end_step', 'unknown')} requires review before resume"


def active_worker_processes(config_path):
    import subprocess

    ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True)
    marker = f"scripts/grpo_worker.py --config {config_path}"
    return [line.strip() for line in ps.stdout.splitlines() if marker in line]


def stream_subprocess(cmd):
    import errno
    import pty
    import signal
    import subprocess

    master_fd, slave_fd = pty.openpty()
    try:
        try:
            proc = subprocess.Popen(cmd, stdout=slave_fd, stderr=slave_fd, close_fds=True)
        finally:
            os.close(slave_fd)
        try:
            while True:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            proc.wait()
            return proc
        except KeyboardInterrupt:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            raise
    finally:
        os.close(master_fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--report-to", nargs="*", default=[])
    args = ap.parse_args()

    import torch

    from config import load_config
    from obs import gpu_snapshot
    from scripts.env_switch import apply_smoke, resolve_device
    from train.grpo import run_grpo

    cfg = apply_smoke(load_config(args.config))
    device = resolve_device(cfg.env)
    torch.cuda.reset_peak_memory_stats()
    status = {"completed": False, "gpu": gpu_snapshot()}
    try:
        run_grpo(
            cfg,
            device,
            adapter_path=args.adapter,
            out_dir=args.out_dir,
            report_to=args.report_to,
            log_dir=args.log_dir,
            run_name=args.run_name,
            allow_oom_fallback=cfg.exec.allow_oom_fallback,
        )
        status["completed"] = True
    except BaseException as exc:
        status.update(classify_failure(exc, torch))
        raise
    finally:
        status["peak_vram_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        save_status(args.status, status)


if __name__ == "__main__":
    main()
