"""Development runner: the API (uvicorn, 127.0.0.1:8000) and the worker as two subprocesses.

    uv run python scripts/dev.py

Output lines are prefixed `[api]` / `[worker]`. Ctrl+C, or either process exiting, stops both.
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO

BACKEND_DIR = Path(__file__).resolve().parents[1]
EOL = "\r\n"
COMMANDS = {
    "api": [sys.executable, "-m", "uvicorn", "app.api.main:create_app", "--factory",
            "--host", "127.0.0.1", "--port", "8000"],
    "worker": [sys.executable, "-m", "app.worker.main"],
}


def prefix_lines(stream: IO[str], tag: str, out: IO[str]) -> None:
    """Copy every line of `stream` to `out` as `[tag] line` until the stream ends."""
    for line in stream:
        out.write(f"[{tag}] {line.rstrip(EOL)}\n")
        out.flush()


def _start(tag: str, cmd: list[str]) -> tuple[subprocess.Popen, threading.Thread]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(cmd, cwd=BACKEND_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    pump = threading.Thread(target=prefix_lines, args=(proc.stdout, tag, sys.stdout), daemon=True)
    pump.start()
    return proc, pump


def _terminate_tree(proc: subprocess.Popen) -> None:
    """Terminate `proc` and on Windows its descendant processes (e.g. running ffmpeg/yt-dlp)."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return
        except OSError:
            pass
    proc.terminate()


def _stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            _terminate_tree(p)
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def main() -> int:
    started = [_start(tag, cmd) for tag, cmd in COMMANDS.items()]
    procs = [p for p, _ in started]
    code = 0
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
        code = next(p.returncode for p in procs if p.returncode is not None)
        print(f"[dev] salah satu proses berhenti (kode {code}); menghentikan yang lain", flush=True)
    except KeyboardInterrupt:
        print("[dev] berhenti", flush=True)
    finally:
        _stop(procs)
        for _, pump in started:
            pump.join(timeout=2)
    return code


if __name__ == "__main__":
    sys.exit(main())
