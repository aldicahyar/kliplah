import io
import os
import subprocess
import sys
import time

from scripts import dev


def _wait_dead(pid: int, timeout_ms: int = 3000) -> bool:
    if os.name == "nt":
        import ctypes

        SYNCHRONIZE = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            return True
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(handle, timeout_ms) == 0
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(0.05)
    return False


def test_prefix_lines_tags_every_line():
    out = io.StringIO()
    dev.prefix_lines(io.StringIO("INFO mulai\r\nbaris dua\nakhir tanpa newline"), "api", out)
    assert out.getvalue() == "[api] INFO mulai\n[api] baris dua\n[api] akhir tanpa newline\n"


def test_commands_run_api_and_worker_from_backend():
    assert dev.COMMANDS["api"][1:] == ["-m", "uvicorn", "app.api.main:create_app", "--factory",
                                       "--host", "127.0.0.1", "--port", "8000"]
    assert dev.COMMANDS["worker"][1:] == ["-m", "app.worker.main"]
    assert (dev.BACKEND_DIR / "app" / "cli.py").is_file()


def test_stop_terminates_child_process_tree(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(c.pid), encoding='utf-8'); "
        "print('ready', flush=True); "
        "time.sleep(30)"
    )
    parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert parent.stdout.readline().strip() == "ready"
        child_pid = int(child_pid_file.read_text(encoding="utf-8"))
        dev._stop([parent])
        assert parent.poll() is not None
        assert _wait_dead(child_pid)
    finally:
        if parent.poll() is None:
            dev._stop([parent])
