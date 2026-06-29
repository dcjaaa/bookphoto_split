"""
Backend process manager: start/stop/monitor the FastAPI server as a subprocess.

Usage:
    from scripts.gui.backend import BackendManager

    bm = BackendManager()
    bm.start()
    bm.wait_ready(timeout=10)  # block until /api/health returns 200
    ...
    bm.stop()
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

SERVER_MODULE = "scripts.api.server:app"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class BackendManager:
    """Manage the FastAPI backend lifecycle."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self._proc: subprocess.Popen | None = None
        self._registered_cleanup = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the backend. Kills any stale process on our port first."""
        if self.is_running():
            return

        self._kill_port_owner()

        project_root = str(
            Path(__file__).resolve().parent.parent.parent
        )
        cmd = [
            sys.executable, "-m", "uvicorn",
            SERVER_MODULE,
            "--host", self.host,
            "--port", str(self.port),
        ]
        self._proc = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not self._registered_cleanup:
            atexit.register(self._cleanup)
            self._registered_cleanup = True

    def _kill_port_owner(self) -> None:
        """If something is already listening on our port, kill it."""
        pids = self._find_pids_on_port()
        for pid in pids:
            # never kill ourselves
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.3)
                # if still alive, force-kill
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass  # already dead
            except (OSError, PermissionError):
                pass

    @staticmethod
    def _find_pids_on_port(port: int = DEFAULT_PORT) -> list[int]:
        """Return PIDs of processes listening on the given TCP port."""
        pids: list[int] = []

        # 1. find socket inode from /proc/net/tcp
        hex_port = f"{port:04X}"
        inodes: set[str] = set()
        try:
            with open("/proc/net/tcp") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 10:
                        continue
                    local = parts[1]  # HEXIP:HEXPORT, e.g. 0100007F:1F40
                    if local.endswith(f":{hex_port}"):
                        inodes.add(parts[9])
        except OSError:
            pass

        if not inodes:
            return pids

        # 2. scan /proc/*/fd for those socket inodes
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == os.getpid():
                continue
            try:
                fd_dir = f"/proc/{entry}/fd"
                for fd_name in os.listdir(fd_dir):
                    try:
                        link = os.readlink(f"{fd_dir}/{fd_name}")
                        for inode in inodes:
                            if f"socket:[{inode}]" in link:
                                pids.append(pid)
                                break
                    except OSError:
                        continue
            except (OSError, PermissionError):
                continue

        return pids

    def stop(self) -> None:
        """Terminate the backend process if running."""
        self._cleanup()

    def _cleanup(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except (subprocess.TimeoutExpired, Exception):
            try:
                proc.kill()
            except Exception:
                pass
        self._proc = None

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        """True if the subprocess is alive."""
        return self._proc is not None and self._proc.poll() is None

    def is_healthy(self, timeout: float = 2.0) -> bool:
        """True if GET /api/health returns 200."""
        try:
            r = httpx.get(f"{self.base_url}/api/health", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def wait_ready(self, timeout: float = 15.0) -> bool:
        """Block until the backend responds to /api/health, or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_healthy(timeout=2.0):
                return True
            if not self.is_running():
                return False  # process died
            time.sleep(0.3)
        return False

    def health_info(self) -> dict:
        """Return the /api/health JSON dict, or empty dict on failure."""
        try:
            r = httpx.get(f"{self.base_url}/api/health", timeout=2.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return {}
