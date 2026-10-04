"""
sandbox/runner.py: Manages isolated child process lifecycle, resource watchdog, and IPC.

Principle: The strategy runs in an isolated spawned child process with an empty environment,
fresh temporary working directory, and locked-in audit hooks. The parent feeds bars one by one
and enforces strict per-bar and cumulative timeouts as well as RSS memory limits.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing.connection import Listener
from typing import Any

import numpy as np
import pandas as pd
import psutil

from core.signals import Action, Direction, Signal
from sandbox.policy import validate_ast_policy

_log = logging.getLogger(__name__)


class SandboxError(RuntimeError):
    """Raised when strategy code violates sandbox rules or raises an exception."""
    pass


class SandboxTimeoutError(SandboxError):
    """Raised when strategy execution exceeds per-bar or wall-clock timeout."""
    pass


class SandboxMemoryError(SandboxError):
    """Raised when strategy execution exceeds configured memory limit."""
    pass


class SandboxImportError(SandboxError):
    """Raised when strategy attempts to import a forbidden module."""
    pass


def kill_process_tree(pid: int):
    """Terminate a process and all its descendants cleanly."""
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as ex:
                _log.debug("Process child kill: %s", ex)
        parent.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied) as ex:
        _log.debug("Parent process kill: %s", ex)


class IsolatedSandboxProcess:
    """Manages an isolated child process for running strategy execution."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        sb_cfg = cfg.get("sandbox", {})
        self.wall_clock_timeout = float(sb_cfg.get("wall_clock_timeout", 120.0))
        self.per_bar_timeout = float(sb_cfg.get("per_bar_timeout", 5.0))
        self.memory_limit_mb = float(sb_cfg.get("memory_limit_mb", 2048.0))
        self.max_code_len = int(cfg.get("strategy", {}).get("max_code_length", 15000))
        self.allowed_imports = set(cfg.get("strategy", {}).get("allowed_imports", ["numpy", "pandas", "numba", "math"]))

        self.tmp_dir: str | None = None
        self.proc: subprocess.Popen | None = None
        self.conn = None
        self.listener = None
        self.watchdog_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.memory_exceeded = threading.Event()

        self.cumulative_time: float = 0.0
        self.bars_fed: int = 0
        self.is_closed = False

    def spawn(self):
        """Spawn the isolated child process."""
        self.tmp_dir = tempfile.mkdtemp(prefix="qf_sbx_")
        authkey = secrets.token_bytes(16)

        # Ephemeral local port
        self.listener = Listener(("127.0.0.1", 0), family="AF_INET", authkey=authkey)
        try:
            self.listener._listener._socket.settimeout(10.0)
        except Exception as e:
            _log.debug("Listener socket settimeout: %s", e)
        port = self.listener.address[1]

        # Clean environment: ZERO engine secrets (no MONGO_URL, no API keys)
        clean_env = {
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
            "WINDIR": os.environ.get("WINDIR", r"C:\Windows"),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.path.abspath("."),
            "PYTHONUNBUFFERED": "1",
        }

        child_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "child.py"))
        cmd = [
            sys.executable,
            "-I",  # Isolated mode
            child_script,
            f"127.0.0.1:{port}",
            authkey.hex(),
        ]

        self.proc = subprocess.Popen(
            cmd,
            env=clean_env,
            cwd=self.tmp_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Accept connection from child
        try:
            self.conn = self.listener.accept()
            msg = self.conn.recv()
            if msg.get("status") != "ready":
                raise SandboxError(f"Child failed to initialize: {msg}")
        except Exception as e:
            stderr_out = ""
            if self.proc and self.proc.poll() is not None and self.proc.stderr:
                try:
                    stderr_out = self.proc.stderr.read().decode("utf-8", errors="replace")
                except Exception as ex:
                    _log.debug("Child stderr read error: %s", ex)
            self.close()
            raise SandboxError(f"Failed to establish child connection: {e}. {stderr_out}") from e
        finally:
            try:
                self.listener.close()
            except Exception as e:
                _log.debug("Listener close error: %s", e)

        # Start memory watchdog thread
        self.stop_event.clear()
        self.memory_exceeded.clear()
        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            args=(self.proc.pid, self.memory_limit_mb, self.stop_event, self.memory_exceeded),
            daemon=True,
        )
        self.watchdog_thread.start()

    @staticmethod
    def _watchdog_loop(pid: int, limit_mb: float, stop_ev: threading.Event, mem_ev: threading.Event):
        """Monitor child process RSS memory every 100ms."""
        while not stop_ev.is_set():
            try:
                p = psutil.Process(pid)
                rss_mb = p.memory_info().rss / (1024 * 1024)
                if rss_mb > limit_mb:
                    mem_ev.set()
                    kill_process_tree(pid)
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            stop_ev.wait(0.1)

    def init_strategy(
        self,
        code: str,
        params: dict | None = None,
        seed: int = 42,
        max_lookback: int = 1500,
    ):
        """Validate AST policy and initialize strategy inside child process."""
        # 1. Static AST Policy Scan
        scan_res = validate_ast_policy(
            code,
            allowed_imports=self.allowed_imports,
            max_code_len=self.max_code_len,
        )
        if not scan_res.passed:
            errors = [v for v in scan_res.violations if v.severity == "error"]
            detail = "; ".join(f"L{v.line}: {v.detail}" for v in errors[:5])
            raise SandboxError(f"Static scan failed: {detail}")

        # 2. Send initialization command
        init_payload = {
            "cmd": "init",
            "code": code,
            "params": params or {},
            "seed": seed,
            "max_lookback": max_lookback,
        }
        self.conn.send(init_payload)

        # 3. Wait for child initialization response
        if not self.conn.poll(self.wall_clock_timeout):
            self.close()
            raise SandboxTimeoutError(f"Strategy initialization exceeded timeout ({self.wall_clock_timeout}s)")

        reply = self.conn.recv()
        if reply.get("status") != "initialized":
            err_msg = reply.get("message", "Unknown error")
            tb = reply.get("traceback", "")
            self.close()
            raise SandboxError(f"Strategy initialization failed: {err_msg}\n{tb}")

    def on_bar(self, bars: pd.DataFrame) -> Signal | None:
        """Process incoming bars incrementally and return signal for latest bar."""
        if self.is_closed or self.proc is None:
            raise SandboxError("Sandbox child process is closed.")

        if self.memory_exceeded.is_set():
            self.close()
            raise SandboxMemoryError(
                f"Strategy exceeded memory limit of {self.memory_limit_mb} MB on bar {self.bars_fed}"
            )

        if self.proc.poll() is not None:
            stderr_out = ""
            if self.proc.stderr:
                try:
                    stderr_out = self.proc.stderr.read().decode("utf-8", errors="replace")
                except Exception as ex:
                    _log.debug("Stderr read error: %s", ex)
            self.close()
            raise SandboxError(
                f"Sandbox child terminated unexpectedly (code {self.proc.returncode}). {stderr_out[:200]}"
            )

        n_bars = len(bars)
        if n_bars == 0:
            return None

        # Determine which bars need to be fed
        if n_bars <= self.bars_fed:
            start_idx = n_bars - 1
        else:
            start_idx = self.bars_fed

        last_signal = None
        for i in range(start_idx, n_bars):
            row = bars.iloc[i]
            bar_dict = row.to_dict()
            ts = bar_dict.get("timestamp")
            if hasattr(ts, "isoformat"):
                bar_dict["timestamp"] = ts.isoformat()

            t0 = time.perf_counter()
            try:
                self.conn.send({"cmd": "bar", "bar": bar_dict})
            except Exception as e:
                self.close()
                raise SandboxError(f"Failed to communicate with sandbox child: {e}") from e

            # Effective timeout for this bar: bounded by remaining wall-clock time
            remaining_wall = max(0.05, self.wall_clock_timeout - self.cumulative_time)
            effective_timeout = min(self.per_bar_timeout, remaining_wall)

            if not self.conn.poll(effective_timeout):
                self.close()
                raise SandboxTimeoutError(
                    f"Strategy timeout (timed out) after {effective_timeout:.2f}s on bar {i}"
                )

            reply = self.conn.recv()
            dt = time.perf_counter() - t0
            self.cumulative_time += dt

            # Cumulative wall-clock timeout check
            if self.cumulative_time > self.wall_clock_timeout:
                self.close()
                raise SandboxTimeoutError(
                    f"Strategy exceeded wall-clock timeout: {self.cumulative_time:.1f}s > {self.wall_clock_timeout}s"
                )

            if reply.get("status") == "error":
                err_type = reply.get("error_type", "Error")
                msg = reply.get("message", "")
                tb = reply.get("traceback", "")
                self.close()
                raise SandboxError(f"Strategy error on bar {i}: {err_type}: {msg}\n{tb}")

            self.bars_fed += 1

            sig_data = reply.get("signal")
            if sig_data is None:
                last_signal = None
            else:
                act = Action(sig_data["action"]) if "action" in sig_data else Action.NONE
                raw_dir = sig_data.get("direction")
                direction = None
                if raw_dir is not None and str(raw_dir) != "None":
                    try:
                        direction = Direction(int(raw_dir)) if str(raw_dir).lstrip("-").isdigit() else Direction[str(raw_dir).upper()]
                    except Exception as ex:
                        _log.debug("Direction parse error: %s", ex)
                        direction = None

                def _safe_f(v):
                    if v is None:
                        return np.nan
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return np.nan

                sl = _safe_f(sig_data.get("stop_loss"))
                tp = _safe_f(sig_data.get("take_profit"))
                sl_dist = _safe_f(sig_data.get("sl_distance"))
                tp_dist = _safe_f(sig_data.get("tp_distance"))
                tag = str(sig_data.get("tag", ""))

                last_signal = Signal(
                    action=act,
                    direction=direction,
                    sl_distance=sl_dist,
                    tp_distance=tp_dist,
                    stop_loss=sl,
                    take_profit=tp,
                    tag=tag,
                )

        return last_signal

    def close(self):
        """Shut down the child process, stop watchdog, and clean up temporary directory."""
        if self.is_closed:
            return
        self.is_closed = True

        self.stop_event.set()
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            self.watchdog_thread.join(timeout=0.5)

        if self.conn:
            try:
                self.conn.send({"cmd": "close"})
                self.conn.close()
            except Exception as ex:
                _log.debug("Conn close error: %s", ex)

        if self.proc:
            try:
                kill_process_tree(self.proc.pid)
                self.proc.wait(timeout=1.0)
            except Exception as ex:
                _log.debug("Proc wait error: %s", ex)

        if self.tmp_dir and os.path.exists(self.tmp_dir):
            try:
                shutil.rmtree(self.tmp_dir, ignore_errors=True)
            except Exception as ex:
                _log.debug("Temp dir cleanup error: %s", ex)
