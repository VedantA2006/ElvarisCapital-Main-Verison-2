"""
sandbox/pool.py: Pool of pre-warmed isolated child processes.

Maintains pre-warmed sandbox workers to avoid Python startup and module import overhead
on every strategy run. Children are strictly discarded after each run (no cross-run state).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any

from sandbox.runner import IsolatedSandboxProcess

_log = logging.getLogger(__name__)


class SandboxPool:
    """Pool of pre-warmed IsolatedSandboxProcess workers."""

    _instance: SandboxPool | None = None
    _lock = threading.Lock()

    def __init__(self, cfg: dict):
        self._cfg = cfg
        sb_cfg = cfg.get("sandbox", {})
        default_size = max(1, (os.cpu_count() or 2) - 1)
        self.pool_size = int(sb_cfg.get("pool_size", default_size))
        self._queue: queue.Queue[IsolatedSandboxProcess] = queue.Queue(maxsize=self.pool_size)
        self._closing = False
        self._replenish_thread = threading.Thread(target=self._replenish_loop, daemon=True)
        self._replenish_thread.start()

    @classmethod
    def get_instance(cls, cfg: dict) -> SandboxPool:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cfg)
            return cls._instance

    def _replenish_loop(self):
        """Background thread keeping the pool filled with pre-warmed workers."""
        while not self._closing:
            try:
                if self._queue.qsize() < self.pool_size:
                    proc = IsolatedSandboxProcess(self._cfg)
                    proc.spawn()
                    self._queue.put(proc, timeout=1.0)
                else:
                    threading.Event().wait(0.2)
            except Exception as e:
                _log.debug("Replenish worker creation exception: %s", e)
                threading.Event().wait(0.5)

    def acquire(self) -> IsolatedSandboxProcess:
        """Acquire a pre-warmed sandbox worker or spawn one on demand."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            proc = IsolatedSandboxProcess(self._cfg)
            proc.spawn()
            return proc

    def shutdown(self):
        """Shut down the pool and terminate all pooled workers."""
        self._closing = True
        while not self._queue.empty():
            try:
                proc = self._queue.get_nowait()
                proc.close()
            except Exception as e:
                _log.debug("Pool shutdown close error: %s", e)
