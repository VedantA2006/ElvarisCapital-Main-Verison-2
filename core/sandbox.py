"""
core/sandbox.py – Safe execution of LLM-generated strategy code with process isolation.

Uses:
1. Strict AST policy validation before execution (sandbox.policy).
2. Process isolation via spawned child process (sandbox.runner).
3. Empty environment in child process (no MONGO_URL, no API secrets).
4. Locked-in audit hooks (sys.addaudithook) blocking network, subprocess, OS calls, file writes.
5. Incremental 1-bar-at-a-time feeding into bounded window with fresh deep copies.
6. Resource watchdog for wall-clock timeouts, per-bar timeouts, and RSS memory limits.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from core.signals import Action, Direction, Signal
from sandbox.policy import (
    BANNED_CALL_NAMES,
    PolicyResult,
    SAFE_BUILTINS,
    make_safe_builtins,
    validate_ast_policy,
)
from sandbox.pool import SandboxPool
from sandbox.runner import (
    IsolatedSandboxProcess,
    SandboxError,
    SandboxImportError,
    SandboxMemoryError,
    SandboxTimeoutError,
)

_log = logging.getLogger(__name__)

# Compatibility alias for blocked builtins
BLOCKED_BUILTINS = set(BANNED_CALL_NAMES) | {
    "open", "exec", "eval", "compile", "__import__",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr",
    "breakpoint", "exit", "quit",
    "input", "print", "memoryview", "bytearray",
}


def _make_safe_builtins() -> dict[str, Any]:
    return make_safe_builtins()


class SandboxStrategyWrapper:
    """Wraps an isolated child worker running a strategy."""

    def __init__(self, proc: IsolatedSandboxProcess):
        self._proc = proc
        self._inner = None  # Backward compatibility for exploit checks

    def on_bar(self, bars: pd.DataFrame) -> Signal | None:
        return self._proc.on_bar(bars)

    def close(self):
        if self._proc:
            self._proc.close()

    def __del__(self):
        try:
            self.close()
        except Exception as exc:
            _log.debug("Failed closing sandbox process in __del__: %s", exc)


class Sandbox:
    """Load and execute LLM-generated strategy code in an isolated process sandbox."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._sandbox_cfg = cfg.get("sandbox", {})
        self._strategy_cfg = cfg.get("strategy", {})
        self._wall_timeout = float(self._sandbox_cfg.get("wall_clock_timeout", 120.0))
        self._per_bar_timeout = float(self._sandbox_cfg.get("per_bar_timeout", 5.0))
        self._memory_limit_mb = float(self._sandbox_cfg.get("memory_limit_mb", 2048.0))
        self._max_code_len = int(self._strategy_cfg.get("max_code_length", 15000))
        self._allowed_imports = set(self._strategy_cfg.get("allowed_imports", ["numpy", "pandas", "numba", "math"]))
        self._pool = SandboxPool.get_instance(cfg)

    def validate_source(self, source: str) -> PolicyResult:
        """Run AST static policy checks. Returns scan result; does NOT raise."""
        return validate_ast_policy(
            source,
            allowed_imports=self._allowed_imports,
            max_code_len=self._max_code_len,
        )

    def load_strategy(
        self,
        source: str,
        strategy_class_name: str = "Strategy",
        params: dict | None = None,
        seed: int = 42,
        max_lookback: int = 1500,
    ) -> SandboxStrategyWrapper:
        """Validate code, acquire an isolated worker, and initialize strategy."""
        # 1. Static AST Policy Scan
        scan_res = self.validate_source(source)
        if not scan_res.passed:
            errors = [v for v in scan_res.violations if v.severity == "error"]
            detail = "; ".join(f"L{v.line}: {v.detail}" for v in errors[:5])
            raise SandboxError(f"Static scan failed: {detail}")

        # 2. Acquire sandbox process worker
        proc = self._pool.acquire()
        # Override timeouts/limits from instance config
        proc.wall_clock_timeout = self._wall_timeout
        proc.per_bar_timeout = self._per_bar_timeout
        proc.memory_limit_mb = self._memory_limit_mb

        # 3. Initialize strategy inside child
        try:
            proc.init_strategy(
                source,
                params=params,
                seed=seed,
                max_lookback=max_lookback,
            )
        except Exception:
            proc.close()
            raise

        return SandboxStrategyWrapper(proc)
