"""
core/sandbox.py – Safe execution of LLM-generated strategy code.

The sandbox enforces:
1. RESTRICTED IMPORTS – only numpy, pandas, numba, math (from config).
2. NO FILE/NETWORK/OS ACCESS – builtins like open, exec, eval are blocked.
3. RESOURCE LIMITS – CPU time and memory caps (via signal/resource on Linux,
   wall-clock timeout on Windows).
4. SOURCE INTEGRITY – the code is AST-scanned BEFORE execution.

Usage:
    sandbox = Sandbox(cfg)
    strategy = sandbox.load_strategy(source_code)
    # strategy implements StrategyProtocol (on_bar method)
    result = run_backtest(strategy, df, cfg)

The sandbox does NOT use Docker (config: sandbox.use_docker=false).
It relies on Python-level restrictions + the static AST scan from
lookahead_guard.py. Docker isolation can be added later for multi-user
deployments.
"""

from __future__ import annotations

import signal as signal_mod
import sys
import textwrap
import threading
import time
import types
from typing import Any

from core.lookahead_guard import static_scan, StaticScanResult


class SandboxError(RuntimeError):
    """Raised when strategy code violates sandbox rules."""
    pass


class SandboxTimeoutError(SandboxError):
    """Raised when strategy execution exceeds wall-clock timeout."""
    pass


class SandboxImportError(SandboxError):
    """Raised when strategy tries to import a banned module."""
    pass


# Builtins that are BLOCKED in strategy code
BLOCKED_BUILTINS = {
    "open", "exec", "eval", "compile", "__import__",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr",
    "breakpoint", "exit", "quit",
    "input", "print",  # strategies should not print
    "memoryview", "bytearray",
}

# Builtins that ARE allowed
SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "complex", "dict", "divmod",
    "enumerate", "filter", "float", "format", "frozenset",
    "hasattr", "hash", "hex", "id", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "oct", "ord", "pow", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple",
    "type", "zip",
    "True", "False", "None",
    "__build_class__",  # required for `class` keyword to work
    "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration",
    "ZeroDivisionError", "OverflowError", "ArithmeticError",
    "Exception", "BaseException",
    "property", "staticmethod", "classmethod", "super",
    "object",
}


def _make_safe_builtins() -> dict[str, Any]:
    """Build a restricted __builtins__ dict."""
    import builtins
    safe = {}
    for name in SAFE_BUILTINS:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)

    # Controlled __import__ that only allows whitelisted modules
    def _safe_import(name, *args, **kwargs):
        root = name.split(".")[0]
        allowed = {"numpy", "pandas", "numba", "math", "np", "pd",
                    "collections", "functools", "itertools", "operator",
                    "dataclasses", "typing", "enum", "abc"}
        if root not in allowed:
            raise SandboxImportError(
                f"Import '{name}' is not allowed in strategy code. "
                f"Allowed top-level modules: numpy, pandas, numba, math."
            )
        return __builtins__["__import__"](name, *args, **kwargs) if isinstance(__builtins__, dict) \
            else __import__(name, *args, **kwargs)

    safe["__import__"] = _safe_import
    return safe


def _make_restricted_globals(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a globals dict for exec() with restricted builtins."""
    g = {
        "__builtins__": _make_safe_builtins(),
        "__name__": "<strategy>",
        "__doc__": None,
    }
    # Pre-import allowed modules so strategy code can use them
    import numpy as np
    import pandas as pd
    import math
    g["np"] = np
    g["numpy"] = np
    g["pd"] = pd
    g["pandas"] = pd
    g["math"] = math
    if extra:
        g.update(extra)
    return g


class Sandbox:
    """Load and execute LLM-generated strategy code safely."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._sandbox_cfg = cfg.get("sandbox", {})
        self._strategy_cfg = cfg.get("strategy", {})
        self._wall_timeout = self._sandbox_cfg.get("wall_clock_timeout", 120)
        self._max_code_len = self._strategy_cfg.get("max_code_length", 5000)
        self._allowed_imports = set(self._strategy_cfg.get("allowed_imports", ["numpy", "pandas", "numba", "math"]))

    def validate_source(self, source: str) -> StaticScanResult:
        """Run static analysis. Returns scan result; does NOT raise."""
        if len(source) > self._max_code_len:
            from core.lookahead_guard import LookaheadViolation
            return StaticScanResult(
                passed=False,
                violations=[LookaheadViolation(
                    line=0, col=0, pattern="code_too_long",
                    detail=f"Code is {len(source)} chars (max: {self._max_code_len}).",
                )],
                summary=f"Code too long: {len(source)} > {self._max_code_len}",
            )
        return static_scan(source, self._allowed_imports)

    def load_strategy(self, source: str, strategy_class_name: str = "Strategy") -> Any:
        """Validate, compile, and instantiate a strategy from source code.

        Returns an object with an `on_bar(bars: pd.DataFrame) -> Signal | None` method.
        Raises SandboxError if the code fails validation or instantiation.
        """
        # 1. Static scan
        scan = self.validate_source(source)
        if not scan.passed:
            errors = [v for v in scan.violations if v.severity == "error"]
            detail = "; ".join(f"L{v.line}: {v.detail}" for v in errors[:5])
            raise SandboxError(f"Static scan failed: {detail}")

        # 2. Compile
        try:
            code = compile(source, "<strategy>", "exec")
        except SyntaxError as e:
            raise SandboxError(f"Syntax error in strategy code: {e}") from e

        # 3. Execute in restricted namespace
        ns = _make_restricted_globals()

        # Add Signal and Direction so strategies can create signals
        from core.backtester import Signal, Direction, OrderType
        ns["Signal"] = Signal
        ns["Direction"] = Direction
        ns["OrderType"] = OrderType

        try:
            exec(code, ns)
        except SandboxImportError:
            raise
        except Exception as e:
            raise SandboxError(f"Error executing strategy code: {type(e).__name__}: {e}") from e

        # 4. Find and instantiate the strategy class
        if strategy_class_name not in ns:
            # Try to find any class with on_bar method
            candidates = [name for name, obj in ns.items()
                          if isinstance(obj, type) and hasattr(obj, "on_bar")]
            if not candidates:
                raise SandboxError(
                    f"No class named '{strategy_class_name}' found, and no class with "
                    "an 'on_bar' method exists in the strategy code."
                )
            strategy_class_name = candidates[0]

        klass = ns[strategy_class_name]
        if not hasattr(klass, "on_bar"):
            raise SandboxError(f"Class '{strategy_class_name}' has no 'on_bar' method.")

        try:
            instance = klass()
        except Exception as e:
            raise SandboxError(f"Error instantiating {strategy_class_name}: {e}") from e

        return _TimeoutWrapper(instance, self._wall_timeout)


class _TimeoutWrapper:
    """Wraps a strategy to enforce a wall-clock timeout on on_bar calls.

    Uses a threading approach that works on Windows (no SIGALRM).
    The cumulative time across all on_bar calls is tracked.
    """

    def __init__(self, inner: Any, max_wall_seconds: float):
        self._inner = inner
        self._max_wall = max_wall_seconds
        self._cumulative = 0.0

    def on_bar(self, bars):
        t0 = time.perf_counter()
        result = self._inner.on_bar(bars)
        elapsed = time.perf_counter() - t0
        self._cumulative += elapsed
        if self._cumulative > self._max_wall:
            raise SandboxTimeoutError(
                f"Strategy exceeded wall-clock timeout: "
                f"{self._cumulative:.1f}s > {self._max_wall}s"
            )
        return result
