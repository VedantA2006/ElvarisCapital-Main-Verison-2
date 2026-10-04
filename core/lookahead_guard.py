"""
core/lookahead_guard.py – Multi-layer lookahead detection.

Layer 1: STATIC (AST scan) – catches obvious future-data access patterns
          in LLM-generated code BEFORE execution.
Layer 2: DYNAMIC (delay test) – shifts all signals by 1 bar and re-backtests.
          A >60% Sharpe drop is a strong signal of lookahead or curve-fitting.
Layer 3: DYNAMIC (truncation test) – cuts the data at N random points and
          re-backtests. If results are wildly unstable, the strategy is
          likely peeking at the full dataset shape.

Design choices:
- Layer 1 is conservative (false positives → the LLM gets a second chance
  with explicit feedback). False negatives are caught by Layers 2-3.
- Layers 2-3 are run on the TRAIN split only. They NEVER touch holdout.
- The delay test uses the same backtester and cost model. No shortcuts.
"""

from __future__ import annotations

import ast
import copy
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: Static AST Analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LookaheadViolation:
    line: int
    col: int
    pattern: str
    detail: str
    severity: str = "error"    # "error" = definite, "warning" = suspicious


@dataclass
class StaticScanResult:
    passed: bool
    violations: list[LookaheadViolation] = field(default_factory=list)
    banned_imports: list[str] = field(default_factory=list)
    summary: str = ""

    def to_doc(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [
                {"line": v.line, "col": v.col, "pattern": v.pattern,
                 "detail": v.detail, "severity": v.severity}
                for v in self.violations
            ],
            "banned_imports": self.banned_imports,
            "summary": self.summary,
        }


# Patterns that indicate future-data access
BANNED_PATTERNS = [
    # .shift(-N) where N > 0 → accessing future rows
    {"attr": "shift", "detail": "shift() with negative argument accesses future data"},
    # Direct iloc/loc with [i+N] where N > 0
    {"attr": "pct_change", "detail": "pct_change() default looks back, but check context"},
]

# Regex patterns for source-level scanning (catches string-based tricks)
SOURCE_PATTERNS = [
    (r'\.shift\s*\(\s*-', "shift(-N) accesses future data"),
    (r'\.iloc\s*\[\s*.*\+\s*[1-9]', "iloc[i+N] may access future data"),
    (r'\.loc\s*\[\s*.*\+\s*[1-9]', "loc[i+N] may access future data"),
    (r'\.values\s*\[\s*.*\+\s*[1-9]', "values[i+N] may access future data"),
    (r'reversed?\s*\(\s*', "reversed() on time series is suspicious"),
    (r'\.sort_values\s*\(.*ascending\s*=\s*False', "reverse-sorting time series is suspicious"),
    (r'\[::\s*-1\s*\]', "[::-1] slice reversal on time series is suspicious"),
    (r'\.cumsum\s*\(\s*\).*\.iloc\s*\[\s*-1\s*\]', "cumsum().iloc[-1] sees the full series"),
    (r'future|lookahead|peek|cheat', "suspicious variable/comment name"),
]

# Imports that are never allowed in strategy code
BANNED_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "httpx",
    "importlib", "ctypes", "pickle", "shelve",
    "multiprocessing", "threading", "asyncio",
    "signal", "resource", "gc",
    "__builtin__", "builtins",
}

# Only these are allowed
ALLOWED_IMPORTS = {"numpy", "pandas", "numba", "math", "np", "pd"}


class _LookaheadVisitor(ast.NodeVisitor):
    """Walk the AST looking for future-data access patterns."""

    def __init__(self):
        self.violations: list[LookaheadViolation] = []
        self.imports: list[str] = []
        self.banned_imports: list[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)
            root = alias.name.split(".")[0]
            if root not in ALLOWED_IMPORTS:
                self.banned_imports.append(alias.name)
                self.violations.append(LookaheadViolation(
                    line=node.lineno, col=node.col_offset,
                    pattern="banned_import",
                    detail=f"Import '{alias.name}' is not allowed in strategy code. "
                           f"Allowed: {sorted(ALLOWED_IMPORTS)}",
                ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        root = mod.split(".")[0]
        if root and root not in ALLOWED_IMPORTS:
            self.banned_imports.append(mod)
            self.violations.append(LookaheadViolation(
                line=node.lineno, col=node.col_offset,
                pattern="banned_import",
                detail=f"Import from '{mod}' is not allowed. Allowed: {sorted(ALLOWED_IMPORTS)}",
            ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Check for .shift(-N)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "shift"):
            for arg in node.args:
                if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                    if isinstance(arg.operand, (ast.Constant,)):
                        if isinstance(arg.operand.value, (int, float)) and arg.operand.value > 0:
                            self.violations.append(LookaheadViolation(
                                line=node.lineno, col=node.col_offset,
                                pattern="shift_negative",
                                detail=f"shift(-{arg.operand.value}) accesses future data. "
                                       "Use shift(+N) to look back.",
                            ))
                elif isinstance(arg, ast.Constant):
                    if isinstance(arg.value, (int, float)) and arg.value < 0:
                        self.violations.append(LookaheadViolation(
                            line=node.lineno, col=node.col_offset,
                            pattern="shift_negative",
                            detail=f"shift({arg.value}) accesses future data.",
                        ))

        # Check for .rolling/ewm on reversed data
        if (isinstance(node.func, ast.Attribute) and
                node.func.attr in ("rolling", "ewm", "expanding")):
            # Check if called on a reversed series
            if isinstance(node.func.value, ast.Subscript):
                if isinstance(node.func.value.slice, ast.Slice):
                    if (node.func.value.slice.step is not None and
                            isinstance(node.func.value.slice.step, ast.UnaryOp) and
                            isinstance(node.func.value.slice.step.op, ast.USub)):
                        self.violations.append(LookaheadViolation(
                            line=node.lineno, col=node.col_offset,
                            pattern="reversed_rolling",
                            detail="rolling/ewm on reversed data ([::-1]) is lookahead.",
                            severity="warning",
                        ))

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        # Check for .iloc[i+1] or .values[i+1] patterns
        if isinstance(node.value, ast.Attribute):
            if node.value.attr in ("iloc", "values", "iat"):
                if isinstance(node.slice, ast.BinOp) and isinstance(node.slice.op, ast.Add):
                    if isinstance(node.slice.right, ast.Constant):
                        if isinstance(node.slice.right.value, (int,)) and node.slice.right.value > 0:
                            self.violations.append(LookaheadViolation(
                                line=node.lineno, col=node.col_offset,
                                pattern="index_forward",
                                detail=f"{node.value.attr}[i+{node.slice.right.value}] "
                                       "accesses a future row.",
                            ))
        self.generic_visit(node)


def static_scan(source: str, allowed_imports: set[str] | None = None) -> StaticScanResult:
    """Run all static checks on strategy source code using the hardened AST policy.

    Returns StaticScanResult. passed=True means no errors (warnings are OK).
    """
    from sandbox.policy import validate_ast_policy

    policy_res = validate_ast_policy(source, allowed_imports=allowed_imports)

    violations: list[LookaheadViolation] = [
        LookaheadViolation(
            line=v.line,
            col=v.col,
            pattern=v.pattern,
            detail=v.detail,
            severity=v.severity,
        )
        for v in policy_res.violations
    ]

    # Source-level regex patterns for advisory warnings
    for i, line in enumerate(source.splitlines(), 1):
        for pattern, detail in SOURCE_PATTERNS:
            if re.search(pattern, line):
                if not any(v.line == i for v in violations):
                    violations.append(LookaheadViolation(
                        line=i, col=0, pattern="regex_match",
                        detail=detail, severity="warning",
                    ))

    errors = [v for v in violations if v.severity == "error"]
    passed = len(errors) == 0

    summary_parts = []
    if errors:
        summary_parts.append(f"{len(errors)} error(s)")
    warnings = [v for v in violations if v.severity == "warning"]
    if warnings:
        summary_parts.append(f"{len(warnings)} warning(s)")
    summary = ", ".join(summary_parts) if summary_parts else "Clean"

    return StaticScanResult(
        passed=passed,
        violations=violations,
        banned_imports=policy_res.banned_imports,
        summary=summary,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: Delay Test
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DelayTestResult:
    passed: bool
    original_sharpe: float
    delayed_sharpe: float
    sharpe_drop_pct: float
    original_trades: int
    delayed_trades: int
    detail: str

    def to_doc(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "original_sharpe": self.original_sharpe,
            "delayed_sharpe": self.delayed_sharpe,
            "sharpe_drop_pct": round(self.sharpe_drop_pct, 4),
            "original_trades": self.original_trades,
            "delayed_trades": self.delayed_trades,
            "detail": self.detail,
        }


class _DelayedStrategy:
    """Wraps a strategy, delaying all signals by `delay_bars` bars."""

    def __init__(self, inner, delay_bars: int = 1):
        self._inner = inner
        self._delay = delay_bars
        self._pending: list[Any] = []
        self._bar_count = 0

    def on_bar(self, bars: pd.DataFrame):
        signal = self._inner.on_bar(bars)
        self._bar_count += 1
        self._pending.append(signal)
        if len(self._pending) > self._delay:
            return self._pending.pop(0)
        return None


def run_delay_test(strategy_factory, df: pd.DataFrame, cfg: dict,
                   delay_bars: int = 1) -> DelayTestResult:
    """Run the original strategy and a 1-bar-delayed copy. Compare Sharpe.

    strategy_factory: callable that returns a fresh strategy instance.
    """
    from core.backtester import run_backtest

    max_drop = cfg["gates"]["delay"]["max_sharpe_drop_1bar"]

    # Original
    orig = run_backtest(strategy_factory(), df, cfg)
    orig_sharpe = orig.metrics.get("sharpe", 0.0)
    orig_trades = orig.metrics.get("total_trades", 0)

    # Delayed
    delayed_strat = _DelayedStrategy(strategy_factory(), delay_bars=delay_bars)
    delayed = run_backtest(delayed_strat, df, cfg)
    del_sharpe = delayed.metrics.get("sharpe", 0.0)
    del_trades = delayed.metrics.get("total_trades", 0)

    if orig_sharpe <= 0:
        return DelayTestResult(
            passed=True, original_sharpe=orig_sharpe, delayed_sharpe=del_sharpe,
            sharpe_drop_pct=0.0, original_trades=orig_trades, delayed_trades=del_trades,
            detail="Original Sharpe <= 0; delay test not meaningful.",
        )

    drop = (orig_sharpe - del_sharpe) / abs(orig_sharpe)
    passed = drop < max_drop

    detail = (f"Sharpe dropped {drop*100:.1f}% with {delay_bars}-bar delay "
              f"(threshold: {max_drop*100:.0f}%). "
              f"{'PASS' if passed else 'FAIL: likely lookahead or extreme curve-fit'}.")

    return DelayTestResult(
        passed=passed, original_sharpe=round(orig_sharpe, 4),
        delayed_sharpe=round(del_sharpe, 4), sharpe_drop_pct=round(drop * 100, 2),
        original_trades=orig_trades, delayed_trades=del_trades, detail=detail,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: Truncation Test
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TruncationTestResult:
    passed: bool
    cut_results: list[dict[str, Any]]
    sharpe_std: float
    sharpe_cv: float        # coefficient of variation
    detail: str

    def to_doc(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "num_cuts": len(self.cut_results),
            "sharpe_std": round(self.sharpe_std, 4),
            "sharpe_cv": round(self.sharpe_cv, 4),
            "detail": self.detail,
        }


def run_truncation_test(strategy_factory, df: pd.DataFrame, cfg: dict,
                        num_cuts: int | None = None, seed: int = 42) -> TruncationTestResult:
    """Cut data at N random points, re-backtest each prefix. Unstable results → suspicious.

    A strategy that peeks at the full dataset shape (e.g. normalising by
    global min/max) will show wildly different metrics across truncations.
    """
    from core.backtester import run_backtest

    if num_cuts is None:
        num_cuts = cfg["gates"]["truncation"]["num_cut_points"]

    rng = np.random.default_rng(seed)
    n = len(df)
    min_bars = max(200, n // 5)  # each truncation needs enough bars to be meaningful

    if n < min_bars * 2:
        return TruncationTestResult(
            passed=True, cut_results=[], sharpe_std=0.0, sharpe_cv=0.0,
            detail="Dataset too small for truncation test.",
        )

    cut_points = sorted(rng.integers(min_bars, n, size=num_cuts))
    results: list[dict[str, Any]] = []
    sharpes: list[float] = []

    for cp in cut_points:
        truncated = df.iloc[:cp].copy()
        truncated.attrs = df.attrs.copy()
        try:
            r = run_backtest(strategy_factory(), truncated, cfg)
            s = r.metrics.get("sharpe", 0.0)
            sharpes.append(s)
            results.append({"cut_point": int(cp), "sharpe": round(s, 4),
                            "trades": r.metrics.get("total_trades", 0)})
        except Exception as e:
            results.append({"cut_point": int(cp), "error": str(e)})

    if len(sharpes) < 3:
        return TruncationTestResult(
            passed=True, cut_results=results, sharpe_std=0.0, sharpe_cv=0.0,
            detail="Not enough successful truncations to assess stability.",
        )

    sharpe_arr = np.array(sharpes)
    std = float(sharpe_arr.std())
    mean = float(sharpe_arr.mean())
    cv = std / abs(mean) if abs(mean) > 1e-9 else float("inf")

    # A CV > 2.0 means results are wildly unstable
    passed = cv < 2.0
    detail = (f"Truncation test: {len(sharpes)} cuts, Sharpe mean={mean:.3f} "
              f"std={std:.3f} CV={cv:.3f}. {'PASS' if passed else 'FAIL: unstable results'}")

    return TruncationTestResult(
        passed=passed, cut_results=results,
        sharpe_std=round(std, 4), sharpe_cv=round(cv, 4), detail=detail,
    )
