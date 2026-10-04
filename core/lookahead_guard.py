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
# Layer 2: Delay Test on Tapes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DelayTestResult:
    passed: bool
    original_sharpe: float
    delayed_sharpe: float
    delay1_sharpe: float
    delay2_sharpe: float
    sharpe_drop_pct: float
    original_trades: int
    delayed_trades: int
    detail: str

    def to_doc(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "original_sharpe": self.original_sharpe,
            "delayed_sharpe": self.delayed_sharpe,
            "delay1_sharpe": self.delay1_sharpe,
            "delay2_sharpe": self.delay2_sharpe,
            "sharpe_drop_pct": round(self.sharpe_drop_pct, 4),
            "original_trades": self.original_trades,
            "delayed_trades": self.delayed_trades,
            "detail": self.detail,
        }


def run_delay_test_on_tape(
    tape_or_strategy: Any,
    df: pd.DataFrame,
    cfg: dict,
    base_result: Any = None,
    delay_bars: int = 1,
) -> DelayTestResult:
    """Execute delay test on pre-computed SignalTape (no strategy re-run).

    Shifts tape by 1 and 2 bars. Fails if:
    1. Base strategy is non-profitable (Sharpe <= 0 or 0 trades).
    2. Net Sharpe drops by more than max_sharpe_drop_1bar (default 60%).
    3. Net Sharpe at +2 bars is negative while base is strongly positive (Sharpe >= 0.80).
    """
    from core.backtester import generate_signal_tape, run_simulation_on_tape
    from core.signals import SignalTape

    if isinstance(tape_or_strategy, SignalTape):
        tape = tape_or_strategy
    else:
        tape, errs, first_err = generate_signal_tape(tape_or_strategy, df, cfg)
        if errs > 0:
            return DelayTestResult(
                passed=False,
                original_sharpe=0.0,
                delayed_sharpe=0.0,
                delay1_sharpe=0.0,
                delay2_sharpe=0.0,
                sharpe_drop_pct=100.0,
                original_trades=0,
                delayed_trades=0,
                detail=f"Failed generating strategy tape: {first_err}",
            )

    if base_result is None:
        base_result = run_simulation_on_tape(tape, df, cfg)

    orig_sharpe = float(base_result.metrics.get("sharpe", 0.0) or 0.0)
    orig_trades = int(base_result.metrics.get("total_trades", 0))

    # Reject non-profitable base strategy: delay test cannot return ambiguous pass
    if orig_sharpe <= 0 or orig_trades == 0:
        return DelayTestResult(
            passed=False,
            original_sharpe=orig_sharpe,
            delayed_sharpe=0.0,
            delay1_sharpe=0.0,
            delay2_sharpe=0.0,
            sharpe_drop_pct=100.0,
            original_trades=orig_trades,
            delayed_trades=0,
            detail=f"Base strategy is not profitable (Sharpe={orig_sharpe:.2f}, trades={orig_trades}); delay test fails.",
        )

    # Shift by 1 bar and simulate
    tape_d1 = tape.shifted(1)
    res_d1 = run_simulation_on_tape(tape_d1, df, cfg)
    s1 = float(res_d1.metrics.get("sharpe", 0.0) or 0.0)
    trades_d1 = int(res_d1.metrics.get("total_trades", 0))

    # Shift by 2 bars and simulate
    tape_d2 = tape.shifted(2)
    res_d2 = run_simulation_on_tape(tape_d2, df, cfg)
    s2 = float(res_d2.metrics.get("sharpe", 0.0) or 0.0)

    # Drop evaluation
    max_drop = float(cfg.get("gates", {}).get("delay", {}).get("max_sharpe_drop_1bar", 0.60))
    drop_1 = (orig_sharpe - s1) / abs(orig_sharpe)

    drop_failed = drop_1 > max_drop
    neg2_failed = (s2 < 0 and orig_sharpe >= 0.80)

    passed = not (drop_failed or neg2_failed)

    failures = []
    if drop_failed:
        failures.append(f"Sharpe dropped {drop_1*100:.1f}% > {max_drop*100:.0f}% with 1-bar delay")
    if neg2_failed:
        failures.append(f"Sharpe collapsed to negative ({s2:.2f}) with 2-bar delay")

    detail = (
        f"Delay test: base Sharpe={orig_sharpe:.2f} -> +1bar={s1:.2f} (drop {drop_1*100:.1f}%), "
        f"+2bar={s2:.2f}. {'PASS' if passed else 'FAIL: ' + '; '.join(failures)}"
    )

    return DelayTestResult(
        passed=passed,
        original_sharpe=round(orig_sharpe, 4),
        delayed_sharpe=round(s1, 4),
        delay1_sharpe=round(s1, 4),
        delay2_sharpe=round(s2, 4),
        sharpe_drop_pct=round(drop_1 * 100, 2),
        original_trades=orig_trades,
        delayed_trades=trades_d1,
        detail=detail,
    )


# Alias for backwards compatibility
run_delay_test = run_delay_test_on_tape


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: Determinism Test
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DeterminismTestResult:
    passed: bool
    first_diverging_bar: int | None = None
    first_diverging_timestamp: str | None = None
    diverging_field: str | None = None
    run1_value: Any = None
    run2_value: Any = None
    detail: str = ""

    def to_doc(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "first_diverging_bar": self.first_diverging_bar,
            "first_diverging_timestamp": self.first_diverging_timestamp,
            "diverging_field": self.diverging_field,
            "detail": self.detail,
        }


def run_determinism_test(
    strategy_or_source: Any,
    df: pd.DataFrame,
    cfg: dict,
    seed: int = 42,
) -> DeterminismTestResult:
    """Run full strategy twice with identical seed; tapes must be bit-identical.

    Catches unseeded randomness, time-dependence, or cross-run memory.
    """
    from core.backtester import generate_signal_tape

    tape1, errs1, e1 = generate_signal_tape(strategy_or_source, df, cfg, seed=seed)
    if errs1 > 0:
        return DeterminismTestResult(passed=False, detail=f"Run 1 failed with error: {e1}")

    tape2, errs2, e2 = generate_signal_tape(strategy_or_source, df, cfg, seed=seed)
    if errs2 > 0:
        return DeterminismTestResult(passed=False, detail=f"Run 2 failed with error: {e2}")

    # Bitwise comparison
    act_eq = (tape1.actions == tape2.actions)
    sl_eq = np.isclose(tape1.sl_distances, tape2.sl_distances, equal_nan=True, atol=1e-9)
    tp_eq = np.isclose(tape1.tp_distances, tape2.tp_distances, equal_nan=True, atol=1e-9)
    tr_eq = np.isclose(tape1.trail_distances, tape2.trail_distances, equal_nan=True, atol=1e-9)
    ts_eq = (tape1.time_stops == tape2.time_stops)
    be_eq = np.isclose(tape1.breakeven_r, tape2.breakeven_r, equal_nan=True, atol=1e-9)

    all_eq = act_eq & sl_eq & tp_eq & tr_eq & ts_eq & be_eq
    if not np.all(all_eq):
        idx = int(np.where(~all_eq)[0][0])
        ts = str(df["timestamp"].iloc[idx]) if "timestamp" in df.columns else str(idx)

        diff_field = "action"
        v1, v2 = tape1.actions[idx], tape2.actions[idx]
        if not sl_eq[idx]:
            diff_field, v1, v2 = "sl_distance", tape1.sl_distances[idx], tape2.sl_distances[idx]
        elif not tp_eq[idx]:
            diff_field, v1, v2 = "tp_distance", tape1.tp_distances[idx], tape2.tp_distances[idx]
        elif not tr_eq[idx]:
            diff_field, v1, v2 = "trail_distance", tape1.trail_distances[idx], tape2.trail_distances[idx]
        elif not ts_eq[idx]:
            diff_field, v1, v2 = "time_stops", tape1.time_stops[idx], tape2.time_stops[idx]
        elif not be_eq[idx]:
            diff_field, v1, v2 = "breakeven_r", tape1.breakeven_r[idx], tape2.breakeven_r[idx]

        detail = (
            f"Non-deterministic signal at bar {idx} ({ts}): {diff_field} run1={v1} != run2={v2}. "
            "Strategy has unseeded randomness or time/state leak."
        )
        return DeterminismTestResult(
            passed=False,
            first_diverging_bar=idx,
            first_diverging_timestamp=ts,
            diverging_field=diff_field,
            run1_value=v1,
            run2_value=v2,
            detail=detail,
        )

    return DeterminismTestResult(
        passed=True,
        detail=f"Bit-identical signal tapes produced on two independent runs with seed={seed}.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4: Truncation Test (Bitwise Signal Identity)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TruncationTestResult:
    passed: bool
    num_cuts: int
    cut_results: list[dict[str, Any]]
    first_diverging_bar: int | None = None
    first_diverging_timestamp: str | None = None
    diverging_field: str | None = None
    detail: str = ""

    def to_doc(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "num_cuts": self.num_cuts,
            "first_diverging_bar": self.first_diverging_bar,
            "first_diverging_timestamp": self.first_diverging_timestamp,
            "diverging_field": self.diverging_field,
            "detail": self.detail,
        }


def run_truncation_test(
    strategy_or_source: Any,
    df: pd.DataFrame,
    cfg: dict,
    num_cuts: int | None = None,
    seed: int = 42,
) -> TruncationTestResult:
    """The Real Truncation Test: enforces bitwise signal identity T_c[0..c] == T_full[0..c].

    1. Runs strategy on full train split to get T_full.
    2. Chooses N cut points c (seeded, spread over the split).
    3. Runs strategy on prefix 0..c to get T_c.
    4. Asserts T_c[0..c] == T_full[0..c] for all signal fields within 1e-9 tolerance.
    Deletes the old Sharpe-CV implementation entirely.
    """
    from core.backtester import generate_signal_tape

    if num_cuts is None:
        num_cuts = cfg.get("gates", {}).get("truncation", {}).get("num_cut_points", 20)

    n_bars = len(df)
    first_live = 0
    if "is_warmup" in df.columns:
        w = np.where(~df["is_warmup"].to_numpy())[0]
        if len(w) > 0:
            first_live = int(w[0])

    min_bars = max(first_live + 10, min(50, n_bars // 4))

    if n_bars < 100:
        return TruncationTestResult(
            passed=True,
            num_cuts=0,
            cut_results=[],
            detail="Dataset too small for truncation test.",
        )

    # 1. Run full feed
    tape_full, errs_full, first_err = generate_signal_tape(strategy_or_source, df, cfg, seed=seed)
    if errs_full > 0:
        return TruncationTestResult(
            passed=False,
            num_cuts=0,
            cut_results=[],
            detail=f"Full train feed failed with error: {first_err}",
        )

    # 2. Choose cut points spread over split (including last bar and just after warm-up)
    rng = np.random.default_rng(seed)
    if num_cuts <= 2:
        cut_points = [min_bars, n_bars]
    else:
        interior_cuts = rng.integers(min_bars + 1, n_bars, size=num_cuts - 2)
        cut_points = sorted(list(set([min_bars] + list(interior_cuts) + [n_bars])))

    cut_results = []

    # 3. Check each prefix
    for c in cut_points:
        df_c = df.iloc[:c].copy()
        df_c.attrs = df.attrs.copy()

        tape_c, c_errs, c_err = generate_signal_tape(strategy_or_source, df_c, cfg, seed=seed)
        if c_errs > 0:
            return TruncationTestResult(
                passed=False,
                num_cuts=len(cut_points),
                cut_results=cut_results,
                first_diverging_bar=c,
                detail=f"Cut at bar {c} raised strategy error: {c_err}",
            )

        # 4. Bitwise comparison over 0..c
        act_eq = (tape_c.actions[:c] == tape_full.actions[:c])
        sl_eq = np.isclose(tape_c.sl_distances[:c], tape_full.sl_distances[:c], equal_nan=True, atol=1e-9)
        tp_eq = np.isclose(tape_c.tp_distances[:c], tape_full.tp_distances[:c], equal_nan=True, atol=1e-9)
        tr_eq = np.isclose(tape_c.trail_distances[:c], tape_full.trail_distances[:c], equal_nan=True, atol=1e-9)
        ts_eq = (tape_c.time_stops[:c] == tape_full.time_stops[:c])
        be_eq = np.isclose(tape_c.breakeven_r[:c], tape_full.breakeven_r[:c], equal_nan=True, atol=1e-9)

        all_eq = act_eq & sl_eq & tp_eq & tr_eq & ts_eq & be_eq
        if not np.all(all_eq):
            idx = int(np.where(~all_eq)[0][0])
            ts = str(df["timestamp"].iloc[idx]) if "timestamp" in df.columns else str(idx)

            diff_field = "action"
            v_full, v_cut = tape_full.actions[idx], tape_c.actions[idx]
            if not sl_eq[idx]:
                diff_field, v_full, v_cut = "sl_distance", tape_full.sl_distances[idx], tape_c.sl_distances[idx]
            elif not tp_eq[idx]:
                diff_field, v_full, v_cut = "tp_distance", tape_full.tp_distances[idx], tape_c.tp_distances[idx]
            elif not tr_eq[idx]:
                diff_field, v_full, v_cut = "trail_distance", tape_full.trail_distances[idx], tape_c.trail_distances[idx]
            elif not ts_eq[idx]:
                diff_field, v_full, v_cut = "time_stops", tape_full.time_stops[idx], tape_c.time_stops[idx]
            elif not be_eq[idx]:
                diff_field, v_full, v_cut = "breakeven_r", tape_full.breakeven_r[idx], tape_c.breakeven_r[idx]

            detail = (
                f"Lookahead or repainting detected: signal at bar {idx} ({ts}) differs between "
                f"full run and prefix 0..{c}. Field: {diff_field} (full={v_full} != cut={v_cut})."
            )
            cut_results.append({"cut": c, "passed": False, "diverging_bar": idx})
            return TruncationTestResult(
                passed=False,
                num_cuts=len(cut_points),
                cut_results=cut_results,
                first_diverging_bar=idx,
                first_diverging_timestamp=ts,
                diverging_field=diff_field,
                detail=detail,
            )

        cut_results.append({"cut": c, "passed": True})

    detail = f"Verified signal identity T_c[0..c] == T_full[0..c] across {len(cut_points)} cut points (tolerance 1e-9)."
    return TruncationTestResult(
        passed=True,
        num_cuts=len(cut_points),
        cut_results=cut_results,
        detail=detail,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 5: Block-Shuffle / Permutation Test
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BlockShuffleResult:
    passed: bool
    real_sharpe: float
    synthetic_sharpe_p95: float
    synthetic_sharpes: list[float]
    n_paths: int
    detail: str

    def to_doc(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "real_sharpe": round(self.real_sharpe, 4),
            "synthetic_sharpe_p95": round(self.synthetic_sharpe_p95, 4),
            "n_paths": self.n_paths,
            "detail": self.detail,
        }


def run_block_shuffle_test(
    strategy_or_source: Any,
    df: pd.DataFrame,
    cfg: dict,
    n_paths: int = 30,
    block_length: int | None = None,
    seed: int = 42,
) -> BlockShuffleResult:
    """Block-shuffle bar returns to construct synthetic paths; real Sharpe must > 95th percentile."""
    from core.backtester import run_backtest

    if block_length is None:
        perm_cfg = cfg.get("gates", {}).get("permutation", {})
        bl_cfg = perm_cfg.get("block_length_bars", {})
        tf = df.attrs.get("timeframe", "1h")
        block_length = int(bl_cfg.get(tf, bl_cfg.get("default", 24 if tf == "1h" else 10)))

    # Real Sharpe
    base_res = run_backtest(strategy_or_source, df, cfg)
    real_sharpe = float(base_res.metrics.get("sharpe", 0.0) or 0.0)

    n_bars = len(df)
    n_blocks = max(1, n_bars // block_length)
    blocks = [df.iloc[b * block_length: min((b + 1) * block_length, n_bars)].copy() for b in range(n_blocks)]

    rng = np.random.default_rng(seed)
    synthetic_sharpes: list[float] = []

    for path_idx in range(n_paths):
        perm_order = rng.permutation(n_blocks)
        # Reconstruct path from permuted blocks
        chunks = []
        cur_close = float(df["open"].iloc[0])
        for b_idx in perm_order:
            b_df = blocks[b_idx].copy()
            b_open0 = float(b_df["open"].iloc[0])
            scale = cur_close / b_open0 if b_open0 > 0 else 1.0
            b_df["open"] *= scale
            b_df["high"] *= scale
            b_df["low"] *= scale
            b_df["close"] *= scale
            cur_close = float(b_df["close"].iloc[-1])
            chunks.append(b_df)

        synth_df = pd.concat(chunks, ignore_index=True)
        synth_df["timestamp"] = df["timestamp"].iloc[: len(synth_df)].values
        if "session" in df.columns:
            synth_df["session"] = df["session"].iloc[: len(synth_df)].values
        synth_df.attrs = df.attrs.copy()

        synth_res = run_backtest(strategy_or_source, synth_df, cfg)
        s_sharpe = float(synth_res.metrics.get("sharpe", 0.0) or 0.0)
        synthetic_sharpes.append(s_sharpe)

    p95 = float(np.percentile(synthetic_sharpes, 95.0)) if synthetic_sharpes else 0.0
    passed = real_sharpe > p95

    detail = (
        f"Block-shuffle test ({n_paths} paths, block={block_length}): real Sharpe={real_sharpe:.2f} "
        f"vs 95th percentile={p95:.2f}. {'PASS' if passed else 'FAIL: real Sharpe <= 95th percentile of synthetic paths'}"
    )

    return BlockShuffleResult(
        passed=passed,
        real_sharpe=real_sharpe,
        synthetic_sharpe_p95=p95,
        synthetic_sharpes=synthetic_sharpes,
        n_paths=n_paths,
        detail=detail,
    )
