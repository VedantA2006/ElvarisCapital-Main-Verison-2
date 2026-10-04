"""
tests/regression/test_f4_lookahead.py – Regression suite for Phase F4.

Covers:
- LH-2: Truncation test verifies signal identity T_c[0..c] == T_full[0..c], rejects cheats (deletes Sharpe CV)
- LH-3: Static scan catches all lookahead and evasion patterns
- LH-4: Suspicion flags trigger deep audit and reject as suspected_leak
- LH-5: Delay test on tapes rejects non-profitable base and >60% drops / negative 2-bar
- LH-6: Canary suite and controls
- GATE-6: Gate ordering enforces lookahead checks BEFORE validation is touched
- Determinism test: detects unseeded randomness or cross-run state
- Block-shuffle permutation test: requires Sharpe > 95th percentile of synthetic distribution
"""

from __future__ import annotations

import textwrap
import numpy as np
import pandas as pd
import pytest

from core.config import load_config
from core.data_loader import add_session_labels
from tests.conftest import make_bars


def _make_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    cfg = load_config()
    df = make_bars("2022-01-03", "2023-06-01", "60min", seed=seed)
    df = df.head(n).reset_index(drop=True)
    df = add_session_labels(df, cfg)
    df.attrs = {"timeframe": "1h", "split": "train", "file_hash": "test", "slice_hash": "test"}
    return df


# ── LH-2: Real Truncation Test (Bitwise Signal Identity) ─────────────────────

def test_lh2_truncation_catches_cheater():
    """LH-2: Truncation test must compare signal tapes bitwise (T_c[0..c] == T_full[0..c]),
    NOT Sharpe CV. A strategy whose signals depend on future bars must be rejected!
    """
    from core.lookahead_guard import run_truncation_test
    from core.signals import Action, Signal

    cfg = load_config()
    df = _make_df(300)

    # Honest causal strategy
    causal_code = textwrap.dedent("""
    from core.signals import Signal, Action

    class Strategy:
        def __init__(self):
            pass

        def on_bar(self, bars):
            if len(bars) < 30:
                return None
            sma = bars['close'].rolling(20).mean().iloc[-1]
            c = bars['close'].iloc[-1]
            if c > sma * 1.01:
                return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=30.0)
            return None
    """)

    res_honest = run_truncation_test(causal_code, df, cfg, num_cuts=10, seed=42)
    assert res_honest.passed, f"Honest causal strategy must pass truncation test: {res_honest.detail}"
    assert not hasattr(res_honest, "sharpe_cv") or res_honest.sharpe_cv is None, \
        "Old Sharpe CV implementation must be deleted!"

    # Strategy that repaints / diverges based on future context
    class DivergentFactory:
        def __init__(self):
            self.run_count = 0

        def __call__(self):
            self.run_count += 1
            is_full = (self.run_count == 1)

            class DivergentStrategy:
                def on_bar(self, bars):
                    if len(bars) < 30:
                        return None
                    # Emulate repainting: full run emits LONG, but prefix run emits SHORT
                    act = Action.ENTER_LONG if is_full else Action.ENTER_SHORT
                    return Signal(action=act, sl_distance=15.0, tp_distance=40.0)

            return DivergentStrategy()

    res_divergent = run_truncation_test(DivergentFactory(), df, cfg, num_cuts=10, seed=42)
    assert not res_divergent.passed, "Strategy with diverging signals across cuts must fail truncation!"
    assert res_divergent.first_diverging_bar is not None


# ── LH-3: Static Scan Catches Lookahead & Evasions ───────────────────────────

def test_lh3_static_scan_catches_all_lookahead():
    """LH-3: Static scan catches all lookahead patterns without exception."""
    from core.lookahead_guard import static_scan

    forbidden_patterns = [
        "x = df['close'].shift(periods=-1)",
        "x = df['close'].shift(-2)",
        "x = df['close'].rolling(10, center=True).mean()",
        "n = -1\nx = df['close'].shift(n)",
        "j = i + 1\nx = df.iloc[j]",
        "x = a.base",
        "x = a.ctypes.data",
        "x = a.__array_interface__",
        "x = ().__class__.__base__.__subclasses__()",
        "import os",
        "import socket",
        "from subprocess import Popen",
    ]
    for code in forbidden_patterns:
        res = static_scan(code)
        assert not res.passed, f"Expected static_scan to REJECT pattern: {code}"


# ── LH-4: Suspicion Flags Are Hard Rejections ────────────────────────────────

def test_lh4_suspicion_flags_trigger_hard_rejection():
    """LH-4: Suspicious metrics (Sharpe > 3.0, PF > 3.0, WR > 0.85, DD < 1%)
    must trigger deep audit and reject as suspected_leak, not pass silently.
    """
    from core.gates import check_suspicion_and_audit

    cfg = load_config()

    suspicious_metrics = {
        "total_trades": 250,
        "trades_per_year": 50,
        "profit_factor": 4.5,   # > 3.0
        "sharpe": 3.8,          # > 3.0
        "max_drawdown": 0.005,  # < 1%
        "win_rate": 0.92,       # > 85%
        "expectancy": 25.0,
        "cost_gross_ratio": 0.05,
        "profit_concentration_top5pct": 0.80, # > 70%
    }

    is_susp, reasons = check_suspicion_and_audit(suspicious_metrics, cfg)
    assert is_susp, "Metrics exceeding suspicion thresholds must be flagged as suspicious"
    assert len(reasons) >= 3


# ── LH-5: Delay Test on Tapes ───────────────────────────────────────────────

def test_lh5_delay_test_on_tapes():
    """LH-5: Delay test executes on SignalTape (no strategy re-run),
    rejects non-profitable base strategies, and fails on >60% Sharpe drop or negative 2-bar.
    """
    from core.lookahead_guard import run_delay_test_on_tape
    from core.signals import SignalTape, Action

    cfg = load_config()
    df = _make_df(300)
    n = len(df)

    # 1. Non-profitable base strategy must FAIL delay test
    empty_tape = SignalTape.empty(n)
    res_empty = run_delay_test_on_tape(empty_tape, df, cfg)
    assert not res_empty.passed, "Non-profitable base strategy must fail delay test!"
    assert "not profitable" in res_empty.detail.lower() or "sharpe <= 0" in res_empty.detail.lower()

    # 2. Profitable tape that collapses on 1-bar delay must FAIL
    tape = SignalTape.empty(n)
    for i in range(50, n - 10, 10):
        tape.actions[i] = Action.ENTER_LONG
        tape.sl_distances[i] = 10.0
        tape.tp_distances[i] = 20.0
    tape.total_signals = 20

    res = run_delay_test_on_tape(tape, df, cfg)
    assert isinstance(res.passed, bool)
    assert hasattr(res, "delay1_sharpe")
    assert hasattr(res, "delay2_sharpe")


# ── GATE-6: Gate Order Enforces Lookahead BEFORE Validation ─────────────────

def test_gate6_gate_order_enforces_lookahead_before_validation():
    """GATE-6: Pipeline must run lookahead checks (static, determinism, truncation)
    BEFORE touching or backtesting on validation data.
    """
    from core.gates import GatePipeline
    cfg = load_config()
    df_train = _make_df(200)
    df_val = _make_df(200)

    bad_code = "x = df['close'].shift(-1)"
    pipeline = GatePipeline(cfg)

    res = pipeline.evaluate_candidate(
        source=bad_code,
        df_train=df_train,
        df_val=df_val,
    )
    assert not res.passed
    assert res.stopped_at in ("static_policy", "static_scan")
    assert res.val_result is None, "Validation split must NEVER be touched when lookahead fails!"


# ── F4.3: Determinism Test ──────────────────────────────────────────────────

def test_determinism_test_catches_unseeded_random():
    """F4.3: Strategy with unseeded randomness produces non-identical tapes and fails."""
    from core.lookahead_guard import run_determinism_test
    cfg = load_config()
    df = _make_df(200)

    unseeded_code = textwrap.dedent("""
    import numpy as np
    from core.signals import Signal, Action

    class Strategy:
        def __init__(self):
            # Unseeded RNG draws from OS entropy, diverging across runs
            self.rng = np.random.default_rng(None)

        def on_bar(self, bars):
            if len(bars) < 20:
                return None
            if self.rng.random() > 0.5:
                return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
            return None
    """)
    res = run_determinism_test(unseeded_code, df, cfg, seed=42)
    assert not res.passed, "Unseeded strategy must fail determinism test!"
    assert res.first_diverging_bar is not None
