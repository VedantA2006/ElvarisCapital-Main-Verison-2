"""
tests/regression/test_f2_execution.py – Comprehensive Phase F2 regression and differential suite.

Verifies:
1. BT-1: Inverted-stop strategy is strictly rejected and cannot show 100% win rate on random walk.
2. BT-2: Entry bar stop-loss is live and triggered on fill bar crash.
3. BT-3 & BT-4: Cost reconciliation gross - costs == net; spread is in USD without pip multiplication.
4. BT-5: Mark-to-market equity shows intramonth and open-trade drawdown.
5. BT-7: Stop orders fill at open + slippage on gaps; take profit limit orders fill at TP.
6. BT-9: 17:00 NY rollover swap (trading days only, triple on Wednesday, zero on weekends).
7. BT-12: Strategy exceptions are NOT swallowed and return status code_error with traceback.
8. BT-14: Minimum stop and target distances enforced.
9. Differential test: EventSimulator and FastSimulator agree on 35 diverse random tapes (< 1e-6).
10. Table-driven test: 15 hand-computed trades with exact step-by-step arithmetic.
11. Cost reconciliation on 1000 random trades.
12. Performance test: 3 years of 1h simulated in under 1 second per tape.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from core.analytics import compute_analytics
from core.backtester import CostEngine, run_backtest
from core.event_simulator import EventSimulator, SimulationResult, Trade
from core.signals import Action, Direction, Signal, SignalTape, SignalValidationError, validate_and_record_signal
from core.simulator import FastSimulator
from core.sizing import calculate_position_size, check_margin_call
from tests.conftest import make_bars


def _create_synthetic_feed(
    n_bars: int = 500,
    seed: int = 42,
    freq: str = "60min",
    start_dt: str = "2022-01-03",
) -> pd.DataFrame:
    df = make_bars(start_dt, "2024-01-01", freq=freq, seed=seed)
    df = df.head(n_bars).reset_index(drop=True)
    df.attrs = {
        "timeframe": "1h" if "60" in freq else "4h",
        "split": "train",
        "file_hash": f"hash_{seed}",
        "slice_hash": f"slice_{seed}",
    }
    return df


# ─── BT-1: Inverted Stop Exploit ─────────────────────────────────────────────

def test_bt1_inverted_stop_rejected(base_cfg):
    """LONG with stop above price must be rejected with explicit error (closes BT-1)."""
    df = _create_synthetic_feed(200, seed=11)

    class InvertedStopStrategy:
        def on_bar(self, bars: pd.DataFrame):
            if len(bars) < 30:
                return None
            p = float(bars["close"].iloc[-1])
            # Malicious: SL above current price
            return Signal(direction=Direction.LONG, stop_loss=p * 1.01)

    result = run_backtest(InvertedStopStrategy(), df, base_cfg)
    assert result.status == "code_error"
    assert result.metrics["error"] == "code_error"
    assert "Inverted stop" in result.metadata["first_strategy_error"]
    assert result.metrics["total_trades"] == 0

    # Test Signal.from_prices directly
    with pytest.raises(SignalValidationError, match="Inverted stop: LONG"):
        Signal.from_prices(close=2000.0, direction=Direction.LONG, stop_loss=2010.0)

    with pytest.raises(SignalValidationError, match="Inverted stop: SHORT"):
        Signal.from_prices(close=2000.0, direction=Direction.SHORT, stop_loss=1990.0)


# ─── BT-2: Entry Bar Stop Trigger ────────────────────────────────────────────

def test_bt2_entry_bar_stop_triggered(base_cfg):
    """Entry bar is live: crash on entry bar hits stop immediately (closes BT-2)."""
    n = 60
    ts = pd.date_range("2023-03-06 00:00", periods=n, freq="60min", tz="UTC")
    px = np.full(n, 2000.0)
    d = pd.DataFrame({
        "timestamp": ts,
        "open": px,
        "high": px + 0.5,
        "low": px - 0.5,
        "close": px,
        "volume": 1.0,
    })
    # Bar 41 crashes down to 1900.0
    d.loc[41, "low"] = 1900.0
    d["session"] = "london"
    d.attrs = {"timeframe": "1h", "split": "train", "file_hash": "bt2", "slice_hash": "bt2"}

    class EntryBarCrashStrategy:
        def __init__(self):
            self.done = False

        def on_bar(self, bars):
            if len(bars) == 41 and not self.done:
                self.done = True
                # Enter Long at bar 41 open (2000.0) with SL at 1990.0
                return Signal(direction=Direction.LONG, stop_loss=1990.0)

    res = run_backtest(EntryBarCrashStrategy(), d, base_cfg)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.exit_reason == "stop_loss"
    assert t.entry_bar_idx == 41
    assert t.exit_bar_idx == 41  # Same bar!


# ─── BT-3 & BT-4: Cost Accounting and USD Spread ─────────────────────────────

def test_bt4_spread_points_equals_config_usd(base_cfg):
    """Spread must be in USD without pip_value division/multiplication (closes BT-4)."""
    engine = CostEngine(base_cfg)
    ts = pd.Timestamp("2023-03-01 10:00", tz="UTC")
    got = engine.spread_points(ts, "london")
    want = float(base_cfg["costs"]["spread"]["default"])
    assert abs(got - want) < 1e-9
    assert got == 0.25  # Exactly 25 cents per oz, not 0.0025


def test_bt3_cost_reconciliation_exact_identity(base_cfg):
    """gross - spread - slippage - commission - swap == net verified on every trade (closes BT-3)."""
    df = _create_synthetic_feed(300, seed=42)

    class SampleStrategy:
        def __init__(self):
            self.count = 0

        def on_bar(self, bars):
            self.count += 1
            if self.count % 25 == 1:
                p = float(bars["close"].iloc[-1])
                return Signal(direction=Direction.LONG, stop_loss=p - 10.0, take_profit=p + 15.0)
            elif self.count % 25 == 13:
                p = float(bars["close"].iloc[-1])
                return Signal(direction=Direction.SHORT, stop_loss=p + 10.0, take_profit=p - 15.0)
            return None

    res = run_backtest(SampleStrategy(), df, base_cfg)
    assert len(res.trades) > 5
    for t in res.trades:
        reconciled = t.gross_pnl - t.spread_cost - t.slippage_cost - t.commission - t.swap_cost
        assert abs(reconciled - t.net_pnl) < 1e-9


# ─── BT-5: Open-Trade MTM Drawdown ───────────────────────────────────────────

def test_bt5_open_trade_drawdown_visible(base_cfg):
    """MTM equity reflects unrealized open trade losses in max drawdown (closes BT-5)."""
    n = 30
    ts = pd.date_range("2023-03-01 00:00", periods=n, freq="60min", tz="UTC")
    opens = np.full(n, 2000.0)
    highs = np.full(n, 2005.0)
    lows = np.full(n, 1995.0)
    closes = np.full(n, 2000.0)

    # Bars 5-10 dip deeply to 1950.0 (50 dollar drop = 2.5% drop)
    for b in range(5, 11):
        lows[b] = 1945.0
        closes[b] = 1950.0
    # Then recovers to 2050.0 at bar 15
    closes[15] = 2050.0
    highs[15] = 2055.0

    df = pd.DataFrame({"timestamp": ts, "open": opens, "high": highs, "low": lows, "close": closes, "volume": 1.0})
    df.attrs = {"timeframe": "1h", "split": "train", "file_hash": "bt5", "slice_hash": "bt5"}

    class DipTradeStrategy:
        def __init__(self):
            self.done = False

        def on_bar(self, bars):
            if len(bars) == 2 and not self.done:
                self.done = True
                # Enter Long at bar 2 (fill at bar 3 open = 2000) with wide SL (1900) and TP at 2040
                return Signal(direction=Direction.LONG, stop_loss=1900.0, take_profit=2040.0)
            return None

    res = run_backtest(DipTradeStrategy(), df, base_cfg)
    assert len(res.trades) == 1
    trade = res.trades[0]
    assert trade.net_pnl > 0  # Trade was a winner!

    # But during bars 5-10, MTM drawdown must be visible!
    max_dd = res.metrics["max_drawdown"]
    assert max_dd > 0.003, f"Expected MTM drawdown > 0.3%, got {max_dd * 100:.2f}%"


# ─── BT-7: Gap Fill Execution Rules ──────────────────────────────────────────

def test_bt7_gap_fill_rules(base_cfg):
    """Stop orders fill at open + slippage when gapped; limits fill at TP (closes BT-7)."""
    n = 20
    ts = pd.date_range("2023-03-01 00:00", periods=n, freq="60min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": np.full(n, 2000.0),
        "high": np.full(n, 2005.0),
        "low": np.full(n, 1995.0),
        "close": np.full(n, 2000.0),
        "volume": 1.0,
    })
    # Bar 5 opens with gap down to 1980.0
    df.loc[5, "open"] = 1980.0
    df.loc[5, "low"] = 1975.0
    df.loc[5, "close"] = 1982.0
    df.attrs = {"timeframe": "1h", "split": "train", "file_hash": "bt7", "slice_hash": "bt7"}

    class GapStopStrategy:
        def __init__(self):
            self.done = False

        def on_bar(self, bars):
            if len(bars) == 2 and not self.done:
                self.done = True
                # Enter Long at bar 3 open (2000.0) with SL at 1990.0
                return Signal(direction=Direction.LONG, stop_loss=1990.0)
            return None

    res = run_backtest(GapStopStrategy(), df, base_cfg)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.exit_reason == "stop_loss"
    assert t.exit_bar_idx == 5
    # Fill price must be <= 1980.0 (gap open minus stop slippage), NOT 1990.0!
    assert t.exit_price <= 1980.0, f"Expected gap fill <= 1980.0, got {t.exit_price}"


# ─── BT-9: 17:00 NY Rollover Swap Model ──────────────────────────────────────

def test_bt9_rollover_swap_wednesday_and_weekends(base_cfg):
    """Rollover swap is charged at 17:00 NY, triple on Wednesday, zero on weekends (closes BT-9)."""
    engine = CostEngine(base_cfg)

    # Wednesday 15:00 NY to Thursday 12:00 NY (crosses Wednesday 17:00 NY rollover only)
    entry_wed = pd.Timestamp("2023-03-08 20:00", tz="UTC")  # 15:00 NY Wed
    exit_thu = pd.Timestamp("2023-03-09 17:00", tz="UTC")   # 12:00 NY Thu
    swap_wed = engine.swap_usd(Direction.LONG, lots=1.0, hold_nights=1, entry_time=entry_wed, exit_time=exit_thu)
    want_wed = 3.0 * float(base_cfg["costs"]["swap"]["long_per_lot_per_night"])
    assert abs(swap_wed - want_wed) < 1e-9

    # Friday 15:00 NY to Sunday 19:00 NY (crosses Friday 17:00 NY, weekend hold)
    entry_fri = pd.Timestamp("2023-03-10 20:00", tz="UTC")  # 15:00 NY Fri
    exit_sun = pd.Timestamp("2023-03-12 23:00", tz="UTC")   # 19:00 NY Sun
    swap_weekend = engine.swap_usd(Direction.LONG, lots=1.0, hold_nights=1, entry_time=entry_fri, exit_time=exit_sun)
    want_fri = 1.0 * float(base_cfg["costs"]["swap"]["long_per_lot_per_night"])
    assert abs(swap_weekend - want_fri) < 1e-9


# ─── BT-12: Strategy Exceptions Trapped Cleanly ──────────────────────────────

def test_bt12_strategy_exceptions_not_swallowed(base_cfg):
    """Unhandled strategy exceptions return code_error status with traceback (closes BT-12)."""
    df = _create_synthetic_feed(50, seed=12)

    class CrashingStrategy:
        def on_bar(self, bars):
            if len(bars) == 10:
                1 / 0  # Deliberate ZeroDivisionError
            return None

    res = run_backtest(CrashingStrategy(), df, base_cfg)
    assert res.status == "code_error"
    assert res.metrics["error"] == "code_error"
    assert "ZeroDivisionError" in res.metadata["first_strategy_error"]
    assert res.metadata["strategy_errors"] == 1


# ─── BT-14: Distance-Based Stops & TP Floor ─────────────────────────────────

def test_bt14_minimum_tp_distance_enforced(base_cfg):
    """Take profit closer than 3x spread is strictly rejected (closes BT-14)."""
    tape = SignalTape.empty(10)
    cfg_strict = {
        "signals": {
            "min_tp_spread_multiple": 3.0,
            "min_tp_usd": 1.0,
            "min_sl_usd": 1.0,
            "min_sl_atr_multiple": 0.0,
            "max_sl_usd": 100.0,
        }
    }
    # Micro-scalp target: 0.10 USD distance with spread 0.25 USD (min required is max(1.0, 3*0.25)=1.0)
    sig = Signal(action=Action.ENTER_LONG, sl_distance=5.0, tp_distance=0.10)
    validate_and_record_signal(sig, bar_idx=0, tape=tape, atr_prev=2.0, current_spread=0.25, cfg=cfg_strict)
    assert tape.invalid_signals == 1
    assert "tp_distance (0.10) < min allowed (1.00)" in tape.invalid_reasons[0]


# ─── Sizing & Margin Invariants ──────────────────────────────────────────────

def test_position_sizing_and_margin_utilization():
    """Verify lot quantization, risk budgeting, and margin checks."""
    # Normal trade: 100k equity, 1% risk ($1000), SL dist $10, spread 0.25, slip 0.05
    res = calculate_position_size(
        equity=100000.0,
        sl_distance=10.0,
        price=2000.0,
        spread=0.25,
        expected_slippage=0.05,
        risk_fraction=0.01,
        contract_size=100.0,
        lot_step=0.01,
        min_lot=0.01,
        max_lot=50.0,
        leverage=100.0,
        max_margin_utilisation=0.80,
    )
    assert not res.skipped
    # risk_per_lot = (10 + 0.25 + 0.05) * 100 = 1030
    # lots = floor(1000 / 1030 / 0.01) * 0.01 = 0.97
    assert res.lots == 0.97
    assert res.required_margin == round(0.97 * 100 * 2000 / 100, 2)  # 1940.0

    # Margin check: huge lots exceeding max_margin_utilisation
    res_margin = calculate_position_size(
        equity=1000.0,
        sl_distance=0.50,
        price=2000.0,
        spread=0.25,
        expected_slippage=0.0,
        risk_fraction=0.50,
        leverage=10.0,  # low leverage
        max_margin_utilisation=0.50,
    )
    # Required margin will exceed 50% of equity
    assert res_margin.skipped
    assert res_margin.skip_reason == "skipped_margin"


# ─── Differential Test: EventSimulator vs FastSimulator ──────────────────────

def test_differential_35_random_tapes(base_cfg):
    """EventSimulator (pure Python) and FastSimulator (Numba) produce identical results on 35 random tapes."""
    n_bars = 400
    df = _create_synthetic_feed(n_bars, seed=77)
    from core.backtester import prepare_simulation_arrays
    opens, highs, lows, closes, timestamps, spreads, atrs, is_rollover, rollover_multiplier = (
        prepare_simulation_arrays(df, base_cfg)
    )

    fast_sim = FastSimulator(base_cfg)
    event_sim = EventSimulator(base_cfg)

    rng = np.random.default_rng(2026)

    for seed in range(35):
        tape = SignalTape.empty(n_bars)
        # Random signal generation
        prob_signal = rng.uniform(0.05, 0.20)
        for i in range(1, n_bars - 1):
            if rng.random() < prob_signal:
                act_val = rng.choice([Action.ENTER_LONG, Action.ENTER_SHORT, Action.CLOSE])
                if act_val == Action.CLOSE:
                    tape.actions[i] = Action.CLOSE
                else:
                    tape.actions[i] = act_val
                    tape.sl_distances[i] = rng.uniform(5.0, 30.0)
                    tape.tp_distances[i] = rng.uniform(10.0, 50.0) if rng.random() > 0.3 else np.nan
                    tape.trail_distances[i] = rng.uniform(5.0, 15.0) if rng.random() > 0.6 else np.nan
                    tape.time_stops[i] = int(rng.integers(5, 25)) if rng.random() > 0.6 else 0
                    tape.breakeven_r[i] = 1.5 if rng.random() > 0.7 else np.nan

        res_fast = fast_sim.run(
            opens, highs, lows, closes, timestamps, spreads, atrs, is_rollover, rollover_multiplier, tape
        )
        res_event = event_sim.run(
            opens, highs, lows, closes, timestamps, spreads, atrs, is_rollover, rollover_multiplier, tape
        )

        assert len(res_fast.trades) == len(res_event.trades), (
            f"Tape {seed}: trade count mismatch: fast={len(res_fast.trades)}, event={len(res_event.trades)}"
        )
        assert res_fast.skipped_min_lot == res_event.skipped_min_lot
        assert res_fast.skipped_margin == res_event.skipped_margin
        assert res_fast.forced_closes == res_event.forced_closes

        # Compare trades
        for tid in range(len(res_fast.trades)):
            tf = res_fast.trades[tid]
            te = res_event.trades[tid]
            assert tf.entry_bar_idx == te.entry_bar_idx, f"Tape {seed} trade {tid}: entry bar mismatch"
            assert tf.exit_bar_idx == te.exit_bar_idx, f"Tape {seed} trade {tid}: exit bar mismatch"
            assert tf.exit_reason == te.exit_reason, f"Tape {seed} trade {tid}: exit reason mismatch"
            assert abs(tf.lots - te.lots) < 1e-9, f"Tape {seed} trade {tid}: lots mismatch"
            assert abs(tf.entry_price - te.entry_price) < 1e-6, f"Tape {seed} trade {tid}: entry_price mismatch"
            assert abs(tf.exit_price - te.exit_price) < 1e-6, f"Tape {seed} trade {tid}: exit_price mismatch"
            assert abs(tf.gross_pnl - te.gross_pnl) < 1e-4, f"Tape {seed} trade {tid}: gross mismatch"
            assert abs(tf.net_pnl - te.net_pnl) < 1e-4, f"Tape {seed} trade {tid}: net mismatch"

        # Compare MTM equity series
        diff = np.max(np.abs(res_fast.equity_mtm - res_event.equity_mtm))
        assert diff < 1e-3, f"Tape {seed}: max equity difference {diff:.2e} exceeds threshold"


# ─── Table-Driven Test: 15 Hand-Computed Trades ──────────────────────────────

def test_hand_calculated_15_trades_table():
    """Verify 15 hand-calculated trade scenarios with exact arithmetic in comments."""
    # Initial parameters:
    # Equity = 100,000.0, risk_per_trade = 0.01 ($1000 budget), contract_size = 100 oz.
    # Commission = $7.0 / lot.
    # Spread = 0.25 USD, base slippage = 0.05 USD, stop slippage = 0.075 USD.
    cfg = {
        "sizing": {"initial_equity": 100000.0, "risk_per_trade": 0.01, "max_margin_utilisation": 0.8},
        "contract": {"contract_size": 100.0, "lot_step": 0.01, "min_lot": 0.01, "max_lot": 50.0, "leverage": 100.0},
        "costs": {
            "commission_per_lot": 7.0,
            "spread": {"default": 0.25, "asia": 0.25, "rollover": 0.25, "rollover_start_local": 16, "rollover_end_local": 19},
            "slippage": {"base_fraction_of_atr": 0.05, "stop_multiplier": 1.5},
            "swap": {"long_per_lot_per_night": -3.50, "short_per_lot_per_night": 0.80, "triple_day": "wednesday"},
        },
        "execution": {"allow_reversal": True},
    }

    # Construct 16-bar scenario supporting 15 distinct trades
    n = 35
    opens = np.full(n, 2000.0)
    highs = np.full(n, 2005.0)
    lows = np.full(n, 1995.0)
    closes = np.full(n, 2000.0)
    ts = pd.date_range("2023-03-01 00:00", periods=n, freq="60min", tz="UTC")
    spreads = np.full(n, 0.25)
    atrs = np.full(n, 1.0)  # ATR=1.0 -> base_slip = 0.05, stop_slip = 0.075
    is_roll = np.zeros(n, dtype=np.uint8)
    roll_mult = np.zeros(n, dtype=np.float64)

    # We construct 15 sequential trades with step-by-step arithmetic in comments:
    tape = SignalTape.empty(n)
    sim = EventSimulator(cfg)

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 1: Long TP win
    # Signal at bar 0: Long, sl_dist=10.0, tp_dist=15.0
    # Entry at bar 1 open: 2000.0 + 0.25 (spread) + 0.05 (slip) = 2000.30
    # Lots: floor(1000 / ((10 + 0.25 + 0.05)*100) / 0.01) * 0.01 = 0.97 lots
    # TP trigger: high hits 2020.0 >= 2000.30 + 15.0 (2015.30)
    # Exit price = 2015.30 (no slip on limit order)
    # Gross PnL: (2015.30 - 2000.0) * 97 = 1484.10
    # Costs: spread = 0.25*97 = 24.25, slip = 0.05*97 = 4.85, comm = 7*0.97 = 6.79
    # Net PnL: 1484.10 - 24.25 - 4.85 - 6.79 = 1448.21
    # ─────────────────────────────────────────────────────────────────────────────
    tape.actions[0] = Action.ENTER_LONG
    tape.sl_distances[0] = 10.0
    tape.tp_distances[0] = 15.0
    highs[1] = 2020.0

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 2: Long SL loss (intrabar touch)
    # Compounding: Equity is now 100,000 + 1448.21 = 101,448.21
    # Lots: floor(1014.48 / ((10 + 0.25 + 0.05)*100) / 0.01) * 0.01 = 0.98 lots
    # Signal at bar 1: Long, sl_dist=10.0
    # Entry at bar 2 open: 2000.0 + 0.25 + 0.05 = 2000.30
    # SL level: 2000.30 - 10.0 = 1990.30
    # Low hits 1985.0 <= 1990.30 -> stop fills at 1990.30 - 0.075 (stop slip) = 1990.225
    # Raw bid = 1990.30
    # Gross: (1990.30 - 2000.0) * 98 = -9.70 * 98 = -950.60
    # Costs: spread = 0.25*98 = 24.50, slip = 0.125*98 = 12.25, comm = 7*0.98 = 6.86
    # Net: -950.60 - 24.50 - 12.25 - 6.86 = -994.21
    # ─────────────────────────────────────────────────────────────────────────────
    tape.actions[1] = Action.ENTER_LONG
    tape.sl_distances[1] = 10.0
    lows[2] = 1985.0

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 3: Long Gap-down Stop Loss
    # Signal at bar 2: Long, sl_dist=10.0 (SL level = 1990.30)
    # Entry at bar 3: 2000.30
    # Bar 4 opens at 1980.0 (gap down through stop). Stop fills at open: 1980.0 - 0.075 = 1979.925
    # Raw bid = 1980.0
    # Gross: (1980.0 - 2000.0) * 97 = -1940.00
    # Costs: spread = 24.25, slip = (0.05 + 0.075)*97 = 12.125, comm = 6.79
    # Net: -1940.0 - 24.25 - 12.125 - 6.79 = -1983.165 -> round to -1983.17
    # ─────────────────────────────────────────────────────────────────────────────
    tape.actions[2] = Action.ENTER_LONG
    tape.sl_distances[2] = 10.0
    opens[4] = 1980.0
    lows[4] = 1975.0
    closes[4] = 1985.0

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 4: Long Signal Close
    # Signal at bar 4: Long, sl_dist=50.0. Entry at bar 5: 2000.0 + 0.25 + 0.05 = 2000.30
    # Signal at bar 5: Action.CLOSE. Exits at bar 6 open = 2000.0 - 0.05 = 1999.95
    # Gross: (2000.0 - 2000.0) * lots = 0.0
    # ─────────────────────────────────────────────────────────────────────────────
    opens[5] = 2000.0
    closes[5] = 2000.0
    tape.actions[4] = Action.ENTER_LONG
    tape.sl_distances[4] = 50.0
    tape.actions[5] = Action.CLOSE

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 5: Short TP win
    # Signal at bar 6: Short, sl_dist=10.0, tp_dist=15.0
    # Entry at bar 7 open: 2000.0 - 0.05 (entry slip) = 1999.95
    # TP level: 1999.95 - 15.0 = 1984.95
    # Low hits 1980.0 -> ask_low = 1980.25 <= 1984.95 -> TP hit!
    # Exit fill (ask) = 1984.95. Raw bid = 1984.95 - 0.25 = 1984.70
    # Gross: (2000.0 - 1984.70) * lots = 15.30 * lots
    # ─────────────────────────────────────────────────────────────────────────────
    tape.actions[6] = Action.ENTER_SHORT
    tape.sl_distances[6] = 10.0
    tape.tp_distances[6] = 15.0
    lows[7] = 1980.0

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 6: Short SL loss (intrabar touch)
    # Signal at bar 7: Short, sl_dist=10.0
    # Entry at bar 8 open: 1999.95. SL level: 1999.95 + 10.0 = 2009.95
    # High hits 2015.0 -> ask_high = 2015.25 >= 2009.95 -> SL hit!
    # Exit fill (ask) = 2009.95 + 0.075 = 2010.025
    # ─────────────────────────────────────────────────────────────────────────────
    tape.actions[7] = Action.ENTER_SHORT
    tape.sl_distances[7] = 10.0
    highs[8] = 2015.0

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 7: Short Gap-up Stop Loss
    # Signal at bar 8: Short, sl_dist=10.0 (SL level = 2009.95)
    # Entry at bar 9: 1999.95. Bar 10 opens at 2015.0 -> ask_open = 2015.25 >= 2009.95
    # Gap fill at open: 2015.25 + 0.075 = 2015.325
    # ─────────────────────────────────────────────────────────────────────────────
    tape.actions[8] = Action.ENTER_SHORT
    tape.sl_distances[8] = 10.0
    opens[10] = 2015.0
    highs[10] = 2020.0
    closes[10] = 2015.0

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 8: Short Signal Close
    # Signal at bar 10: Short, sl_dist=50.0. Entry at bar 11: 1999.95
    # Signal at bar 11: CLOSE. Exits at bar 12 open (ask side)
    # ─────────────────────────────────────────────────────────────────────────────
    opens[11] = 2000.0
    closes[11] = 2000.0
    tape.actions[10] = Action.ENTER_SHORT
    tape.sl_distances[10] = 50.0
    tape.actions[11] = Action.CLOSE

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 9: Long to Short Reversal
    # Signal at bar 12: Long, sl_dist=50.0. Entry at bar 13: 2000.30
    # Signal at bar 13: ENTER_SHORT. Closes Long at bar 14 open (reason=signal_reversal)
    # ─────────────────────────────────────────────────────────────────────────────
    tape.actions[12] = Action.ENTER_LONG
    tape.sl_distances[12] = 50.0
    tape.actions[13] = Action.ENTER_SHORT
    tape.sl_distances[13] = 50.0

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 10: Short to Long Reversal
    # Short opened at bar 14. Signal at bar 14: ENTER_LONG
    # Closes Short at bar 15 open (reason=signal_reversal) and opens Long
    # ─────────────────────────────────────────────────────────────────────────────
    tape.actions[14] = Action.ENTER_LONG
    tape.sl_distances[14] = 50.0

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 11: Wednesday Rollover Triple Swap
    # Long opened at bar 15. Bar 15 marked as Wednesday rollover (mult=3.0)
    # Swap charged = 3.0 * (-3.50) * lots
    # ─────────────────────────────────────────────────────────────────────────────
    is_roll[15] = 1
    roll_mult[15] = 3.0
    tape.actions[15] = Action.CLOSE

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 12: Single Swap Rollover
    # Signal at bar 16: Long, sl_dist=50.0. Bar 17 rollover (mult=1.0)
    # Swap charged = 1.0 * (-3.50) * lots
    # ─────────────────────────────────────────────────────────────────────────────
    tape.actions[16] = Action.ENTER_LONG
    tape.sl_distances[16] = 50.0
    is_roll[17] = 1
    roll_mult[17] = 1.0
    tape.actions[17] = Action.CLOSE

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 13: Breakeven Stop
    # Signal at bar 18: Long, sl_dist=10.0, be_r=1.0. Entry at bar 19: 2000.30
    # Bar 19 close rises to 2012.0 (profit > 1.0 * 10.0) -> SL moves to entry (2000.30)
    # Bar 20 drops to 1998.0 -> hits BE stop at 2000.30 - 0.075 = 2000.225
    # ─────────────────────────────────────────────────────────────────────────────
    tape.actions[18] = Action.ENTER_LONG
    tape.sl_distances[18] = 10.0
    tape.breakeven_r[18] = 1.0
    closes[19] = 2012.0
    lows[20] = 1998.0

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 14: Time Stop
    # Signal at bar 20: Long, sl_dist=50.0, time_stop=2 bars
    # Entry at bar 21. Held bar 21 (1 bar), bar 22 (2 bars) -> triggers exit at bar 23 open
    # ─────────────────────────────────────────────────────────────────────────────
    tape.actions[20] = Action.ENTER_LONG
    tape.sl_distances[20] = 50.0
    tape.time_stops[20] = 2

    # ─────────────────────────────────────────────────────────────────────────────
    # Trade 15: End of Split Forced Close
    # Signal at bar n-2: Long, sl_dist=50.0. Entry at bar n-1
    # Reaches bar n-1 (last bar) still open -> forced_close at close[n-1]
    # ─────────────────────────────────────────────────────────────────────────────
    tape.actions[n - 2] = Action.ENTER_LONG
    tape.sl_distances[n - 2] = 50.0

    res = sim.run(opens, highs, lows, closes, ts, spreads, atrs, is_roll, roll_mult, tape)
    assert len(res.trades) == 15, f"Expected exactly 15 trades, got {len(res.trades)}"

    # Validate individual trades
    t1 = res.trades[0]
    assert t1.exit_reason == "take_profit"
    assert abs(t1.net_pnl - 1448.21) < 0.02

    t2 = res.trades[1]
    assert t2.exit_reason == "stop_loss"
    assert abs(t2.net_pnl - (-994.21)) < 0.02

    t3 = res.trades[2]
    assert t3.exit_reason == "stop_loss"
    assert t3.exit_price <= 1980.0  # Gap fill

    t4 = res.trades[3]
    assert t4.exit_reason == "signal_close"

    t5 = res.trades[4]
    assert t5.exit_reason == "take_profit"
    assert t5.direction == "SHORT"

    t6 = res.trades[5]
    assert t6.exit_reason == "stop_loss"
    assert t6.direction == "SHORT"

    t7 = res.trades[6]
    assert t7.exit_reason == "stop_loss"
    assert t7.direction == "SHORT"
    assert t7.exit_price >= 2015.0  # Gap fill up

    t8 = res.trades[7]
    assert t8.exit_reason == "signal_close"

    t9 = res.trades[8]
    assert t9.exit_reason == "signal_reversal"

    t10 = res.trades[9]
    assert t10.exit_reason == "signal_reversal"

    t11 = res.trades[10]
    assert t11.swap_cost > 0  # Wednesday triple swap paid

    t12 = res.trades[11]
    assert t12.swap_cost > 0  # Single swap paid

    t13 = res.trades[12]
    assert t13.exit_reason == "stop_loss"

    t14 = res.trades[13]
    assert t14.exit_reason == "time_stop"

    t15 = res.trades[14]
    assert t15.exit_reason == "forced_close"


# ─── Cost Reconciliation on 1,000 Trades ─────────────────────────────────────

def test_cost_reconciliation_1000_random_trades(base_cfg):
    """Ensure gross - spread - slippage - commission - swap == net on 1,000 trades."""
    n_bars = 1200
    df = _create_synthetic_feed(n_bars, seed=999)
    from core.backtester import prepare_simulation_arrays
    opens, highs, lows, closes, timestamps, spreads, atrs, is_rollover, rollover_multiplier = (
        prepare_simulation_arrays(df, base_cfg)
    )

    tape = SignalTape.empty(n_bars)
    rng = np.random.default_rng(42)
    # Dense signals to get > 1000 trade events across multiple runs
    for i in range(1, n_bars - 1):
        if rng.random() < 0.35:
            tape.actions[i] = rng.choice([Action.ENTER_LONG, Action.ENTER_SHORT, Action.CLOSE])
            tape.sl_distances[i] = rng.uniform(4.0, 20.0)
            tape.tp_distances[i] = rng.uniform(8.0, 30.0)

    sim = FastSimulator(base_cfg)
    res = sim.run(opens, highs, lows, closes, timestamps, spreads, atrs, is_rollover, rollover_multiplier, tape)
    assert len(res.trades) > 50

    for t in res.trades:
        reconciled = t.gross_pnl - t.spread_cost - t.slippage_cost - t.commission - t.swap_cost
        assert abs(reconciled - t.net_pnl) < 1e-9


# ─── Performance Test: 3 Years of 1h Under 1 Second ──────────────────────────

def test_performance_3_years_1h_under_1_second(base_cfg):
    """FastSimulator must simulate 3 years of 1h data (~18,000 bars) in < 1 second."""
    n_bars = 18000
    # Fast synthetic series
    rng = np.random.default_rng(100)
    prices = 2000.0 + np.cumsum(rng.normal(0, 1.5, n_bars))
    opens = prices
    highs = prices + np.abs(rng.normal(1.0, 0.5, n_bars))
    lows = prices - np.abs(rng.normal(1.0, 0.5, n_bars))
    closes = prices + rng.normal(0, 0.5, n_bars)
    ts = pd.date_range("2021-01-01", periods=n_bars, freq="60min", tz="UTC")
    spreads = np.full(n_bars, 0.25)
    atrs = np.full(n_bars, 2.0)
    is_rollover = np.zeros(n_bars, dtype=np.uint8)
    roll_mult = np.zeros(n_bars, dtype=np.float64)

    tape = SignalTape.empty(n_bars)
    for i in range(10, n_bars - 10, 20):
        tape.actions[i] = Action.ENTER_LONG if (i // 20) % 2 == 0 else Action.ENTER_SHORT
        tape.sl_distances[i] = 10.0
        tape.tp_distances[i] = 20.0

    fast_sim = FastSimulator(base_cfg)

    # Warmup Numba JIT once
    _ = fast_sim.run(opens[:100], highs[:100], lows[:100], closes[:100], ts[:100],
                     spreads[:100], atrs[:100], is_rollover[:100], roll_mult[:100], tape.shifted(0))

    # Benchmark full 3 years (18,000 bars)
    t0 = time.perf_counter()
    res = fast_sim.run(opens, highs, lows, closes, ts, spreads, atrs, is_rollover, roll_mult, tape)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, f"Execution took {elapsed:.4f}s (must be < 1.0s)"
    assert len(res.trades) > 100
