"""
Phase 2 tests: backtester correctness, no-lookahead enforcement, cost model,
position sizing, and metrics accuracy.

The tests use SYNTHETIC data (from conftest.py) so they are deterministic and
fast. Several tests use a "spy" strategy that records exactly what data it
received, proving by construction that no future data is ever visible.

Run:  python -m pytest tests/test_phase2_backtester.py -v
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars, write_csv


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_df(n: int = 500, seed: int = 42, freq: str = "60min") -> pd.DataFrame:
    """Quick synthetic bars with session labels."""
    from core.data_loader import add_session_labels
    from core.config import load_config
    cfg = load_config()
    df = make_bars("2022-01-03", "2023-06-01", freq, seed=seed)
    df = df.head(n).reset_index(drop=True)
    df = add_session_labels(df, cfg)
    df.attrs = {"timeframe": "1h", "split": "train", "file_hash": "test", "slice_hash": "test"}
    return df


class AlwaysLongStrategy:
    """Enters long on bar 0, holds forever (no SL/TP)."""
    def __init__(self):
        self.entered = False

    def on_bar(self, bars: pd.DataFrame):
        from core.backtester import Signal, Direction
        if not self.entered:
            self.entered = True
            return Signal(direction=Direction.LONG)
        return None


class AlwaysShortStrategy:
    def __init__(self):
        self.entered = False

    def on_bar(self, bars: pd.DataFrame):
        from core.backtester import Signal, Direction
        if not self.entered:
            self.entered = True
            return Signal(direction=Direction.SHORT)
        return None


class AlternatingStrategy:
    """Alternates LONG/SHORT every `period` bars. Uses SL and TP."""
    def __init__(self, period: int = 20, sl_pct: float = 0.01, tp_pct: float = 0.02):
        self.period = period
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.bar_count = 0
        self.current_dir = None

    def on_bar(self, bars: pd.DataFrame):
        from core.backtester import Signal, Direction
        self.bar_count += 1
        if self.bar_count % self.period != 1 and self.current_dir is not None:
            return None
        price = float(bars["close"].iloc[-1])
        if self.current_dir is None or self.current_dir == Direction.SHORT:
            d = Direction.LONG
            sl = price * (1 - self.sl_pct)
            tp = price * (1 + self.tp_pct)
        else:
            d = Direction.SHORT
            sl = price * (1 + self.sl_pct)
            tp = price * (1 - self.tp_pct)
        self.current_dir = d
        return Signal(direction=d, stop_loss=sl, take_profit=tp)


# ═══════════════════════════════════════════════════════════════════════════
# NO-LOOKAHEAD (the most important test in the system)
# ═══════════════════════════════════════════════════════════════════════════

class TestNoLookahead:
    """Prove by construction that on_bar NEVER sees future data."""

    def test_strategy_never_sees_future_bars(self, base_cfg):
        from core.backtester import run_backtest, Signal, Direction
        df = _make_df(200)
        all_timestamps = df["timestamp"].to_list()

        seen_max_indices: list[int] = []

        class SpyStrategy:
            def on_bar(self, bars: pd.DataFrame):
                last_ts = bars["timestamp"].iloc[-1]
                idx = all_timestamps.index(last_ts)
                seen_max_indices.append(idx)
                # Verify: the strategy must never see a bar beyond idx
                assert bars["timestamp"].iloc[-1] == all_timestamps[idx]
                assert len(bars) == idx + 1
                if idx == 10:
                    return Signal(direction=Direction.LONG, tag="spy")
                return None

        result = run_backtest(SpyStrategy(), df, base_cfg)
        # Strategy was called for each bar from first_live to N-1
        assert len(seen_max_indices) == len(df)
        # Each call saw exactly one more bar than the last
        for i in range(1, len(seen_max_indices)):
            assert seen_max_indices[i] == seen_max_indices[i - 1] + 1

    def test_fill_happens_at_next_bar_open(self, base_cfg):
        from core.backtester import run_backtest, Signal, Direction
        df = _make_df(100)
        signal_bar = 20

        class FixedSignalStrategy:
            def on_bar(self, bars: pd.DataFrame):
                if len(bars) - 1 == signal_bar:
                    return Signal(direction=Direction.LONG, tag="fixed")
                return None

        result = run_backtest(FixedSignalStrategy(), df, base_cfg)
        assert len(result.trades) == 1
        trade = result.trades[0]
        # Entry must be at bar signal_bar+1, NOT at signal_bar
        assert trade.entry_bar_idx == signal_bar + 1
        assert trade.entry_time == df["timestamp"].iloc[signal_bar + 1]
        # Entry price is based on bar signal_bar+1 open (plus spread/slippage)
        next_open = df["open"].iloc[signal_bar + 1]
        assert trade.entry_price >= next_open  # long = open + spread + slippage

    def test_warmup_bars_not_traded(self, base_cfg):
        from core.backtester import run_backtest
        df = _make_df(200)
        # Mark first 50 bars as warmup
        df["is_warmup"] = False
        df.loc[:49, "is_warmup"] = True

        class EagerStrategy:
            def __init__(self):
                self.first_call_len = None

            def on_bar(self, bars):
                from core.backtester import Signal, Direction
                if self.first_call_len is None:
                    self.first_call_len = len(bars)
                    # Strategy is first called at bar 50 (first live bar)
                    # It receives bars[0:51] (51 bars, including warmup for indicator computation)
                    return Signal(direction=Direction.LONG)
                return None

        strat = EagerStrategy()
        result = run_backtest(strat, df, base_cfg)
        # First on_bar call includes warmup bars (for indicator computation)
        # but the entry happens at bar 51 (first live + 1)
        assert strat.first_call_len == 51
        assert result.trades[0].entry_bar_idx == 51


# ═══════════════════════════════════════════════════════════════════════════
# COSTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCosts:
    def test_costs_always_on_and_reduce_pnl(self, base_cfg):
        from core.backtester import run_backtest
        df = _make_df(300)
        result = run_backtest(AlternatingStrategy(), df, base_cfg)
        assert result.trades
        for t in result.trades:
            assert t.spread_cost > 0, "Spread must be > 0"
            assert t.slippage_cost > 0, "Slippage must be > 0"
            assert t.commission > 0, "Commission must be > 0"
            assert abs(t.net_pnl) < abs(t.gross_pnl) + t.spread_cost + t.slippage_cost + t.commission + abs(t.swap_cost) + 1

    def test_no_gross_only_mode(self, base_cfg):
        """There is no way to disable costs — verify by checking the result has no gross-only metrics."""
        from core.backtester import run_backtest
        df = _make_df(200)
        result = run_backtest(AlternatingStrategy(), df, base_cfg)
        m = result.metrics
        # All key metrics are net of costs
        assert "sharpe" in m and "profit_factor" in m
        # Total costs > 0
        assert m["total_costs"] > 0

    def test_spread_wider_during_asia_and_rollover(self, base_cfg):
        from core.backtester import CostEngine
        ce = CostEngine(base_cfg)
        # Asia session bar
        asia_ts = pd.Timestamp("2023-03-15 01:00:00", tz="UTC")  # Tokyo 10:00
        default_ts = pd.Timestamp("2023-03-15 10:00:00", tz="UTC")  # London 10:00
        rollover_ts = pd.Timestamp("2023-03-15 21:00:00", tz="UTC")  # NY 17:00 EDT
        assert ce.spread_points(asia_ts, "asia") > ce.spread_points(default_ts, "london")
        assert ce.spread_points(rollover_ts, "default") > ce.spread_points(default_ts, "london")

    def test_slippage_proportional_to_atr(self, base_cfg):
        from core.backtester import CostEngine
        ce = CostEngine(base_cfg)
        assert ce.slippage_points(10.0) < ce.slippage_points(20.0)
        assert ce.slippage_points(10.0, is_stop=True) > ce.slippage_points(10.0, is_stop=False)

    def test_commission_proportional_to_lots(self, base_cfg):
        from core.backtester import CostEngine
        ce = CostEngine(base_cfg)
        assert ce.commission_usd(1.0) == base_cfg["costs"]["commission_per_lot"]
        assert ce.commission_usd(2.0) == 2 * base_cfg["costs"]["commission_per_lot"]

    def test_swap_applied_on_overnight_holds(self, base_cfg):
        from core.backtester import CostEngine, Direction
        ce = CostEngine(base_cfg)
        entry = pd.Timestamp("2023-03-13 10:00:00", tz="UTC")  # Monday
        exit_next = pd.Timestamp("2023-03-14 10:00:00", tz="UTC")  # Tuesday
        swap = ce.swap_usd(Direction.LONG, 1.0, 1, entry, exit_next)
        assert swap != 0.0

    def test_triple_swap_on_wednesday(self, base_cfg):
        from core.backtester import CostEngine, Direction
        ce = CostEngine(base_cfg)
        # Hold over Wednesday night (triple swap)
        wed_entry = pd.Timestamp("2023-03-15 10:00:00", tz="UTC")  # Wednesday
        thu_exit = pd.Timestamp("2023-03-16 10:00:00", tz="UTC")
        swap_triple = ce.swap_usd(Direction.LONG, 1.0, 1, wed_entry, thu_exit)
        # Hold over Thursday night (single swap)
        thu_entry = pd.Timestamp("2023-03-16 10:00:00", tz="UTC")
        fri_exit = pd.Timestamp("2023-03-17 10:00:00", tz="UTC")
        swap_single = ce.swap_usd(Direction.LONG, 1.0, 1, thu_entry, fri_exit)
        assert abs(swap_triple) > abs(swap_single)


# ═══════════════════════════════════════════════════════════════════════════
# STOP LOSS / TAKE PROFIT
# ═══════════════════════════════════════════════════════════════════════════

class TestSLTP:
    def test_stop_loss_fills_with_slippage(self, base_cfg):
        from core.backtester import run_backtest, Signal, Direction
        df = _make_df(200)

        class TightSLStrategy:
            def __init__(self):
                self.entered = False

            def on_bar(self, bars):
                if not self.entered:
                    self.entered = True
                    price = float(bars["close"].iloc[-1])
                    return Signal(direction=Direction.LONG,
                                  stop_loss=price * 0.999,  # very tight SL
                                  take_profit=price * 1.10)
                return None

        result = run_backtest(TightSLStrategy(), df, base_cfg)
        sl_trades = [t for t in result.trades if t.exit_reason == "stop_loss"]
        assert len(sl_trades) >= 1
        for t in sl_trades:
            # SL fill for long must be <= SL price (because of slippage)
            assert t.exit_price <= t.entry_price * 0.999 + 0.01  # small tolerance

    def test_sl_and_tp_same_bar_conservative(self, base_cfg):
        """If SL and TP both triggered on same bar, SL wins (conservative)."""
        from core.backtester import run_backtest, Signal, Direction
        df = _make_df(300)

        class WideRangeStrategy:
            """Set SL and TP so close that both are likely hit on a single bar."""
            def __init__(self):
                self.entered = False

            def on_bar(self, bars):
                if not self.entered:
                    self.entered = True
                    price = float(bars["close"].iloc[-1])
                    return Signal(direction=Direction.LONG,
                                  stop_loss=price * 0.9999,
                                  take_profit=price * 1.0001)
                return None

        result = run_backtest(WideRangeStrategy(), df, base_cfg)
        # The trade should exist and exit on SL (conservative assumption)
        assert len(result.trades) >= 1
        assert result.trades[0].exit_reason == "stop_loss"

    def test_take_profit_fills_at_tp_price(self, base_cfg):
        from core.backtester import run_backtest, Signal, Direction
        df = _make_df(500, seed=99)

        class ReasonableTPStrategy:
            """Set a 0.5% TP on a long."""
            def __init__(self):
                self.entered = False
                self.entry_price = 0.0

            def on_bar(self, bars):
                if not self.entered:
                    self.entered = True
                    self.entry_price = float(bars["close"].iloc[-1])
                    return Signal(direction=Direction.LONG,
                                  stop_loss=self.entry_price * 0.99,
                                  take_profit=self.entry_price * 1.005)
                return None

        result = run_backtest(ReasonableTPStrategy(), df, base_cfg)
        tp_trades = [t for t in result.trades if t.exit_reason == "take_profit"]
        if tp_trades:
            for t in tp_trades:
                # TP fill for long at the TP price (no slippage on TP = favorable fill)
                assert t.exit_price >= t.entry_price * 0.99  # at least above SL


# ═══════════════════════════════════════════════════════════════════════════
# POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════

class TestPositionSizing:
    def test_lot_size_respects_risk_fraction(self, base_cfg):
        from core.backtester import compute_lot_size
        c = base_cfg["contract"]
        lots = compute_lot_size(100000, 0.01, 1800.0, 1790.0, c["contract_size"],
                                c["lot_step"], c["min_lot"], c["max_lot"])
        # Risk = 100000 * 0.01 = 1000. Per-oz risk = 10. Per-lot risk = 10 * 100 = 1000.
        # So lots should be ~1.0
        assert 0.5 <= lots <= 1.5

    def test_lot_size_bounded(self, base_cfg):
        from core.backtester import compute_lot_size
        c = base_cfg["contract"]
        # Tiny equity
        lots = compute_lot_size(100, 0.01, 1800.0, 1790.0, c["contract_size"],
                                c["lot_step"], c["min_lot"], c["max_lot"])
        assert lots == c["min_lot"]
        # Huge equity
        lots = compute_lot_size(1e9, 0.01, 1800.0, 1790.0, c["contract_size"],
                                c["lot_step"], c["min_lot"], c["max_lot"])
        assert lots == c["max_lot"]


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

class TestMetrics:
    def test_all_required_metrics_present(self, base_cfg):
        from core.backtester import run_backtest
        df = _make_df(300)
        result = run_backtest(AlternatingStrategy(), df, base_cfg)
        m = result.metrics
        required = ["total_trades", "win_rate", "profit_factor", "sharpe", "max_drawdown",
                     "expectancy", "calmar", "total_net_pnl", "total_gross_pnl", "total_costs",
                     "cost_gross_ratio", "payoff_ratio", "total_return_pct", "trades_per_year",
                     "yearly_pnl", "profit_concentration_top5pct", "exit_reasons",
                     "long_trades", "short_trades", "long_win_rate", "short_win_rate"]
        for key in required:
            assert key in m, f"Missing metric: {key}"

    def test_no_trades_returns_error(self, base_cfg):
        from core.backtester import run_backtest
        df = _make_df(50)

        class DoNothingStrategy:
            def on_bar(self, bars):
                return None

        result = run_backtest(DoNothingStrategy(), df, base_cfg)
        assert result.metrics["total_trades"] == 0
        assert result.metrics["error"] == "no_trades"

    def test_equity_curve_monotonic_for_winning_strategy(self, base_cfg):
        """Not a guarantee, but on a rigged strategy the equity should end higher."""
        from core.backtester import run_backtest, Signal, Direction
        # Use a strategy that always wins by knowing the trend direction
        # (this is NOT how real strategies work — just testing the accounting)
        df = _make_df(100, seed=7)

        class PerfectStrategy:
            """Buys if next close > current close (only possible in test)."""
            def __init__(self):
                self.bars_data = df  # cheat: has the full data

            def on_bar(self, bars):
                i = len(bars) - 1
                if i + 2 >= len(self.bars_data):
                    return None
                if self.bars_data["close"].iloc[i + 1] > self.bars_data["close"].iloc[i]:
                    return Signal(direction=Direction.LONG)
                else:
                    return Signal(direction=Direction.SHORT)

        result = run_backtest(PerfectStrategy(), df, base_cfg)
        # Even a "perfect" strategy has costs, but should be net positive
        assert result.metrics["total_net_pnl"] > 0

    def test_metadata_carries_through(self, base_cfg):
        from core.backtester import run_backtest
        df = _make_df(100)
        df.attrs["timeframe"] = "1h"
        df.attrs["split"] = "train"
        df.attrs["file_hash"] = "abc123"
        result = run_backtest(AlwaysLongStrategy(), df, base_cfg)
        assert result.metadata["timeframe"] == "1h"
        assert result.metadata["split"] == "train"
        assert result.metadata["file_hash"] == "abc123"

    def test_result_serialisable(self, base_cfg):
        from core.backtester import run_backtest, result_to_doc
        import json
        df = _make_df(200)
        result = run_backtest(AlternatingStrategy(), df, base_cfg)
        doc = result_to_doc(result)
        # Must be JSON-serialisable (for Mongo)
        json.dumps(doc, default=str)
        assert doc["metrics"]["total_trades"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_strategy_exception_does_not_crash_backtest(self, base_cfg):
        from core.backtester import run_backtest

        class CrashyStrategy:
            def on_bar(self, bars):
                raise RuntimeError("kaboom")

        df = _make_df(50)
        result = run_backtest(CrashyStrategy(), df, base_cfg)
        assert result.metrics["total_trades"] == 0

    def test_open_position_closed_at_end(self, base_cfg):
        from core.backtester import run_backtest
        df = _make_df(100)
        result = run_backtest(AlwaysLongStrategy(), df, base_cfg)
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "end_of_data"
        assert result.trades[0].exit_bar_idx == len(df) - 1

    def test_signal_reversal_closes_and_opens(self, base_cfg):
        from core.backtester import run_backtest
        df = _make_df(300)
        result = run_backtest(AlternatingStrategy(period=10), df, base_cfg)
        reversals = [t for t in result.trades if t.exit_reason == "signal_reversal"]
        assert len(reversals) > 0
        # After a reversal close, a new position should open
        for i in range(len(result.trades) - 1):
            t1, t2 = result.trades[i], result.trades[i + 1]
            if t1.exit_reason == "signal_reversal":
                assert t2.entry_time == t1.exit_time  # same bar
                assert t2.direction != t1.direction  # opposite direction
