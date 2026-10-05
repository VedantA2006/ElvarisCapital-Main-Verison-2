"""
forward/paper_trader.py – Forward paper trading execution engine for holdout_passed strategies.

Closes: OPS-2 (paper trader part)
Features:
- Consumes closed bars from pluggable DataFeed instances.
- Executes strategy signals with full realistic execution costs:
  spread, slippage (ATR fraction), commission, and swap.
- Persists trade-by-trade records in `forward_trades` collection.
- Aggregates rolling forward metrics (calendar days, total trades, Sharpe, PF, drawdown).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from core.event_simulator import EventSimulator
from forward.feeds import DataFeed


class PaperTrader:
    """Manages forward execution and paper trade tracking for validated strategies."""

    def __init__(self, cfg: dict, db: Any, feed: DataFeed):
        self.cfg = cfg
        self.db = db
        self.feed = feed

    def run_session(
        self,
        strategy_id: str,
        max_bars: int = 1000,
    ) -> dict[str, Any]:
        """Process incoming bars from feed and update strategy forward execution."""
        doc = self.db["candidates"].find_one({"strategy_id": strategy_id})
        if not doc:
            raise ValueError(f"Strategy '{strategy_id}' not found.")

        # Collect bars from feed
        bars: list[dict[str, Any]] = []
        count = 0
        while self.feed.has_next() and count < max_bars:
            bars.append(self.feed.next_bar())
            count += 1

        if not bars:
            return {"status": "no_bars_processed", "total_bars_processed": 0}

        df_bars = pd.DataFrame(bars)
        if "timestamp" in df_bars.columns:
            df_bars["timestamp"] = pd.to_datetime(df_bars["timestamp"])

        # Execute signals
        code = doc.get("source_code", "")
        from core.backtester import run_backtest, run_simulation_on_tape
        from core.signals import Action, Direction, SignalTape

        trades: list[Any] = []
        m: dict[str, Any] = {}

        import logging
        _log = logging.getLogger("quantforge.forward")

        if not isinstance(code, str) or ("class Strategy" in code and "def on_bar" in code):
            try:
                bt_res = run_backtest(code, df_bars, self.cfg, use_fast=False)
                trades = bt_res.trades
                m = bt_res.metrics
            except Exception as exc:
                _log.warning("PaperTrader run_backtest error: %s", exc)
        else:
            loc: dict[str, Any] = {}
            try:
                exec(code, loc)
                gen = loc.get("generate_signals")
                sigs = gen(df_bars) if callable(gen) else pd.Series(0, index=df_bars.index)
            except Exception as exc:
                _log.warning("PaperTrader signal generation error: %s", exc)
                sigs = pd.Series(0, index=df_bars.index)

            n = len(df_bars)
            tape = SignalTape.empty(n)
            for i, s in enumerate(sigs):
                if s > 0:
                    tape.actions[i] = Action.ENTER_LONG
                    tape.sl_distances[i] = 10.0
                    tape.tp_distances[i] = 20.0
                elif s < 0:
                    tape.actions[i] = Action.ENTER_SHORT
                    tape.sl_distances[i] = 10.0
                    tape.tp_distances[i] = 20.0

            try:
                bt_res = run_simulation_on_tape(tape, df_bars, self.cfg, use_fast=False)
                trades = bt_res.trades
                m = bt_res.metrics
            except Exception as exc:
                _log.warning("PaperTrader simulation on tape error: %s", exc)

        # Record trades in forward_trades collection
        if trades:
            trade_docs = []
            for t in trades:
                td = {
                    "strategy_id": strategy_id,
                    "trade_id": getattr(t, "trade_id", 0),
                    "entry_time": str(getattr(t, "entry_time", "")),
                    "exit_time": str(getattr(t, "exit_time", "")),
                    "direction": getattr(t, "direction", "LONG"),
                    "net_pnl": getattr(t, "net_pnl", 0.0),
                    "spread_cost": getattr(t, "spread_cost", 0.0),
                    "slippage_cost": getattr(t, "slippage_cost", 0.0),
                    "commission": getattr(t, "commission", 0.0),
                    "recorded_at": time.time(),
                }
                trade_docs.append(td)
            self.db["forward_trades"].insert_many(trade_docs)

        # Calculate calendar days span
        first_ts = df_bars["timestamp"].min() if "timestamp" in df_bars.columns else None
        last_ts = df_bars["timestamp"].max() if "timestamp" in df_bars.columns else None
        cal_days = 0
        if first_ts is not None and last_ts is not None:
            cal_days = max(1, (last_ts - first_ts).days)

        existing_fwd = doc.get("forward_metrics", {})
        accum_trades = existing_fwd.get("total_trades", 0) + len(trades)
        accum_days = max(existing_fwd.get("calendar_days", 0), cal_days)

        fwd_metrics = {
            "calendar_days": accum_days,
            "total_trades": accum_trades,
            "sharpe": float(m.get("sharpe", 0.0)),
            "profit_factor": float(m.get("profit_factor", 1.0)),
            "net_profit": float(m.get("net_profit", 0.0)),
            "max_drawdown": float(m.get("max_drawdown", 0.0)),
            "is_degraded": existing_fwd.get("is_degraded", False),
            "last_updated": time.time(),
        }

        self.db["candidates"].update_one(
            {"strategy_id": strategy_id},
            {"$set": {"forward_metrics": fwd_metrics}},
        )

        return {
            "status": "ok",
            "total_bars_processed": len(bars),
            "trades_generated": len(trades),
            "forward_metrics": fwd_metrics,
        }
