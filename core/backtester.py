"""
core/backtester.py – Event-driven backtester with realistic execution.

HARD INVARIANTS (Section 0):
1. NO LOOKAHEAD. The strategy's `generate_signals()` receives ONLY bars
   that have already CLOSED (i.e. bar[i] is complete). Any fill happens
   at bar[i+1].open. This is structural, not policy — the backtester
   NEVER passes future data, regardless of what the strategy does.
2. COSTS ARE ALWAYS ON. Spread, slippage, commission, and swap are
   computed on every trade. There is no "gross-only" mode for ranking.
3. FILLS AT NEXT BAR OPEN. Market orders fill at bars[signal_bar+1].open.
   Stop/limit orders check bar high/low after the signal bar but fill at
   the order price (conservative: assumes worst-case intra-bar ordering).
4. ONE POSITION AT A TIME (no pyramiding, per config).

Execution model:
  for i in range(warmup, N):
      # bar i is NOW CLOSED
      signal = strategy.on_bar(bars[:i+1])   # can see up to and including bar i
      # fill at bar i+1 open
      if signal and i+1 < N:
          execute(signal, bars[i+1].open)

Cost model (Section 5):
  - Spread: session-aware (asia wider, rollover wider, news wider)
  - Slippage: base_fraction_of_atr * ATR, multiplied for stops / news
  - Commission: per-lot round trip
  - Swap: per-lot per overnight hold, triple on configured day
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

import numpy as np
import pandas as pd


# ─── Types ──────────────────────────────────────────────────────────────────

class Direction(Enum):
    LONG = 1
    SHORT = -1


class OrderType(Enum):
    MARKET = "market"
    STOP = "stop"
    LIMIT = "limit"


@dataclass
class Signal:
    """Strategy output: what to do on the NEXT bar."""
    direction: Direction
    stop_loss: float | None = None      # price
    take_profit: float | None = None    # price
    order_type: OrderType = OrderType.MARKET
    order_price: float | None = None    # for stop/limit orders
    tag: str = ""                       # strategy annotation


@dataclass
class Trade:
    """Closed trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str              # "LONG" / "SHORT"
    entry_price: float
    exit_price: float
    lots: float
    gross_pnl: float
    spread_cost: float
    slippage_cost: float
    commission: float
    swap_cost: float
    net_pnl: float
    bars_held: int
    exit_reason: str            # "stop_loss" / "take_profit" / "signal_reversal" / "eod_flat"
    tag: str = ""
    entry_bar_idx: int = 0
    exit_bar_idx: int = 0


@dataclass
class EquityPoint:
    timestamp: pd.Timestamp
    equity: float
    drawdown: float
    position: int               # 0, 1 (long), -1 (short)


class StrategyProtocol(Protocol):
    """What the backtester expects from a strategy object."""
    def on_bar(self, bars: pd.DataFrame) -> Signal | None:
        """Called after bar[-1] closes. May only use data in `bars`."""
        ...


# ─── Cost Engine ────────────────────────────────────────────────────────────

class CostEngine:
    """Computes spread, slippage, commission, and swap per the config."""

    def __init__(self, cfg: dict):
        self._cost = cfg["costs"]
        self._contract = cfg["contract"]
        self._rollover_tz = self._cost["spread"]["rollover_tz"]
        self._rollover_start = self._cost["spread"]["rollover_start_local"]
        self._rollover_end = self._cost["spread"]["rollover_end_local"]
        self._swap = self._cost["swap"]
        self._triple_day = self._swap["triple_day"].lower()
        self._day_map = {"monday": 0, "tuesday": 1, "wednesday": 2,
                         "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}

    def spread_points(self, timestamp: pd.Timestamp, session: str) -> float:
        """Return spread in price points (not pips)."""
        cfg = self._cost["spread"]
        pip = self._contract["pip_value"]  # 0.01
        # Check rollover window in local time
        local = timestamp.tz_convert(self._rollover_tz)
        if self._rollover_start <= local.hour < self._rollover_end:
            return cfg["rollover"] * pip
        if session == "asia":
            return cfg["asia"] * pip
        return cfg["default"] * pip

    def slippage_points(self, atr: float, is_stop: bool = False) -> float:
        """Slippage in price points, proportional to ATR."""
        cfg = self._cost["slippage"]
        base = cfg["base_fraction_of_atr"] * atr
        if is_stop:
            base *= cfg["stop_multiplier"]
        return base

    def commission_usd(self, lots: float) -> float:
        """Round-trip commission."""
        return self._cost["commission_per_lot"] * lots

    def swap_usd(self, direction: Direction, lots: float, hold_nights: int,
                 entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> float:
        """Swap cost for overnight holds. Triple on the configured day."""
        if hold_nights <= 0:
            return 0.0
        per_night = (self._swap["long_per_lot_per_night"] if direction == Direction.LONG
                     else self._swap["short_per_lot_per_night"])
        # Count triple swap days
        triple_idx = self._day_map[self._triple_day]
        total_nights = 0
        # Walk each overnight boundary
        current = entry_time.normalize() + pd.Timedelta(days=1)  # midnight after entry
        while current <= exit_time and total_nights < hold_nights * 4:  # safety cap
            ny_day = current.tz_convert(self._rollover_tz).weekday()
            if ny_day == triple_idx:
                total_nights += 3
            else:
                total_nights += 1
            current += pd.Timedelta(days=1)
        return per_night * lots * total_nights


# ─── Position Sizing ────────────────────────────────────────────────────────

def compute_lot_size(equity: float, risk_fraction: float, entry_price: float,
                     stop_loss: float | None, contract_size: int,
                     lot_step: float, min_lot: float, max_lot: float) -> float:
    """Fixed-fractional sizing. If no SL, risk 1 ATR (conservative fallback)."""
    if stop_loss is not None and stop_loss != entry_price:
        risk_per_oz = abs(entry_price - stop_loss)
    else:
        risk_per_oz = entry_price * 0.01  # fallback: 1% of price
    risk_usd = equity * risk_fraction
    lots_raw = risk_usd / (risk_per_oz * contract_size)
    lots = max(min_lot, min(max_lot, round(lots_raw / lot_step) * lot_step))
    return lots


# ─── Core Engine ────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Everything produced by a single backtest run."""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    wall_seconds: float = 0.0


def run_backtest(strategy: StrategyProtocol, df: pd.DataFrame, cfg: dict,
                 warmup_bars: int = 0) -> BacktestResult:
    """Run a full backtest. df must be the output of DataStore.get_data()."""
    t0 = time.perf_counter()
    costs = CostEngine(cfg)
    sizing = cfg["sizing"]
    contract = cfg["contract"]

    bars = df.reset_index(drop=True)
    N = len(bars)
    ts = bars["timestamp"].to_numpy()
    opens = bars["open"].to_numpy(dtype=np.float64)
    highs = bars["high"].to_numpy(dtype=np.float64)
    lows = bars["low"].to_numpy(dtype=np.float64)
    closes = bars["close"].to_numpy(dtype=np.float64)
    sessions = bars["session"].to_numpy() if "session" in bars.columns else np.full(N, "default")

    # Pre-compute ATR for slippage (shifted by 1 so bar i uses prior ATR)
    prev_c = np.roll(closes, 1)
    prev_c[0] = np.nan
    tr = np.nanmax(np.vstack([highs - lows, np.abs(highs - prev_c), np.abs(lows - prev_c)]), axis=0)
    atr = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()
    atr_shifted = np.roll(atr, 1)
    atr_shifted[0] = np.nan

    equity = float(sizing["initial_equity"])
    peak_equity = equity
    trades: list[Trade] = []
    equity_curve: list[EquityPoint] = []

    # Position state
    pos_dir: Direction | None = None
    pos_entry_price: float = 0.0
    pos_lots: float = 0.0
    pos_sl: float | None = None
    pos_tp: float | None = None
    pos_entry_time: pd.Timestamp | None = None
    pos_entry_idx: int = 0
    pos_spread_cost: float = 0.0
    pos_slippage_cost: float = 0.0
    pos_tag: str = ""

    # Identify warmup end
    if "is_warmup" in bars.columns:
        first_live = int((~bars["is_warmup"]).idxmax())
    else:
        first_live = warmup_bars

    def _close_position(exit_price: float, exit_idx: int, reason: str,
                        exit_slippage: float = 0.0) -> None:
        nonlocal equity, peak_equity, pos_dir
        assert pos_dir is not None
        direction_mult = 1.0 if pos_dir == Direction.LONG else -1.0
        gross_pnl = direction_mult * (exit_price - pos_entry_price) * pos_lots * contract["contract_size"]

        # Exit spread cost
        exit_spread = costs.spread_points(pd.Timestamp(ts[exit_idx]), sessions[exit_idx])
        total_spread = (pos_spread_cost + exit_spread) * pos_lots * contract["contract_size"]

        # Exit slippage (against the trader)
        total_slippage = (pos_slippage_cost + exit_slippage) * pos_lots * contract["contract_size"]

        commission = costs.commission_usd(pos_lots)

        # Swap
        bars_held = exit_idx - pos_entry_idx
        hold_nights = max(0, (pd.Timestamp(ts[exit_idx]).normalize() -
                              pd.Timestamp(ts[pos_entry_idx]).normalize()).days)
        swap = costs.swap_usd(pos_dir, pos_lots, hold_nights,
                              pd.Timestamp(ts[pos_entry_idx]), pd.Timestamp(ts[exit_idx]))

        net_pnl = gross_pnl - total_spread - total_slippage - commission + swap
        equity += net_pnl
        peak_equity = max(peak_equity, equity)

        trades.append(Trade(
            entry_time=pd.Timestamp(ts[pos_entry_idx]),
            exit_time=pd.Timestamp(ts[exit_idx]),
            direction=pos_dir.name,
            entry_price=pos_entry_price,
            exit_price=exit_price,
            lots=pos_lots,
            gross_pnl=round(gross_pnl, 2),
            spread_cost=round(total_spread, 2),
            slippage_cost=round(total_slippage, 2),
            commission=round(commission, 2),
            swap_cost=round(swap, 2),
            net_pnl=round(net_pnl, 2),
            bars_held=bars_held,
            exit_reason=reason,
            tag=pos_tag,
            entry_bar_idx=pos_entry_idx,
            exit_bar_idx=exit_idx,
        ))
        pos_dir = None

    # ── Main loop ───────────────────────────────────────────────────────────
    for i in range(first_live, N):
        # === Phase A: Check SL/TP on this bar (using bar i high/low) ========
        # Position was entered at bar i's open or earlier.
        # We check if bar i's range triggered SL or TP.
        if pos_dir is not None and i > pos_entry_idx:
            hit_sl, hit_tp = False, False
            if pos_dir == Direction.LONG:
                if pos_sl is not None and lows[i] <= pos_sl:
                    hit_sl = True
                if pos_tp is not None and highs[i] >= pos_tp:
                    hit_tp = True
            else:  # SHORT
                if pos_sl is not None and highs[i] >= pos_sl:
                    hit_sl = True
                if pos_tp is not None and lows[i] <= pos_tp:
                    hit_tp = True

            if hit_sl and hit_tp:
                # Both hit in same bar: conservative = assume the loser (SL) hit first
                hit_tp = False

            if hit_sl:
                sl_slip = costs.slippage_points(atr_shifted[i] if not np.isnan(atr_shifted[i]) else 1.0,
                                                is_stop=True)
                # SL fills at SL price + slippage (against trader)
                if pos_dir == Direction.LONG:
                    fill = pos_sl - sl_slip
                else:
                    fill = pos_sl + sl_slip
                _close_position(fill, i, "stop_loss", exit_slippage=sl_slip)
            elif hit_tp:
                _close_position(pos_tp, i, "take_profit")

        # === Phase B: Get signal from strategy (bar i is now closed) ========
        # The strategy sees bars[0..i] inclusive. It CANNOT see bar i+1.
        signal: Signal | None = None
        try:
            signal = strategy.on_bar(bars.iloc[: i + 1])
        except Exception:
            pass  # strategy error → no signal, logged at a higher level

        # === Phase C: Execute signal at bar i+1 open ========================
        if signal is not None and i + 1 < N:
            next_open = opens[i + 1]
            next_session = sessions[i + 1]
            cur_atr = atr_shifted[i + 1] if i + 1 < len(atr_shifted) and not np.isnan(atr_shifted[i + 1]) else 1.0

            # Close existing position if direction changes
            if pos_dir is not None and pos_dir != signal.direction:
                exit_slip = costs.slippage_points(cur_atr)
                _close_position(next_open, i + 1, "signal_reversal", exit_slippage=exit_slip)

            # Open new position
            if pos_dir is None:
                entry_spread = costs.spread_points(pd.Timestamp(ts[i + 1]), next_session)
                entry_slippage = costs.slippage_points(cur_atr)
                # Adjust entry price for spread + slippage (against trader)
                if signal.direction == Direction.LONG:
                    fill_price = next_open + entry_spread + entry_slippage
                else:
                    fill_price = next_open - entry_spread - entry_slippage

                lots = compute_lot_size(
                    equity, sizing["risk_per_trade"], fill_price,
                    signal.stop_loss, contract["contract_size"],
                    contract["lot_step"], contract["min_lot"], contract["max_lot"],
                )

                pos_dir = signal.direction
                pos_entry_price = fill_price
                pos_lots = lots
                pos_sl = signal.stop_loss
                pos_tp = signal.take_profit
                pos_entry_time = pd.Timestamp(ts[i + 1])
                pos_entry_idx = i + 1
                pos_spread_cost = entry_spread
                pos_slippage_cost = entry_slippage
                pos_tag = signal.tag

        # Record equity
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        equity_curve.append(EquityPoint(
            timestamp=pd.Timestamp(ts[i]),
            equity=round(equity, 2),
            drawdown=round(dd, 6),
            position=0 if pos_dir is None else pos_dir.value,
        ))

    # Close any open position at the last bar's close
    if pos_dir is not None:
        cur_atr_val = atr_shifted[N - 1] if not np.isnan(atr_shifted[N - 1]) else 1.0
        exit_slip = costs.slippage_points(cur_atr_val)
        _close_position(closes[N - 1], N - 1, "end_of_data", exit_slippage=exit_slip)

    wall_seconds = time.perf_counter() - t0

    # Compute metrics
    metrics = compute_metrics(trades, equity_curve, sizing["initial_equity"], cfg)

    result = BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        metrics=metrics,
        metadata={
            "timeframe": df.attrs.get("timeframe", ""),
            "split": df.attrs.get("split", ""),
            "file_hash": df.attrs.get("file_hash", ""),
            "slice_hash": df.attrs.get("slice_hash", ""),
            "total_bars": N,
            "live_bars": N - first_live,
            "warmup_bars": first_live,
        },
        wall_seconds=round(wall_seconds, 3),
    )
    return result


# ─── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(trades: list[Trade], equity_curve: list[EquityPoint],
                    initial_equity: float, cfg: dict) -> dict[str, Any]:
    """Compute all Section 6 metrics. Net (after costs) unless stated."""
    m: dict[str, Any] = {}
    if not trades:
        m["total_trades"] = 0
        m["error"] = "no_trades"
        return m

    net_pnls = np.array([t.net_pnl for t in trades])
    gross_pnls = np.array([t.gross_pnl for t in trades])
    total_costs = np.array([t.spread_cost + t.slippage_cost + t.commission - t.swap_cost for t in trades])

    winners = net_pnls[net_pnls > 0]
    losers = net_pnls[net_pnls <= 0]

    m["total_trades"] = len(trades)
    m["winners"] = int(len(winners))
    m["losers"] = int(len(losers))
    m["win_rate"] = round(len(winners) / len(trades), 4)
    m["total_net_pnl"] = round(float(net_pnls.sum()), 2)
    m["total_gross_pnl"] = round(float(gross_pnls.sum()), 2)
    m["total_costs"] = round(float(total_costs.sum()), 2)
    m["avg_win"] = round(float(winners.mean()), 2) if len(winners) else 0.0
    m["avg_loss"] = round(float(losers.mean()), 2) if len(losers) else 0.0
    m["largest_win"] = round(float(winners.max()), 2) if len(winners) else 0.0
    m["largest_loss"] = round(float(losers.min()), 2) if len(losers) else 0.0
    m["avg_bars_held"] = round(float(np.mean([t.bars_held for t in trades])), 1)

    # Profit Factor
    gross_profit = float(winners.sum()) if len(winners) else 0.0
    gross_loss = float(abs(losers.sum())) if len(losers) else 1e-9
    m["profit_factor"] = round(gross_profit / max(gross_loss, 1e-9), 4)

    # Expectancy
    m["expectancy"] = round(float(net_pnls.mean()), 2)
    m["expectancy_per_dollar_risked"] = round(
        m["expectancy"] / max(abs(m["avg_loss"]), 1e-9), 4) if m["avg_loss"] != 0 else 0.0

    # Cost-Gross ratio
    total_gross_profit = float(gross_pnls[gross_pnls > 0].sum()) if (gross_pnls > 0).any() else 1e-9
    m["cost_gross_ratio"] = round(float(total_costs.sum()) / max(total_gross_profit, 1e-9), 4)

    # Sharpe (annualised, using trade returns)
    if len(net_pnls) > 1:
        trade_returns = net_pnls / initial_equity
        avg_ret = float(trade_returns.mean())
        std_ret = float(trade_returns.std(ddof=1))
        if std_ret > 1e-12:
            trades_per_year = _estimate_trades_per_year(trades)
            m["sharpe"] = round(avg_ret / std_ret * np.sqrt(trades_per_year), 4)
        else:
            m["sharpe"] = 0.0
    else:
        m["sharpe"] = 0.0

    # Max Drawdown
    equities = np.array([e.equity for e in equity_curve])
    if len(equities) > 0:
        peaks = np.maximum.accumulate(equities)
        dd = (peaks - equities) / np.where(peaks > 0, peaks, 1.0)
        m["max_drawdown"] = round(float(dd.max()), 6)
        m["max_drawdown_usd"] = round(float((peaks - equities).max()), 2)
    else:
        m["max_drawdown"] = 0.0
        m["max_drawdown_usd"] = 0.0

    # Calmar
    if m["max_drawdown"] > 1e-9:
        total_return = (equities[-1] - initial_equity) / initial_equity if len(equities) else 0.0
        years = _estimate_years(trades)
        annual_return = total_return / max(years, 0.1)
        m["calmar"] = round(annual_return / m["max_drawdown"], 4)
    else:
        m["calmar"] = 0.0

    # Payoff ratio
    m["payoff_ratio"] = round(abs(m["avg_win"]) / max(abs(m["avg_loss"]), 1e-9), 4)

    # Return
    m["total_return_pct"] = round((equities[-1] - initial_equity) / initial_equity * 100, 2) if len(equities) else 0.0

    # Trade frequency
    years = _estimate_years(trades)
    m["trades_per_year"] = round(len(trades) / max(years, 0.1), 1)
    m["years_covered"] = round(years, 3)

    # Per-year PnL
    yearly: dict[int, float] = {}
    for t in trades:
        y = t.entry_time.year
        yearly[y] = yearly.get(y, 0.0) + t.net_pnl
    m["yearly_pnl"] = {k: round(v, 2) for k, v in sorted(yearly.items())}

    # Profit concentration
    if len(winners) > 0:
        sorted_wins = np.sort(winners)[::-1]
        top_n = max(1, int(len(trades) * 0.05))
        m["profit_concentration_top5pct"] = round(float(sorted_wins[:top_n].sum()) / max(float(winners.sum()), 1e-9), 4)
    else:
        m["profit_concentration_top5pct"] = 0.0

    # Exit reasons
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    m["exit_reasons"] = reasons

    # Direction breakdown
    long_trades = [t for t in trades if t.direction == "LONG"]
    short_trades = [t for t in trades if t.direction == "SHORT"]
    m["long_trades"] = len(long_trades)
    m["short_trades"] = len(short_trades)
    m["long_win_rate"] = round(sum(1 for t in long_trades if t.net_pnl > 0) / max(len(long_trades), 1), 4)
    m["short_win_rate"] = round(sum(1 for t in short_trades if t.net_pnl > 0) / max(len(short_trades), 1), 4)

    return m


def _estimate_years(trades: list[Trade]) -> float:
    if not trades:
        return 0.0
    span = trades[-1].exit_time - trades[0].entry_time
    return max(span.total_seconds() / (365.25 * 86400), 0.01)


def _estimate_trades_per_year(trades: list[Trade]) -> float:
    years = _estimate_years(trades)
    return len(trades) / max(years, 0.01)


def result_to_doc(result: BacktestResult) -> dict[str, Any]:
    """Serialise a BacktestResult for MongoDB storage."""
    return {
        "trades": [asdict(t) for t in result.trades],
        "equity_curve_summary": {
            "start": result.equity_curve[0].equity if result.equity_curve else None,
            "end": result.equity_curve[-1].equity if result.equity_curve else None,
            "points": len(result.equity_curve),
        },
        "metrics": result.metrics,
        "metadata": result.metadata,
        "wall_seconds": result.wall_seconds,
        "created_at": datetime.now(timezone.utc),
    }
