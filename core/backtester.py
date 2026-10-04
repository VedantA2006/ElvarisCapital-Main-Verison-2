"""
core/backtester.py – Production backtester with honest execution engine (Phase F2).

Orchestrates:
1. Data preparation and session/rollover classification
2. Signal generation & validation into SignalTape (closes BT-1, BT-14)
3. Strategy exception trapping without swallowing (closes BT-12)
4. Fast Numba execution (simulator.py) or pure-Python reference (event_simulator.py)
5. Comprehensive MTM portfolio analytics (analytics.py)
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

import numpy as np
import pandas as pd

from core.analytics import compute_analytics
from core.event_simulator import EventSimulator, SimulationResult, Trade
from core.signals import Action, Direction, Signal, SignalTape, SignalValidationError, validate_and_record_signal
from core.simulator import FastSimulator
from core.sizing import calculate_position_size, check_margin_call

_log = logging.getLogger("quantforge.backtester")


class OrderType(Enum):
    MARKET = "market"
    STOP = "stop"
    LIMIT = "limit"


@dataclass
class EquityPoint:
    timestamp: pd.Timestamp
    equity: float
    drawdown: float
    position: int  # 0, 1 (long), -1 (short)


class StrategyProtocol(Protocol):
    """What the backtester expects from a strategy object."""
    def on_bar(self, bars: pd.DataFrame) -> Signal | None:
        """Called after bar[-1] closes. May only use data in `bars`."""
        ...


class CostEngine:
    """Computes spread, slippage, commission, and swap per the config."""

    def __init__(self, cfg: dict):
        self._cost = cfg["costs"]
        self._contract = cfg["contract"]
        self._spread_cfg = self._cost.get("spread", {})
        self._rollover_tz = self._spread_cfg.get("rollover_tz", "America/New_York")
        self._rollover_start = int(self._spread_cfg.get("rollover_start_local", 16))
        self._rollover_end = int(self._spread_cfg.get("rollover_end_local", 19))
        self._swap = self._cost.get("swap", {})
        self._triple_day = self._swap.get("triple_day", "wednesday").lower()
        self._day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
        }

    def spread_points(self, timestamp: pd.Timestamp, session: str) -> float:
        """Return spread in USD per oz (fixes BT-4: no pip_value multiplication)."""
        cfg = self._spread_cfg
        local = timestamp.tz_convert(self._rollover_tz)
        local_hour = local.hour + local.minute / 60.0
        if self._rollover_start <= local_hour < self._rollover_end:
            return float(cfg.get("rollover", 0.50))
        if session == "asia":
            return float(cfg.get("asia", 0.35))
        return float(cfg.get("default", 0.25))

    def slippage_points(self, atr: float, is_stop: bool = False) -> float:
        """Slippage in price points, proportional to ATR."""
        cfg = self._cost.get("slippage", {})
        base = float(cfg.get("base_fraction_of_atr", 0.05)) * atr
        if is_stop:
            base *= float(cfg.get("stop_multiplier", 1.5))
        return base

    def commission_usd(self, lots: float) -> float:
        """Round-trip commission."""
        return float(self._cost.get("commission_per_lot", 7.0)) * lots

    def swap_usd(
        self,
        direction: Direction | str,
        lots: float,
        hold_nights: int,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
    ) -> float:
        """Swap cost for overnight holds at 17:00 NY rollover (trading days only)."""
        if hold_nights <= 0:
            return 0.0
        is_long = (direction == Direction.LONG or direction == "LONG" or direction == 1)
        per_night = (
            float(self._swap.get("long_per_lot_per_night", -3.50))
            if is_long else float(self._swap.get("short_per_lot_per_night", 0.80))
        )
        triple_idx = self._day_map.get(self._triple_day, 2)
        total_nights = 0
        current = entry_time.tz_convert(self._rollover_tz)
        exit_ny = exit_time.tz_convert(self._rollover_tz)

        # Count 17:00 NY rollovers on trading days (Mon-Fri)
        roll_cur = current.replace(hour=17, minute=0, second=0, microsecond=0)
        if current < roll_cur:
            first_roll = roll_cur
        else:
            first_roll = roll_cur + pd.Timedelta(days=1)

        while first_roll <= exit_ny:
            w = first_roll.weekday()
            if w in (0, 1, 2, 3, 4):  # Mon-Fri only
                total_nights += 3 if w == triple_idx else 1
            first_roll += pd.Timedelta(days=1)

        return per_night * lots * total_nights


def compute_lot_size(
    equity: float,
    risk_fraction: float,
    entry_price: float,
    stop_loss: float | None,
    contract_size: int,
    lot_step: float,
    min_lot: float,
    max_lot: float,
) -> float:
    """Legacy helper for backward compatibility."""
    sl_dist = abs(entry_price - stop_loss) if (stop_loss is not None and stop_loss != entry_price) else (entry_price * 0.01)
    res = calculate_position_size(
        equity=equity,
        sl_distance=sl_dist,
        price=entry_price,
        risk_fraction=risk_fraction,
        contract_size=float(contract_size),
        lot_step=lot_step,
        min_lot=min_lot,
        max_lot=max_lot,
    )
    return res.lots if not res.skipped else min_lot


@dataclass
class BacktestResult:
    """Complete output produced by a single backtest run."""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    wall_seconds: float = 0.0
    status: str = "completed"  # "completed" | "code_error"


def prepare_simulation_arrays(df: pd.DataFrame, cfg: dict) -> tuple[
    np.ndarray,  # opens
    np.ndarray,  # highs
    np.ndarray,  # lows
    np.ndarray,  # closes
    np.ndarray,  # timestamps
    np.ndarray,  # spreads
    np.ndarray,  # atrs
    np.ndarray,  # is_rollover
    np.ndarray,  # rollover_multiplier
]:
    """Extract and prepare aligned numeric arrays for simulation."""
    bars = df.reset_index(drop=True)
    n = len(bars)

    opens = np.ascontiguousarray(bars["open"].to_numpy(dtype=np.float64))
    highs = np.ascontiguousarray(bars["high"].to_numpy(dtype=np.float64))
    lows = np.ascontiguousarray(bars["low"].to_numpy(dtype=np.float64))
    closes = np.ascontiguousarray(bars["close"].to_numpy(dtype=np.float64))
    timestamps = np.ascontiguousarray(bars["timestamp"].to_numpy())

    # Prior bar ATR (shifted by 1)
    prev_c = np.roll(closes, 1)
    prev_c[0] = closes[0]
    tr = np.nanmax(np.vstack([highs - lows, np.abs(highs - prev_c), np.abs(lows - prev_c)]), axis=0)
    atr_series = pd.Series(tr).rolling(20, min_periods=1).mean().to_numpy()
    atrs = np.roll(atr_series, 1)
    atrs[0] = atr_series[0] if len(atr_series) > 0 else 1.0

    # Spreads per bar
    cost_cfg = cfg.get("costs", {})
    spread_cfg = cost_cfg.get("spread", {})
    default_spread = float(spread_cfg.get("default", 0.25))
    asia_spread = float(spread_cfg.get("asia", 0.35))
    rollover_spread = float(spread_cfg.get("rollover", 0.50))
    rollover_tz = spread_cfg.get("rollover_tz", "America/New_York")
    rollover_start = int(spread_cfg.get("rollover_start_local", 16))
    rollover_end = int(spread_cfg.get("rollover_end_local", 19))

    ts_dt = pd.to_datetime(bars["timestamp"])
    local_ny = ts_dt.dt.tz_convert(rollover_tz)
    hours_ny = local_ny.dt.hour + local_ny.dt.minute / 60.0
    sessions = bars["session"].to_numpy() if "session" in bars.columns else np.full(n, "default")

    spreads = np.full(n, default_spread, dtype=np.float64)
    spreads[sessions == "asia"] = asia_spread
    spreads[(hours_ny >= rollover_start) & (hours_ny < rollover_end)] = rollover_spread

    # Rollover swap classification (17:00 NY rollover on trading days)
    is_rollover = np.zeros(n, dtype=np.uint8)
    rollover_multiplier = np.zeros(n, dtype=np.float64)
    triple_day_name = cost_cfg.get("swap", {}).get("triple_day", "wednesday").lower()
    day_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4}
    triple_weekday = day_map.get(triple_day_name, 2)

    for i in range(n):
        cur_t = local_ny.iloc[i]
        next_t = local_ny.iloc[i + 1] if i + 1 < n else cur_t + pd.Timedelta(hours=1)
        roll_today = cur_t.replace(hour=17, minute=0, second=0, microsecond=0)
        if cur_t <= roll_today < next_t and roll_today.weekday() in (0, 1, 2, 3, 4):
            is_rollover[i] = 1
            rollover_multiplier[i] = 3.0 if roll_today.weekday() == triple_weekday else 1.0

    return opens, highs, lows, closes, timestamps, spreads, atrs, is_rollover, rollover_multiplier


def generate_signal_tape(
    strategy: Any,
    df: pd.DataFrame,
    cfg: dict,
    warmup_bars: int = 0,
    seed: int = 42,
) -> tuple[SignalTape, int, str | None]:
    """Generate and validate a SignalTape by running strategy over df.

    Handles string source code (via Sandbox), strategy instances, or callables.
    Returns (tape, strategy_errors, first_strategy_error).
    """
    bars = df.reset_index(drop=True)
    n_bars = len(bars)

    first_live = 0
    if "is_warmup" in bars.columns:
        w = np.where(~bars["is_warmup"].to_numpy())[0]
        if len(w) > 0:
            first_live = int(w[0])
    first_live = max(first_live, warmup_bars)

    opens, highs, lows, closes, timestamps, spreads, atrs, is_rollover, rollover_multiplier = (
        prepare_simulation_arrays(bars, cfg)
    )

    tape = SignalTape.empty(n_bars)
    strategy_errors = 0
    first_strategy_error: str | None = None

    strat_obj = None
    if isinstance(strategy, str):
        from core.sandbox import Sandbox
        strat_obj = Sandbox(cfg).load_strategy(strategy, seed=seed)
    elif isinstance(strategy, type):
        strat_obj = strategy()
    elif callable(strategy) and not hasattr(strategy, "on_bar"):
        strat_obj = strategy()
    else:
        strat_obj = strategy

    try:
        for i in range(first_live, n_bars):
            try:
                sig = strat_obj.on_bar(bars.iloc[: i + 1])
            except Exception as exc:
                # BT-12: Strategy exceptions are NOT swallowed!
                strategy_errors += 1
                tb_str = traceback.format_exc()
                first_strategy_error = tb_str
                _log.error("strategy.on_bar raised at bar %d: %s: %s\n%s", i, type(exc).__name__, exc, tb_str)
                return tape, strategy_errors, first_strategy_error

            if sig is None:
                continue

            # Handle legacy Signal instances with absolute stop_loss / take_profit or missing SL
            if isinstance(sig, Signal):
                modern_sig = sig
                if np.isnan(sig.sl_distance):
                    try:
                        modern_sig = Signal.from_prices(
                            close=closes[i],
                            direction=sig.direction,
                            stop_loss=sig.stop_loss,
                            take_profit=sig.take_profit,
                            tag=sig.tag,
                        )
                    except SignalValidationError as ve:
                        # Inverted stop or invalid price level: BT-1 caught!
                        strategy_errors += 1
                        tb_str = traceback.format_exc()
                        first_strategy_error = tb_str
                        _log.error("Signal validation failed at bar %d: %s", i, ve)
                        return tape, strategy_errors, first_strategy_error

                validate_and_record_signal(
                    signal=modern_sig,
                    bar_idx=i,
                    tape=tape,
                    atr_prev=atrs[i],
                    current_spread=spreads[i],
                    cfg=cfg,
                )

        # Validate signal tape
        try:
            tape.validate_or_raise()
        except SignalValidationError as ve:
            strategy_errors += 1
            tb_str = traceback.format_exc()
            first_strategy_error = tb_str
            return tape, strategy_errors, first_strategy_error

    finally:
        if hasattr(strat_obj, "close"):
            try:
                strat_obj.close()
            except Exception as ex:
                _log.debug("Strategy close error: %s", ex)

    return tape, strategy_errors, first_strategy_error


def run_simulation_on_tape(
    tape: SignalTape,
    df: pd.DataFrame,
    cfg: dict,
    use_fast: bool = True,
    warmup_bars: int = 0,
    strategy_errors: int = 0,
    first_strategy_error: str | None = None,
) -> BacktestResult:
    """Execute simulation on a pre-computed SignalTape and compute MTM analytics."""
    t0 = time.perf_counter()
    bars = df.reset_index(drop=True)
    n_bars = len(bars)

    first_live = 0
    if "is_warmup" in bars.columns:
        w = np.where(~bars["is_warmup"].to_numpy())[0]
        if len(w) > 0:
            first_live = int(w[0])
    first_live = max(first_live, warmup_bars)

    opens, highs, lows, closes, timestamps, spreads, atrs, is_rollover, rollover_multiplier = (
        prepare_simulation_arrays(bars, cfg)
    )

    # Step 2: Execute simulation
    sim = FastSimulator(cfg) if use_fast else EventSimulator(cfg)
    sim_res: SimulationResult = sim.run(
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        timestamps=timestamps,
        spreads=spreads,
        atrs=atrs,
        is_rollover=is_rollover,
        rollover_multiplier=rollover_multiplier,
        tape=tape,
    )

    # Step 3: Compute analytics
    initial_equity = float(cfg.get("sizing", {}).get("initial_equity", 100000.0))
    metrics = compute_analytics(
        trades=sim_res.trades,
        equity_mtm=sim_res.equity_mtm,
        positions=sim_res.positions,
        timestamps=timestamps,
        closes=closes,
        opens=opens,
        initial_equity=initial_equity,
        cfg=cfg,
    )

    # Build equity_curve
    equity_curve: list[EquityPoint] = [
        EquityPoint(
            timestamp=pd.Timestamp(timestamps[i]),
            equity=round(float(sim_res.equity_mtm[i]), 2),
            drawdown=round(float(sim_res.drawdown_mtm[i]), 6),
            position=int(sim_res.positions[i]),
        )
        for i in range(n_bars)
    ]

    wall_seconds = time.perf_counter() - t0
    return BacktestResult(
        trades=sim_res.trades,
        equity_curve=equity_curve,
        metrics=metrics,
        metadata={
            "timeframe": df.attrs.get("timeframe", ""),
            "split": df.attrs.get("split", ""),
            "file_hash": df.attrs.get("file_hash", ""),
            "slice_hash": df.attrs.get("slice_hash", ""),
            "total_bars": n_bars,
            "live_bars": n_bars - first_live,
            "warmup_bars": first_live,
            "skipped_min_lot": sim_res.skipped_min_lot,
            "skipped_margin": sim_res.skipped_margin,
            "forced_closes": sim_res.forced_closes,
            "strategy_errors": strategy_errors,
            "first_strategy_error": first_strategy_error,
        },
        wall_seconds=round(wall_seconds, 3),
        status="completed",
    )


def run_backtest(
    strategy: StrategyProtocol | str | Any,
    df: pd.DataFrame,
    cfg: dict,
    warmup_bars: int = 0,
    use_fast: bool = True,
) -> BacktestResult:
    """Run full backtest with honest execution and MTM accounting."""
    t0 = time.perf_counter()
    tape, strategy_errors, first_strategy_error = generate_signal_tape(
        strategy=strategy,
        df=df,
        cfg=cfg,
        warmup_bars=warmup_bars,
    )

    if strategy_errors > 0 or first_strategy_error:
        wall_seconds = time.perf_counter() - t0
        return BacktestResult(
            trades=[],
            equity_curve=[],
            metrics={"total_trades": 0, "error": "code_error", "exception": first_strategy_error},
            metadata={
                "timeframe": df.attrs.get("timeframe", ""),
                "split": df.attrs.get("split", ""),
                "file_hash": df.attrs.get("file_hash", ""),
                "slice_hash": df.attrs.get("slice_hash", ""),
                "total_bars": len(df),
                "strategy_errors": strategy_errors,
                "first_strategy_error": first_strategy_error,
            },
            wall_seconds=round(wall_seconds, 3),
            status="code_error",
        )

    res = run_simulation_on_tape(
        tape=tape,
        df=df,
        cfg=cfg,
        use_fast=use_fast,
        warmup_bars=warmup_bars,
        strategy_errors=strategy_errors,
        first_strategy_error=first_strategy_error,
    )
    res.wall_seconds = round(time.perf_counter() - t0, 3)
    return res


def compute_metrics(
    trades: list[Trade],
    equity_curve: list[EquityPoint],
    initial_equity: float,
    cfg: dict,
) -> dict[str, Any]:
    """Backward compatibility wrapper for metric calculations."""
    if not trades:
        return {"total_trades": 0, "error": "no_trades"}
    eq_mtm = np.array([e.equity for e in equity_curve])
    pos_arr = np.array([e.position for e in equity_curve])
    ts_arr = np.array([e.timestamp for e in equity_curve])
    closes = eq_mtm  # fallback if raw closes not passed
    opens = eq_mtm
    return compute_analytics(trades, eq_mtm, pos_arr, ts_arr, closes, opens, initial_equity, cfg)


def result_to_doc(result: BacktestResult) -> dict[str, Any]:
    """Serialize a BacktestResult for MongoDB storage with stringified keys."""
    def _clean(val: Any) -> Any:
        if isinstance(val, dict):
            return {str(k): _clean(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_clean(x) for x in val]
        if isinstance(val, (np.floating, float)):
            return float(val) if math.isfinite(val) else None
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, pd.Timestamp):
            return val.isoformat()
        return val

    return {
        "trades": [_clean(asdict(t)) for t in result.trades],
        "equity_curve_summary": {
            "start": result.equity_curve[0].equity if result.equity_curve else None,
            "end": result.equity_curve[-1].equity if result.equity_curve else None,
            "points": len(result.equity_curve),
        },
        "metrics": _clean(result.metrics),
        "metadata": _clean(result.metadata),
        "status": result.status,
        "wall_seconds": result.wall_seconds,
        "created_at": datetime.now(timezone.utc),
    }
