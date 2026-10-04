"""
core/event_simulator.py – Independent, pure-Python bar-by-bar execution simulator (Phase F2).

Written without importing simulator.py to serve as the reference standard
for differential verification.

Implements all Section F2.3 execution invariants:
- Bid price basis with exact spread and slippage
- Live entry bar (SL/TP checked on entry bar)
- Same-bar SL/TP: SL hit first
- Gaps: stops fill at open + slippage; limits (TP) fill at TP price
- Trailing / breakeven / time stop evaluated at close, active from next bar
- 17:00 NY rollover swap (trading days only, triple on Wednesday)
- Full cost reconciliation: gross - spread - slippage - commission - swap == net
- Mark-to-market equity per bar
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.signals import Action, Direction, SignalTape
from core.sizing import calculate_position_size, check_margin_call


@dataclass
class Trade:
    """Complete record of a closed trade."""
    trade_id: int
    entry_bar_idx: int
    exit_bar_idx: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str                     # "LONG" or "SHORT"
    lots: float
    entry_price: float                 # actual fill price (ask for long, bid for short)
    exit_price: float                  # actual fill price (bid for long, ask for short)
    gross_pnl: float                   # zero-cost path
    spread_cost: float
    slippage_cost: float
    commission: float
    swap_cost: float                   # positive = cost (-accumulated_swap)
    net_pnl: float                     # gross_pnl - spread - slippage - commission - swap
    bars_held: int
    exit_reason: str                   # stop_loss, take_profit, signal_close, signal_reversal, time_stop, margin_call, forced_close
    tag: str = ""
    pnl_r: float = 0.0                 # net_pnl / initial_risk_usd
    mae: float = 0.0                   # max adverse excursion ($)
    mfe: float = 0.0                   # max favorable excursion ($)
    initial_sl_distance: float = 0.0


@dataclass
class SimulationResult:
    """Complete output of a simulation run."""
    trades: list[Trade]
    equity_mtm: np.ndarray             # shape (N,), float64
    cash_equity: np.ndarray            # shape (N,), float64
    drawdown_mtm: np.ndarray           # shape (N,), float64
    positions: np.ndarray              # shape (N,), int8: 1=long, -1=short, 0=flat
    skipped_min_lot: int = 0
    skipped_margin: int = 0
    forced_closes: int = 0


class EventSimulator:
    """Pure-Python reference simulation engine."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        costs = cfg.get("costs", {})
        contract = cfg.get("contract", {})
        sizing = cfg.get("sizing", {})
        swap = costs.get("swap", {})
        slippage = costs.get("slippage", {})

        self.initial_equity = float(sizing.get("initial_equity", 100000.0))
        self.risk_fraction = float(sizing.get("risk_per_trade", 0.01))
        self.contract_size = float(contract.get("contract_size", 100.0))
        self.lot_step = float(contract.get("lot_step", 0.01))
        self.min_lot = float(contract.get("min_lot", 0.01))
        self.max_lot = float(contract.get("max_lot", 50.0))
        self.leverage = float(contract.get("leverage", 100.0))
        self.max_margin_utilisation = float(sizing.get("max_margin_utilisation", 0.80))
        self.stop_out_level = float(sizing.get("stop_out_level", 0.50))
        self.min_lot_budget_tolerance = float(sizing.get("min_lot_budget_tolerance", 1.50))

        self.commission_per_lot = float(costs.get("commission_per_lot", 7.0))
        self.swap_long_per_lot = float(swap.get("long_per_lot_per_night", -3.50))
        self.swap_short_per_lot = float(swap.get("short_per_lot_per_night", 0.80))
        self.slip_atr_fraction = float(slippage.get("base_fraction_of_atr", 0.05))
        self.stop_multiplier = float(slippage.get("stop_multiplier", 1.5))
        self.allow_reversal = bool(cfg.get("execution", {}).get("allow_reversal", True))

    def run(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamps: np.ndarray,
        spreads: np.ndarray,
        atrs: np.ndarray,
        is_rollover: np.ndarray,
        rollover_multiplier: np.ndarray,
        tape: SignalTape,
    ) -> SimulationResult:
        """Execute SignalTape over price bars."""
        n_bars = len(opens)
        tape.validate_or_raise()

        trades: list[Trade] = []
        equity_mtm = np.zeros(n_bars, dtype=np.float64)
        cash_equity_arr = np.zeros(n_bars, dtype=np.float64)
        drawdown_mtm = np.zeros(n_bars, dtype=np.float64)
        positions = np.zeros(n_bars, dtype=np.int8)

        cash_equity = self.initial_equity
        peak_mtm = self.initial_equity
        skipped_min_lot = 0
        skipped_margin = 0
        forced_closes = 0
        trade_counter = 0

        # Position state variables
        in_pos = False
        pos_dir = 0  # 1 for LONG, -1 for SHORT
        pos_lots = 0.0
        pos_entry_price = 0.0
        pos_open_bid = 0.0
        pos_entry_spread = 0.0
        pos_entry_slippage = 0.0
        pos_sl_price = 0.0
        pos_tp_price = np.nan
        pos_trail_dist = np.nan
        pos_be_r = np.nan
        pos_initial_sl_dist = 0.0
        pos_initial_risk_usd = 0.0
        pos_time_stop = 0
        pos_entry_bar = -1
        pos_entry_time = None
        pos_accumulated_swap = 0.0
        pos_tag = ""
        pos_mae = 0.0
        pos_mfe = 0.0
        pending_time_stop_exit = False

        for i in range(n_bars):
            cur_open = float(opens[i])
            cur_high = float(highs[i])
            cur_low = float(lows[i])
            cur_close = float(closes[i])
            cur_ts = pd.Timestamp(timestamps[i])
            cur_spread = float(spreads[i])
            cur_atr = float(atrs[i]) if np.isfinite(atrs[i]) else 1.0
            cur_rollover = bool(is_rollover[i])
            cur_mult = float(rollover_multiplier[i])

            base_slip = max(0.0, self.slip_atr_fraction * cur_atr)
            stop_slip = base_slip * self.stop_multiplier

            # --- Step 1: Open of Bar i (Process signals from bar i-1) ---
            if i > 0:
                sig_act = tape.actions[i - 1]
                sig_sl_dist = float(tape.sl_distances[i - 1])
                sig_tp_dist = float(tape.tp_distances[i - 1])
                sig_trail = float(tape.trail_distances[i - 1])
                sig_time_stop = int(tape.time_stops[i - 1])
                sig_be_r = float(tape.breakeven_r[i - 1])
                sig_tag = tape.tags[i - 1]

                # Check pending time stop exit from close of bar i-1
                if in_pos and pending_time_stop_exit:
                    pending_time_stop_exit = False
                    # Close at open of bar i
                    if pos_dir == 1:
                        fill_bid = cur_open - base_slip
                        raw_bid = cur_open
                    else:
                        fill_bid = cur_open + cur_spread + base_slip
                        raw_bid = cur_open
                    trade = self._make_trade(
                        trade_id=trade_counter,
                        entry_bar=pos_entry_bar,
                        exit_bar=i,
                        entry_time=pos_entry_time,
                        exit_time=cur_ts,
                        direction="LONG" if pos_dir == 1 else "SHORT",
                        lots=pos_lots,
                        entry_price=pos_entry_price,
                        exit_fill=fill_bid,
                        raw_exit_bid=raw_bid,
                        open_entry_bid=pos_open_bid,
                        entry_spread=pos_entry_spread,
                        exit_spread=cur_spread,
                        entry_slippage=pos_entry_slippage,
                        exit_slippage=base_slip,
                        accumulated_swap=pos_accumulated_swap,
                        exit_reason="time_stop",
                        tag=pos_tag,
                        initial_risk=pos_initial_risk_usd,
                        mae=pos_mae,
                        mfe=pos_mfe,
                        initial_sl_dist=pos_initial_sl_dist,
                    )
                    trade_counter += 1
                    trades.append(trade)
                    cash_equity += trade.net_pnl
                    in_pos = False
                    pos_dir = 0

                # Check explicit CLOSE signal
                if in_pos and sig_act == Action.CLOSE:
                    if pos_dir == 1:
                        fill_bid = cur_open - base_slip
                        raw_bid = cur_open
                    else:
                        fill_bid = cur_open + cur_spread + base_slip
                        raw_bid = cur_open
                    trade = self._make_trade(
                        trade_id=trade_counter,
                        entry_bar=pos_entry_bar,
                        exit_bar=i,
                        entry_time=pos_entry_time,
                        exit_time=cur_ts,
                        direction="LONG" if pos_dir == 1 else "SHORT",
                        lots=pos_lots,
                        entry_price=pos_entry_price,
                        exit_fill=fill_bid,
                        raw_exit_bid=raw_bid,
                        open_entry_bid=pos_open_bid,
                        entry_spread=pos_entry_spread,
                        exit_spread=cur_spread,
                        entry_slippage=pos_entry_slippage,
                        exit_slippage=base_slip,
                        accumulated_swap=pos_accumulated_swap,
                        exit_reason="signal_close",
                        tag=pos_tag,
                        initial_risk=pos_initial_risk_usd,
                        mae=pos_mae,
                        mfe=pos_mfe,
                        initial_sl_dist=pos_initial_sl_dist,
                    )
                    trade_counter += 1
                    trades.append(trade)
                    cash_equity += trade.net_pnl
                    in_pos = False
                    pos_dir = 0

                # Check Reversal signal
                if in_pos and self.allow_reversal:
                    is_reversal = (
                        (pos_dir == 1 and sig_act == Action.ENTER_SHORT)
                        or (pos_dir == -1 and sig_act == Action.ENTER_LONG)
                    )
                    if is_reversal:
                        if pos_dir == 1:
                            fill_bid = cur_open - base_slip
                            raw_bid = cur_open
                        else:
                            fill_bid = cur_open + cur_spread + base_slip
                            raw_bid = cur_open
                        trade = self._make_trade(
                            trade_id=trade_counter,
                            entry_bar=pos_entry_bar,
                            exit_bar=i,
                            entry_time=pos_entry_time,
                            exit_time=cur_ts,
                            direction="LONG" if pos_dir == 1 else "SHORT",
                            lots=pos_lots,
                            entry_price=pos_entry_price,
                            exit_fill=fill_bid,
                            raw_exit_bid=raw_bid,
                            open_entry_bid=pos_open_bid,
                            entry_spread=pos_entry_spread,
                            exit_spread=cur_spread,
                            entry_slippage=pos_entry_slippage,
                            exit_slippage=base_slip,
                            accumulated_swap=pos_accumulated_swap,
                            exit_reason="signal_reversal",
                            tag=pos_tag,
                            initial_risk=pos_initial_risk_usd,
                            mae=pos_mae,
                            mfe=pos_mfe,
                            initial_sl_dist=pos_initial_sl_dist,
                        )
                        trade_counter += 1
                        trades.append(trade)
                        cash_equity += trade.net_pnl
                        in_pos = False
                        pos_dir = 0

                # Open new position if flat
                if not in_pos and (sig_act == Action.ENTER_LONG or sig_act == Action.ENTER_SHORT):
                    is_long = (sig_act == Action.ENTER_LONG)
                    entry_slip = base_slip

                    # Position sizing calculation
                    size_res = calculate_position_size(
                        equity=cash_equity,
                        sl_distance=sig_sl_dist,
                        price=cur_open,
                        spread=cur_spread,
                        expected_slippage=entry_slip,
                        risk_fraction=self.risk_fraction,
                        contract_size=self.contract_size,
                        lot_step=self.lot_step,
                        min_lot=self.min_lot,
                        max_lot=self.max_lot,
                        leverage=self.leverage,
                        max_margin_utilisation=self.max_margin_utilisation,
                        min_lot_budget_tolerance=self.min_lot_budget_tolerance,
                    )

                    if size_res.skipped:
                        if size_res.skip_reason == "skipped_min_lot":
                            skipped_min_lot += 1
                        elif size_res.skip_reason == "skipped_margin":
                            skipped_margin += 1
                    else:
                        in_pos = True
                        pos_dir = 1 if is_long else -1
                        pos_lots = size_res.lots
                        pos_open_bid = cur_open
                        pos_entry_spread = cur_spread
                        pos_entry_slippage = entry_slip
                        pos_initial_sl_dist = sig_sl_dist
                        pos_initial_risk_usd = size_res.risk_usd
                        pos_trail_dist = sig_trail
                        pos_be_r = sig_be_r
                        pos_time_stop = sig_time_stop
                        pos_entry_bar = i
                        pos_entry_time = cur_ts
                        pos_accumulated_swap = 0.0
                        pos_tag = sig_tag
                        pos_mae = 0.0
                        pos_mfe = 0.0
                        pending_time_stop_exit = False

                        if is_long:
                            pos_entry_price = cur_open + cur_spread + entry_slip
                            pos_sl_price = pos_entry_price - sig_sl_dist
                            pos_tp_price = (pos_entry_price + sig_tp_dist) if np.isfinite(sig_tp_dist) else np.nan
                        else:
                            pos_entry_price = cur_open - entry_slip
                            pos_sl_price = pos_entry_price + sig_sl_dist
                            pos_tp_price = (pos_entry_price - sig_tp_dist) if np.isfinite(sig_tp_dist) else np.nan

            # --- Step 2: Intrabar checks for current position (ENTRY BAR IS LIVE!) ---
            if in_pos:
                # Update MAE / MFE
                size_oz = pos_lots * self.contract_size
                if pos_dir == 1:
                    adv = (pos_entry_price - cur_low) * size_oz
                    fav = (cur_high - pos_entry_price) * size_oz
                    pos_mae = max(pos_mae, max(0.0, adv))
                    pos_mfe = max(pos_mfe, max(0.0, fav))
                else:
                    adv = ((cur_high + cur_spread) - pos_entry_price) * size_oz
                    fav = (pos_entry_price - (cur_low + cur_spread)) * size_oz
                    pos_mae = max(pos_mae, max(0.0, adv))
                    pos_mfe = max(pos_mfe, max(0.0, fav))

                # Margin call check at worst intrabar price
                req_margin = (pos_lots * self.contract_size * cur_open) / self.leverage
                worst_px = cur_low if pos_dir == 1 else (cur_high + cur_spread)
                if pos_dir == 1:
                    worst_equity = cash_equity + (worst_px - pos_entry_price) * size_oz
                else:
                    worst_equity = cash_equity + (pos_entry_price - worst_px) * size_oz

                if check_margin_call(worst_equity, req_margin, self.stop_out_level):
                    # Stop-out triggered!
                    if pos_dir == 1:
                        fill_bid = cur_low - stop_slip
                        raw_bid = cur_low
                    else:
                        fill_bid = cur_high + cur_spread + stop_slip
                        raw_bid = cur_high
                    trade = self._make_trade(
                        trade_id=trade_counter,
                        entry_bar=pos_entry_bar,
                        exit_bar=i,
                        entry_time=pos_entry_time,
                        exit_time=cur_ts,
                        direction="LONG" if pos_dir == 1 else "SHORT",
                        lots=pos_lots,
                        entry_price=pos_entry_price,
                        exit_fill=fill_bid,
                        raw_exit_bid=raw_bid,
                        open_entry_bid=pos_open_bid,
                        entry_spread=pos_entry_spread,
                        exit_spread=cur_spread,
                        entry_slippage=pos_entry_slippage,
                        exit_slippage=stop_slip,
                        accumulated_swap=pos_accumulated_swap,
                        exit_reason="margin_call",
                        tag=pos_tag,
                        initial_risk=pos_initial_risk_usd,
                        mae=pos_mae,
                        mfe=pos_mfe,
                        initial_sl_dist=pos_initial_sl_dist,
                    )
                    trade_counter += 1
                    trades.append(trade)
                    cash_equity += trade.net_pnl
                    in_pos = False
                    pos_dir = 0

            # SL and TP trigger evaluation
            if in_pos:
                is_entry_bar = (i == pos_entry_bar)
                hit_sl = False
                hit_tp = False
                sl_fill_bid = 0.0
                sl_raw_bid = 0.0
                tp_fill_bid = 0.0
                tp_raw_bid = 0.0

                if pos_dir == 1:  # LONG
                    # Gap check on subsequent bars: if open <= sl_price -> gap fill at open + slippage
                    if not is_entry_bar and cur_open <= pos_sl_price:
                        hit_sl = True
                        sl_fill_bid = cur_open - stop_slip
                        sl_raw_bid = cur_open
                    elif cur_low <= pos_sl_price:
                        hit_sl = True
                        sl_fill_bid = pos_sl_price - stop_slip
                        sl_raw_bid = pos_sl_price

                    if np.isfinite(pos_tp_price) and cur_high >= pos_tp_price:
                        hit_tp = True
                        tp_fill_bid = pos_tp_price
                        tp_raw_bid = pos_tp_price

                else:  # SHORT
                    ask_open = cur_open + cur_spread
                    ask_high = cur_high + cur_spread
                    ask_low = cur_low + cur_spread

                    # Gap check on subsequent bars: if ask_open >= sl_price -> gap fill at open + slippage
                    if not is_entry_bar and ask_open >= pos_sl_price:
                        hit_sl = True
                        sl_fill_bid = ask_open + stop_slip
                        sl_raw_bid = cur_open
                    elif ask_high >= pos_sl_price:
                        hit_sl = True
                        sl_fill_bid = pos_sl_price + stop_slip
                        sl_raw_bid = pos_sl_price - cur_spread

                    if np.isfinite(pos_tp_price) and ask_low <= pos_tp_price:
                        hit_tp = True
                        tp_fill_bid = pos_tp_price
                        tp_raw_bid = pos_tp_price - cur_spread

                # Conservative rule: same-bar SL and TP -> SL was hit first!
                if hit_sl and hit_tp:
                    hit_tp = False

                if hit_sl:
                    trade = self._make_trade(
                        trade_id=trade_counter,
                        entry_bar=pos_entry_bar,
                        exit_bar=i,
                        entry_time=pos_entry_time,
                        exit_time=cur_ts,
                        direction="LONG" if pos_dir == 1 else "SHORT",
                        lots=pos_lots,
                        entry_price=pos_entry_price,
                        exit_fill=sl_fill_bid,
                        raw_exit_bid=sl_raw_bid,
                        open_entry_bid=pos_open_bid,
                        entry_spread=pos_entry_spread,
                        exit_spread=cur_spread,
                        entry_slippage=pos_entry_slippage,
                        exit_slippage=stop_slip,
                        accumulated_swap=pos_accumulated_swap,
                        exit_reason="stop_loss",
                        tag=pos_tag,
                        initial_risk=pos_initial_risk_usd,
                        mae=pos_mae,
                        mfe=pos_mfe,
                        initial_sl_dist=pos_initial_sl_dist,
                    )
                    trade_counter += 1
                    trades.append(trade)
                    cash_equity += trade.net_pnl
                    in_pos = False
                    pos_dir = 0
                elif hit_tp:
                    trade = self._make_trade(
                        trade_id=trade_counter,
                        entry_bar=pos_entry_bar,
                        exit_bar=i,
                        entry_time=pos_entry_time,
                        exit_time=cur_ts,
                        direction="LONG" if pos_dir == 1 else "SHORT",
                        lots=pos_lots,
                        entry_price=pos_entry_price,
                        exit_fill=tp_fill_bid,
                        raw_exit_bid=tp_raw_bid,
                        open_entry_bid=pos_open_bid,
                        entry_spread=pos_entry_spread,
                        exit_spread=cur_spread,
                        entry_slippage=pos_entry_slippage,
                        exit_slippage=0.0,  # limit order: no slippage
                        accumulated_swap=pos_accumulated_swap,
                        exit_reason="take_profit",
                        tag=pos_tag,
                        initial_risk=pos_initial_risk_usd,
                        mae=pos_mae,
                        mfe=pos_mfe,
                        initial_sl_dist=pos_initial_sl_dist,
                    )
                    trade_counter += 1
                    trades.append(trade)
                    cash_equity += trade.net_pnl
                    in_pos = False
                    pos_dir = 0

            # --- Step 3: Close of Bar i (Trailing stop, Breakeven, Time stop, Rollover Swap) ---
            if in_pos:
                # 1. Rollover Swap
                if cur_rollover and cur_mult > 0:
                    rate = self.swap_long_per_lot if pos_dir == 1 else self.swap_short_per_lot
                    pos_accumulated_swap += rate * pos_lots * cur_mult

                # 2. Trailing stop and breakeven update (applies from NEXT bar)
                if pos_dir == 1:
                    # Trailing stop
                    if np.isfinite(pos_trail_dist) and pos_trail_dist > 0:
                        candidate_sl = cur_close - pos_trail_dist
                        if candidate_sl > pos_sl_price:
                            pos_sl_price = candidate_sl
                    # Breakeven stop
                    if np.isfinite(pos_be_r) and pos_be_r > 0:
                        profit_usd = cur_close - pos_entry_price
                        if profit_usd >= (pos_be_r * pos_initial_sl_dist):
                            pos_sl_price = max(pos_sl_price, pos_entry_price)
                else:
                    cur_ask_close = cur_close + cur_spread
                    # Trailing stop
                    if np.isfinite(pos_trail_dist) and pos_trail_dist > 0:
                        candidate_sl = cur_ask_close + pos_trail_dist
                        if candidate_sl < pos_sl_price:
                            pos_sl_price = candidate_sl
                    # Breakeven stop
                    if np.isfinite(pos_be_r) and pos_be_r > 0:
                        profit_usd = pos_entry_price - cur_ask_close
                        if profit_usd >= (pos_be_r * pos_initial_sl_dist):
                            pos_sl_price = min(pos_sl_price, pos_entry_price)

                # 3. Time stop evaluation
                bars_held = i - pos_entry_bar + 1
                if pos_time_stop > 0 and bars_held >= pos_time_stop:
                    pending_time_stop_exit = True

            # --- Step 4: Forced close at End of Split ---
            if in_pos and i == n_bars - 1:
                forced_closes += 1
                if pos_dir == 1:
                    fill_bid = cur_close - base_slip
                    raw_bid = cur_close
                else:
                    fill_bid = cur_close + cur_spread + base_slip
                    raw_bid = cur_close
                trade = self._make_trade(
                    trade_id=trade_counter,
                    entry_bar=pos_entry_bar,
                    exit_bar=i,
                    entry_time=pos_entry_time,
                    exit_time=cur_ts,
                    direction="LONG" if pos_dir == 1 else "SHORT",
                    lots=pos_lots,
                    entry_price=pos_entry_price,
                    exit_fill=fill_bid,
                    raw_exit_bid=raw_bid,
                    open_entry_bid=pos_open_bid,
                    entry_spread=pos_entry_spread,
                    exit_spread=cur_spread,
                    entry_slippage=pos_entry_slippage,
                    exit_slippage=base_slip,
                    accumulated_swap=pos_accumulated_swap,
                    exit_reason="forced_close",
                    tag=pos_tag,
                    initial_risk=pos_initial_risk_usd,
                    mae=pos_mae,
                    mfe=pos_mfe,
                    initial_sl_dist=pos_initial_sl_dist,
                )
                trade_counter += 1
                trades.append(trade)
                cash_equity += trade.net_pnl
                in_pos = False
                pos_dir = 0

            # --- Step 5: Mark-to-Market Equity at Bar Close ---
            if in_pos:
                positions[i] = pos_dir
                size_oz = pos_lots * self.contract_size
                mkt_slip = base_slip

                if pos_dir == 1:
                    # Long position marked at bid close
                    unrealized_exit = cur_close - mkt_slip
                    unrealized_price_pnl = (unrealized_exit - pos_entry_price) * size_oz
                else:
                    # Short position marked at ask close
                    unrealized_exit = cur_close + cur_spread + mkt_slip
                    unrealized_price_pnl = (pos_entry_price - unrealized_exit) * size_oz

                unrealized_comm = self.commission_per_lot * pos_lots
                cur_mtm = cash_equity + unrealized_price_pnl - unrealized_comm + pos_accumulated_swap
            else:
                positions[i] = 0
                cur_mtm = cash_equity

            equity_mtm[i] = cur_mtm
            cash_equity_arr[i] = cash_equity
            if cur_mtm > peak_mtm:
                peak_mtm = cur_mtm
            drawdown_mtm[i] = (peak_mtm - cur_mtm) / peak_mtm if peak_mtm > 0 else 0.0

        return SimulationResult(
            trades=trades,
            equity_mtm=equity_mtm,
            cash_equity=cash_equity_arr,
            drawdown_mtm=drawdown_mtm,
            positions=positions,
            skipped_min_lot=skipped_min_lot,
            skipped_margin=skipped_margin,
            forced_closes=forced_closes,
        )

    def _make_trade(
        self,
        trade_id: int,
        entry_bar: int,
        exit_bar: int,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
        direction: str,
        lots: float,
        entry_price: float,
        exit_fill: float,
        raw_exit_bid: float,
        open_entry_bid: float,
        entry_spread: float,
        exit_spread: float,
        entry_slippage: float,
        exit_slippage: float,
        accumulated_swap: float,
        exit_reason: str,
        tag: str,
        initial_risk: float,
        mae: float,
        mfe: float,
        initial_sl_dist: float,
    ) -> Trade:
        """Construct Trade and strictly assert exact cost accounting identity."""
        size_oz = lots * self.contract_size
        is_long = (direction == "LONG")

        if is_long:
            gross_pnl = (raw_exit_bid - open_entry_bid) * size_oz
            spread_cost = entry_spread * size_oz
            slippage_cost = (entry_slippage + exit_slippage) * size_oz
            price_pnl = (exit_fill - entry_price) * size_oz
        else:
            gross_pnl = (open_entry_bid - raw_exit_bid) * size_oz
            spread_cost = exit_spread * size_oz
            slippage_cost = (entry_slippage + exit_slippage) * size_oz
            price_pnl = (entry_price - exit_fill) * size_oz

        commission = self.commission_per_lot * lots
        swap_cost = -accumulated_swap

        gross_pnl = round(gross_pnl, 4)
        spread_cost = round(spread_cost, 4)
        slippage_cost = round(slippage_cost, 4)
        commission = round(commission, 4)
        swap_cost = round(swap_cost, 4)
        net_pnl = round(gross_pnl - spread_cost - slippage_cost - commission - swap_cost, 4)

        bars_held = exit_bar - entry_bar
        pnl_r = round((net_pnl / initial_risk), 4) if initial_risk > 0 else 0.0

        return Trade(
            trade_id=trade_id,
            entry_bar_idx=entry_bar,
            exit_bar_idx=exit_bar,
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            lots=lots,
            entry_price=entry_price,
            exit_price=exit_fill,
            gross_pnl=gross_pnl,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            commission=commission,
            swap_cost=swap_cost,
            net_pnl=net_pnl,
            bars_held=bars_held,
            exit_reason=exit_reason,
            tag=tag,
            pnl_r=pnl_r,
            mae=round(mae, 2),
            mfe=round(mfe, 2),
            initial_sl_distance=round(initial_sl_dist, 4),
        )
