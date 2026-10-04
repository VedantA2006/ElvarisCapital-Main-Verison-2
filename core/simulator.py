"""
core/simulator.py – Fast JIT-accelerated (Numba) execution engine (Phase F2).

Implements the exact same execution model and invariants as core/event_simulator.py:
- Bid price basis with exact spread and slippage
- Live entry bar (SL/TP checked on entry bar)
- Same-bar SL/TP: SL hit first
- Gaps: stops fill at open + slippage; limits (TP) fill at TP price
- Trailing / breakeven / time stop evaluated at close, active from next bar
- 17:00 NY rollover swap (trading days only, triple on Wednesday)
- Exact cost reconciliation: gross - spread - slippage - commission - swap == net
- Mark-to-market equity per bar
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
import pandas as pd

from core.event_simulator import SimulationResult, Trade
from core.signals import Action, Direction, SignalTape


EXIT_REASONS = {
    1: "stop_loss",
    2: "take_profit",
    3: "signal_close",
    4: "signal_reversal",
    5: "time_stop",
    6: "margin_call",
    7: "forced_close",
}


@numba.njit(fastmath=False)
def _simulate_numba_core(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    spreads: np.ndarray,
    atrs: np.ndarray,
    is_rollover: np.ndarray,
    rollover_multiplier: np.ndarray,
    actions: np.ndarray,
    sl_distances: np.ndarray,
    tp_distances: np.ndarray,
    trail_distances: np.ndarray,
    time_stops: np.ndarray,
    breakeven_r: np.ndarray,
    initial_equity: float,
    risk_fraction: float,
    contract_size: float,
    lot_step: float,
    min_lot: float,
    max_lot: float,
    leverage: float,
    max_margin_utilisation: float,
    stop_out_level: float,
    min_lot_budget_tolerance: float,
    commission_per_lot: float,
    swap_long_per_lot: float,
    swap_short_per_lot: float,
    slip_atr_fraction: float,
    stop_multiplier: float,
    allow_reversal: bool,
    step_decimals: int,
    max_trades: int,
):
    n_bars = len(opens)
    trades_data = np.zeros((max_trades, 18), dtype=np.float64)
    equity_mtm = np.zeros(n_bars, dtype=np.float64)
    cash_equity_arr = np.zeros(n_bars, dtype=np.float64)
    drawdown_mtm = np.zeros(n_bars, dtype=np.float64)
    positions = np.zeros(n_bars, dtype=np.int8)

    cash_equity = initial_equity
    peak_mtm = initial_equity
    skipped_min_lot = 0
    skipped_margin = 0
    forced_closes = 0
    n_trades = 0

    in_pos = False
    pos_dir = 0
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
    pos_accumulated_swap = 0.0
    pos_mae = 0.0
    pos_mfe = 0.0
    pending_time_stop_exit = False

    for i in range(n_bars):
        cur_open = opens[i]
        cur_high = highs[i]
        cur_low = lows[i]
        cur_close = closes[i]
        cur_spread = spreads[i]
        cur_atr = atrs[i] if np.isfinite(atrs[i]) else 1.0
        cur_rollover = is_rollover[i] == 1
        cur_mult = rollover_multiplier[i]

        base_slip = max(0.0, slip_atr_fraction * cur_atr)
        stop_slip = base_slip * stop_multiplier

        # --- Step 1: Open of Bar i (Process signals from bar i-1) ---
        if i > 0:
            sig_act = actions[i - 1]
            sig_sl_dist = sl_distances[i - 1]
            sig_tp_dist = tp_distances[i - 1]
            sig_trail = trail_distances[i - 1]
            sig_time_stop = time_stops[i - 1]
            sig_be_r = breakeven_r[i - 1]

            # Pending time stop exit from close of bar i-1
            if in_pos and pending_time_stop_exit:
                pending_time_stop_exit = False
                fill_bid = (cur_open - base_slip) if pos_dir == 1 else (cur_open + cur_spread + base_slip)
                raw_bid = cur_open
                if n_trades < max_trades:
                    _record_trade_numba(
                        trades_data, n_trades, n_trades, pos_entry_bar, i,
                        pos_dir, pos_lots, pos_entry_price, fill_bid, raw_bid,
                        pos_open_bid, pos_entry_spread, cur_spread,
                        pos_entry_slippage, base_slip, pos_accumulated_swap,
                        5, pos_initial_risk_usd, pos_mae, pos_mfe,
                        pos_initial_sl_dist, contract_size, commission_per_lot,
                    )
                    cash_equity += trades_data[n_trades, 12]  # net_pnl
                    n_trades += 1
                in_pos = False
                pos_dir = 0

            # Explicit CLOSE signal
            if in_pos and sig_act == 3:  # CLOSE
                fill_bid = (cur_open - base_slip) if pos_dir == 1 else (cur_open + cur_spread + base_slip)
                raw_bid = cur_open
                if n_trades < max_trades:
                    _record_trade_numba(
                        trades_data, n_trades, n_trades, pos_entry_bar, i,
                        pos_dir, pos_lots, pos_entry_price, fill_bid, raw_bid,
                        pos_open_bid, pos_entry_spread, cur_spread,
                        pos_entry_slippage, base_slip, pos_accumulated_swap,
                        3, pos_initial_risk_usd, pos_mae, pos_mfe,
                        pos_initial_sl_dist, contract_size, commission_per_lot,
                    )
                    cash_equity += trades_data[n_trades, 12]
                    n_trades += 1
                in_pos = False
                pos_dir = 0

            # Reversal signal
            if in_pos and allow_reversal:
                is_reversal = (pos_dir == 1 and sig_act == 2) or (pos_dir == -1 and sig_act == 1)
                if is_reversal:
                    fill_bid = (cur_open - base_slip) if pos_dir == 1 else (cur_open + cur_spread + base_slip)
                    raw_bid = cur_open
                    if n_trades < max_trades:
                        _record_trade_numba(
                            trades_data, n_trades, n_trades, pos_entry_bar, i,
                            pos_dir, pos_lots, pos_entry_price, fill_bid, raw_bid,
                            pos_open_bid, pos_entry_spread, cur_spread,
                            pos_entry_slippage, base_slip, pos_accumulated_swap,
                            4, pos_initial_risk_usd, pos_mae, pos_mfe,
                            pos_initial_sl_dist, contract_size, commission_per_lot,
                        )
                        cash_equity += trades_data[n_trades, 12]
                        n_trades += 1
                    in_pos = False
                    pos_dir = 0

            # Open new position if flat
            if (not in_pos) and (sig_act == 1 or sig_act == 2):
                is_long = (sig_act == 1)
                entry_slip = base_slip

                # Position sizing
                risk_budget = risk_fraction * cash_equity
                effective_sl = sig_sl_dist + cur_spread + entry_slip
                risk_per_lot = effective_sl * contract_size
                raw_lots = risk_budget / risk_per_lot
                steps = np.floor(raw_lots / lot_step + 1e-12)
                lots = steps * lot_step
                lots = np.round(lots, step_decimals)

                skip_code = 0
                if lots < min_lot:
                    min_lot_risk = min_lot * risk_per_lot
                    if min_lot_risk <= (min_lot_budget_tolerance * risk_budget + 1e-9):
                        lots = min_lot
                    else:
                        skip_code = 1  # skipped_min_lot
                if skip_code == 0:
                    if lots > max_lot:
                        lots = max_lot
                    lots = np.round(lots, step_decimals)
                    req_margin = (lots * contract_size * cur_open) / leverage
                    if req_margin > (max_margin_utilisation * cash_equity + 1e-9):
                        skip_code = 2  # skipped_margin

                if skip_code == 1:
                    skipped_min_lot += 1
                elif skip_code == 2:
                    skipped_margin += 1
                else:
                    in_pos = True
                    pos_dir = 1 if is_long else -1
                    pos_lots = lots
                    pos_open_bid = cur_open
                    pos_entry_spread = cur_spread
                    pos_entry_slippage = entry_slip
                    pos_initial_sl_dist = sig_sl_dist
                    pos_initial_risk_usd = lots * risk_per_lot
                    pos_trail_dist = sig_trail
                    pos_be_r = sig_be_r
                    pos_time_stop = sig_time_stop
                    pos_entry_bar = i
                    pos_accumulated_swap = 0.0
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
            size_oz = pos_lots * contract_size
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

            req_margin = (pos_lots * contract_size * cur_open) / leverage
            worst_px = cur_low if pos_dir == 1 else (cur_high + cur_spread)
            worst_equity = (cash_equity + (worst_px - pos_entry_price) * size_oz) if pos_dir == 1 else (cash_equity + (pos_entry_price - worst_px) * size_oz)

            if req_margin > 0 and (worst_equity / req_margin) <= (stop_out_level + 1e-9):
                fill_bid = (cur_low - stop_slip) if pos_dir == 1 else (cur_high + cur_spread + stop_slip)
                raw_bid = cur_low if pos_dir == 1 else cur_high
                if n_trades < max_trades:
                    _record_trade_numba(
                        trades_data, n_trades, n_trades, pos_entry_bar, i,
                        pos_dir, pos_lots, pos_entry_price, fill_bid, raw_bid,
                        pos_open_bid, pos_entry_spread, cur_spread,
                        pos_entry_slippage, stop_slip, pos_accumulated_swap,
                        6, pos_initial_risk_usd, pos_mae, pos_mfe,
                        pos_initial_sl_dist, contract_size, commission_per_lot,
                    )
                    cash_equity += trades_data[n_trades, 12]
                    n_trades += 1
                in_pos = False
                pos_dir = 0

        # SL / TP triggers
        if in_pos:
            is_entry_bar = (i == pos_entry_bar)
            hit_sl = False
            hit_tp = False
            sl_fill_bid = 0.0
            sl_raw_bid = 0.0
            tp_fill_bid = 0.0
            tp_raw_bid = 0.0

            if pos_dir == 1:
                if (not is_entry_bar) and (cur_open <= pos_sl_price):
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
            else:
                ask_open = cur_open + cur_spread
                ask_high = cur_high + cur_spread
                ask_low = cur_low + cur_spread

                if (not is_entry_bar) and (ask_open >= pos_sl_price):
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

            if hit_sl and hit_tp:
                hit_tp = False

            if hit_sl:
                if n_trades < max_trades:
                    _record_trade_numba(
                        trades_data, n_trades, n_trades, pos_entry_bar, i,
                        pos_dir, pos_lots, pos_entry_price, sl_fill_bid, sl_raw_bid,
                        pos_open_bid, pos_entry_spread, cur_spread,
                        pos_entry_slippage, stop_slip, pos_accumulated_swap,
                        1, pos_initial_risk_usd, pos_mae, pos_mfe,
                        pos_initial_sl_dist, contract_size, commission_per_lot,
                    )
                    cash_equity += trades_data[n_trades, 12]
                    n_trades += 1
                in_pos = False
                pos_dir = 0
            elif hit_tp:
                if n_trades < max_trades:
                    _record_trade_numba(
                        trades_data, n_trades, n_trades, pos_entry_bar, i,
                        pos_dir, pos_lots, pos_entry_price, tp_fill_bid, tp_raw_bid,
                        pos_open_bid, pos_entry_spread, cur_spread,
                        pos_entry_slippage, 0.0, pos_accumulated_swap,
                        2, pos_initial_risk_usd, pos_mae, pos_mfe,
                        pos_initial_sl_dist, contract_size, commission_per_lot,
                    )
                    cash_equity += trades_data[n_trades, 12]
                    n_trades += 1
                in_pos = False
                pos_dir = 0

        # --- Step 3: Close of Bar i ---
        if in_pos:
            if cur_rollover and cur_mult > 0:
                rate = swap_long_per_lot if pos_dir == 1 else swap_short_per_lot
                pos_accumulated_swap += rate * pos_lots * cur_mult

            if pos_dir == 1:
                if np.isfinite(pos_trail_dist) and pos_trail_dist > 0:
                    cand = cur_close - pos_trail_dist
                    if cand > pos_sl_price:
                        pos_sl_price = cand
                if np.isfinite(pos_be_r) and pos_be_r > 0:
                    profit_usd = cur_close - pos_entry_price
                    if profit_usd >= (pos_be_r * pos_initial_sl_dist):
                        pos_sl_price = max(pos_sl_price, pos_entry_price)
            else:
                cur_ask_close = cur_close + cur_spread
                if np.isfinite(pos_trail_dist) and pos_trail_dist > 0:
                    cand = cur_ask_close + pos_trail_dist
                    if cand < pos_sl_price:
                        pos_sl_price = cand
                if np.isfinite(pos_be_r) and pos_be_r > 0:
                    profit_usd = pos_entry_price - cur_ask_close
                    if profit_usd >= (pos_be_r * pos_initial_sl_dist):
                        pos_sl_price = min(pos_sl_price, pos_entry_price)

            bars_held = i - pos_entry_bar + 1
            if pos_time_stop > 0 and bars_held >= pos_time_stop:
                pending_time_stop_exit = True

        # --- Step 4: End of split forced close ---
        if in_pos and i == n_bars - 1:
            forced_closes += 1
            fill_bid = (cur_close - base_slip) if pos_dir == 1 else (cur_close + cur_spread + base_slip)
            raw_bid = cur_close
            if n_trades < max_trades:
                _record_trade_numba(
                    trades_data, n_trades, n_trades, pos_entry_bar, i,
                    pos_dir, pos_lots, pos_entry_price, fill_bid, raw_bid,
                    pos_open_bid, pos_entry_spread, cur_spread,
                    pos_entry_slippage, base_slip, pos_accumulated_swap,
                    7, pos_initial_risk_usd, pos_mae, pos_mfe,
                    pos_initial_sl_dist, contract_size, commission_per_lot,
                )
                cash_equity += trades_data[n_trades, 12]
                n_trades += 1
            in_pos = False
            pos_dir = 0

        # --- Step 5: Mark-to-market Equity ---
        if in_pos:
            positions[i] = pos_dir
            size_oz = pos_lots * contract_size
            mkt_slip = base_slip

            if pos_dir == 1:
                unrealized_exit = cur_close - mkt_slip
                unrealized_price_pnl = (unrealized_exit - pos_entry_price) * size_oz
            else:
                unrealized_exit = cur_close + cur_spread + mkt_slip
                unrealized_price_pnl = (pos_entry_price - unrealized_exit) * size_oz

            unrealized_comm = commission_per_lot * pos_lots
            cur_mtm = cash_equity + unrealized_price_pnl - unrealized_comm + pos_accumulated_swap
        else:
            positions[i] = 0
            cur_mtm = cash_equity

        equity_mtm[i] = cur_mtm
        cash_equity_arr[i] = cash_equity
        if cur_mtm > peak_mtm:
            peak_mtm = cur_mtm
        drawdown_mtm[i] = (peak_mtm - cur_mtm) / peak_mtm if peak_mtm > 0 else 0.0

    return (
        trades_data[:n_trades],
        equity_mtm,
        cash_equity_arr,
        drawdown_mtm,
        positions,
        skipped_min_lot,
        skipped_margin,
        forced_closes,
    )


@numba.njit(fastmath=False)
def _record_trade_numba(
    trades_data: np.ndarray,
    row_idx: int,
    trade_id: int,
    entry_bar: int,
    exit_bar: int,
    direction: int,
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
    exit_reason_code: int,
    initial_risk: float,
    mae: float,
    mfe: float,
    initial_sl_dist: float,
    contract_size: float,
    commission_per_lot: float,
):
    size_oz = lots * contract_size
    is_long = (direction == 1)

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

    commission = commission_per_lot * lots
    swap_cost = -accumulated_swap

    gross_pnl = np.round(gross_pnl, 4)
    spread_cost = np.round(spread_cost, 4)
    slippage_cost = np.round(slippage_cost, 4)
    commission = np.round(commission, 4)
    swap_cost = np.round(swap_cost, 4)
    net_pnl = np.round(gross_pnl - spread_cost - slippage_cost - commission - swap_cost, 4)

    bars_held = exit_bar - entry_bar
    pnl_r = np.round(net_pnl / initial_risk, 4) if initial_risk > 0 else 0.0

    trades_data[row_idx, 0] = trade_id
    trades_data[row_idx, 1] = entry_bar
    trades_data[row_idx, 2] = exit_bar
    trades_data[row_idx, 3] = direction
    trades_data[row_idx, 4] = lots
    trades_data[row_idx, 5] = entry_price
    trades_data[row_idx, 6] = exit_fill
    trades_data[row_idx, 7] = gross_pnl
    trades_data[row_idx, 8] = spread_cost
    trades_data[row_idx, 9] = slippage_cost
    trades_data[row_idx, 10] = commission
    trades_data[row_idx, 11] = swap_cost
    trades_data[row_idx, 12] = net_pnl
    trades_data[row_idx, 13] = exit_reason_code
    trades_data[row_idx, 14] = pnl_r
    trades_data[row_idx, 15] = mae
    trades_data[row_idx, 16] = mfe
    trades_data[row_idx, 17] = initial_sl_dist


class FastSimulator:
    """Numba-accelerated simulator implementing exact F2.3 execution invariants."""

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

        if self.lot_step < 1:
            self.step_decimals = max(0, int(round(-math.log10(self.lot_step))))
        else:
            self.step_decimals = 0

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
        """Run accelerated simulation."""
        tape.validate_or_raise()
        n_bars = len(opens)
        max_trades = max(10000, n_bars)

        # Ensure correct dtypes for Numba
        opens_f = np.ascontiguousarray(opens, dtype=np.float64)
        highs_f = np.ascontiguousarray(highs, dtype=np.float64)
        lows_f = np.ascontiguousarray(lows, dtype=np.float64)
        closes_f = np.ascontiguousarray(closes, dtype=np.float64)
        spreads_f = np.ascontiguousarray(spreads, dtype=np.float64)
        atrs_f = np.ascontiguousarray(atrs, dtype=np.float64)
        rollover_b = np.ascontiguousarray(is_rollover, dtype=np.uint8)
        rollover_mult_f = np.ascontiguousarray(rollover_multiplier, dtype=np.float64)

        actions_u = np.ascontiguousarray(tape.actions, dtype=np.uint8)
        sl_f = np.ascontiguousarray(tape.sl_distances, dtype=np.float64)
        tp_f = np.ascontiguousarray(tape.tp_distances, dtype=np.float64)
        trail_f = np.ascontiguousarray(tape.trail_distances, dtype=np.float64)
        time_stops_i = np.ascontiguousarray(tape.time_stops, dtype=np.int32)
        be_r_f = np.ascontiguousarray(tape.breakeven_r, dtype=np.float64)

        (
            trades_raw,
            equity_mtm,
            cash_equity,
            drawdown_mtm,
            positions,
            skipped_min_lot,
            skipped_margin,
            forced_closes,
        ) = _simulate_numba_core(
            opens=opens_f,
            highs=highs_f,
            lows=lows_f,
            closes=closes_f,
            spreads=spreads_f,
            atrs=atrs_f,
            is_rollover=rollover_b,
            rollover_multiplier=rollover_mult_f,
            actions=actions_u,
            sl_distances=sl_f,
            tp_distances=tp_f,
            trail_distances=trail_f,
            time_stops=time_stops_i,
            breakeven_r=be_r_f,
            initial_equity=self.initial_equity,
            risk_fraction=self.risk_fraction,
            contract_size=self.contract_size,
            lot_step=self.lot_step,
            min_lot=self.min_lot,
            max_lot=self.max_lot,
            leverage=self.leverage,
            max_margin_utilisation=self.max_margin_utilisation,
            stop_out_level=self.stop_out_level,
            min_lot_budget_tolerance=self.min_lot_budget_tolerance,
            commission_per_lot=self.commission_per_lot,
            swap_long_per_lot=self.swap_long_per_lot,
            swap_short_per_lot=self.swap_short_per_lot,
            slip_atr_fraction=self.slip_atr_fraction,
            stop_multiplier=self.stop_multiplier,
            allow_reversal=self.allow_reversal,
            step_decimals=self.step_decimals,
            max_trades=max_trades,
        )

        # Convert raw trade rows into list of Trade objects
        trades: list[Trade] = []
        for row in trades_raw:
            trade_id = int(row[0])
            entry_bar = int(row[1])
            exit_bar = int(row[2])
            direction = "LONG" if row[3] == 1 else "SHORT"
            lots = float(row[4])
            entry_price = float(row[5])
            exit_price = float(row[6])
            gross_pnl = float(row[7])
            spread_cost = float(row[8])
            slippage_cost = float(row[9])
            commission = float(row[10])
            swap_cost = float(row[11])
            net_pnl = float(row[12])
            exit_reason_code = int(row[13])
            exit_reason = EXIT_REASONS.get(exit_reason_code, "unknown")
            pnl_r = float(row[14])
            mae = float(row[15])
            mfe = float(row[16])
            initial_sl_dist = float(row[17])

            entry_time = pd.Timestamp(timestamps[entry_bar])
            exit_time = pd.Timestamp(timestamps[exit_bar])
            tag = tape.tags[entry_bar - 1] if (entry_bar > 0 and entry_bar - 1 < len(tape.tags)) else ""

            # Check exact cost identity
            reconciled = gross_pnl - spread_cost - slippage_cost - commission - swap_cost
            assert abs(reconciled - net_pnl) < 1e-9, (
                f"FastSimulator accounting mismatch in trade {trade_id}: "
                f"diff={abs(reconciled - net_pnl):.2e}"
            )

            trades.append(
                Trade(
                    trade_id=trade_id,
                    entry_bar_idx=entry_bar,
                    exit_bar_idx=exit_bar,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    direction=direction,
                    lots=lots,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    gross_pnl=round(gross_pnl, 4),
                    spread_cost=round(spread_cost, 4),
                    slippage_cost=round(slippage_cost, 4),
                    commission=round(commission, 4),
                    swap_cost=round(swap_cost, 4),
                    net_pnl=round(net_pnl, 4),
                    bars_held=exit_bar - entry_bar,
                    exit_reason=exit_reason,
                    tag=tag,
                    pnl_r=round(pnl_r, 4),
                    mae=round(mae, 2),
                    mfe=round(mfe, 2),
                    initial_sl_distance=round(initial_sl_dist, 4),
                )
            )

        return SimulationResult(
            trades=trades,
            equity_mtm=equity_mtm,
            cash_equity=cash_equity,
            drawdown_mtm=drawdown_mtm,
            positions=positions,
            skipped_min_lot=int(skipped_min_lot),
            skipped_margin=int(skipped_margin),
            forced_closes=int(forced_closes),
        )
