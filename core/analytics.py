"""
core/analytics.py – Comprehensive MTM portfolio analytics (Phase F2.4).

Computes all Section F2.4 metrics from mark-to-market equity series and trade lists:
- Return, CAGR, daily-return Sharpe & Sortino (annualized with sqrt(252)), Calmar, Omega
- MTM Max drawdown, avg drawdown, max duration, time under water %, Ulcer index, recovery factor
- Profit factor, win rate, expectancy ($ and R), payoff, streaks
- Holding time, exposure %, trades per month, long/short breakdown, exit-reason breakdown, MAE/MFE
- Monthly table (calendar months, compounded, string keys), avg/median monthly return,
  best/worst month, % profitable months, avg win/loss month, avg monthly max drawdown
- Yearly table (calendar years, compounded, string keys)
- Cost-to-gross ratio
- Buy-and-hold benchmark return and strategy beta
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from core.event_simulator import Trade


def compute_analytics(
    trades: list[Trade],
    equity_mtm: np.ndarray,
    positions: np.ndarray,
    timestamps: np.ndarray,
    closes: np.ndarray,
    opens: np.ndarray,
    initial_equity: float = 100000.0,
    cfg: dict | None = None,
) -> dict[str, Any]:
    """Compute complete analytical report."""
    n_bars = len(equity_mtm)
    if n_bars == 0:
        return {"error": "empty_bars", "total_trades": 0}

    ts_series = pd.to_datetime(timestamps)
    final_equity = float(equity_mtm[-1])
    total_net_pnl = final_equity - initial_equity
    total_return_pct = (final_equity / initial_equity) - 1.0

    # Date range & CAGR
    start_time = ts_series[0]
    end_time = ts_series[-1]
    total_seconds = max(1.0, (end_time - start_time).total_seconds())
    total_days = total_seconds / 86400.0
    years = total_days / 365.25

    if years > 0 and final_equity > 0:
        cagr = float((final_equity / initial_equity) ** (1.0 / years) - 1.0)
    else:
        cagr = -1.0 if final_equity <= 0 else 0.0

    # Drawdown series (MTM)
    peaks = np.maximum.accumulate(equity_mtm)
    dd_series = np.where(peaks > 0, (peaks - equity_mtm) / peaks, 0.0)
    max_drawdown = float(np.max(dd_series)) if len(dd_series) else 0.0
    average_drawdown = float(np.mean(dd_series)) if len(dd_series) else 0.0
    time_under_water = float(np.mean(dd_series > 1e-6)) if len(dd_series) else 0.0
    ulcer_index = float(np.sqrt(np.mean(dd_series ** 2))) if len(dd_series) else 0.0

    # Max drawdown duration
    max_dd_bars = 0
    cur_dd_bars = 0
    for dd in dd_series:
        if dd > 1e-6:
            cur_dd_bars += 1
            if cur_dd_bars > max_dd_bars:
                max_dd_bars = cur_dd_bars
        else:
            cur_dd_bars = 0

    max_dd_days = (max_dd_bars / n_bars) * total_days if n_bars > 0 else 0.0
    calmar = float(cagr / max_drawdown) if max_drawdown > 1e-6 else 0.0
    max_dollar_dd = float(np.max(peaks - equity_mtm)) if len(peaks) else 0.0
    recovery_factor = float(total_net_pnl / max_dollar_dd) if max_dollar_dd > 1e-6 else 0.0

    # Exposure
    exposure_pct = float(np.mean(positions != 0)) if len(positions) else 0.0

    # Daily returns (group by calendar trading day)
    eq_df = pd.DataFrame({"timestamp": ts_series, "equity": equity_mtm, "close": closes})
    eq_df["date"] = eq_df["timestamp"].dt.date
    daily_summary = eq_df.groupby("date").agg({"equity": "last", "close": "last"})
    daily_returns = daily_summary["equity"].pct_change().dropna()
    daily_asset_returns = daily_summary["close"].pct_change().dropna()

    if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 1e-12:
        mean_d = float(daily_returns.mean())
        std_d = float(daily_returns.std(ddof=1))
        sharpe = float((mean_d / std_d) * np.sqrt(252))
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0:
            downside_std = float(np.sqrt(np.mean(downside ** 2)))
            sortino = float((mean_d / downside_std) * np.sqrt(252)) if downside_std > 1e-12 else 0.0
        else:
            sortino = 99.0

        # Omega ratio
        gains = float(daily_returns[daily_returns > 0].sum())
        losses = float(np.abs(daily_returns[daily_returns < 0].sum()))
        omega = (gains / losses) if losses > 1e-12 else (99.0 if gains > 0 else 0.0)
    else:
        sharpe = 0.0
        sortino = 0.0
        omega = 0.0

    # Buy and hold benchmark
    bnh_return_pct = float((closes[-1] / opens[0]) - 1.0) if opens[0] > 0 else 0.0
    if len(daily_returns) > 5 and len(daily_asset_returns) > 5:
        aligned = pd.concat([daily_returns, daily_asset_returns], axis=1).dropna()
        if len(aligned) > 5 and aligned.iloc[:, 1].var() > 1e-12:
            cov = float(np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1])
            var_m = float(np.var(aligned.iloc[:, 1]))
            beta = float(cov / var_m)
        else:
            beta = 0.0
    else:
        beta = 0.0

    # Monthly table & calendar breakdown
    eq_df["year"] = eq_df["timestamp"].dt.year
    eq_df["month"] = eq_df["timestamp"].dt.month
    monthly_groups = eq_df.groupby(["year", "month"])

    monthly_table: dict[str, dict[str, float]] = {}
    monthly_returns_list: list[float] = []
    monthly_max_dds_list: list[float] = []

    for (yr, mo), group in monthly_groups:
        yr_str = str(yr)
        mo_str = f"{mo:02d}"
        if yr_str not in monthly_table:
            monthly_table[yr_str] = {}

        eq_start = float(group["equity"].iloc[0])
        eq_end = float(group["equity"].iloc[-1])
        m_ret = (eq_end / eq_start) - 1.0 if eq_start > 0 else 0.0
        monthly_table[yr_str][mo_str] = round(m_ret * 100.0, 2)
        monthly_returns_list.append(m_ret)

        grp_peaks = np.maximum.accumulate(group["equity"].to_numpy())
        grp_dds = np.where(grp_peaks > 0, (grp_peaks - group["equity"].to_numpy()) / grp_peaks, 0.0)
        monthly_max_dds_list.append(float(np.max(grp_dds)))

    if monthly_returns_list:
        m_arr = np.array(monthly_returns_list)
        avg_monthly_return = float(np.mean(m_arr))
        median_monthly_return = float(np.median(m_arr))
        best_month = float(np.max(m_arr))
        worst_month = float(np.min(m_arr))
        pct_profitable_months = float(np.mean(m_arr > 0))
        win_months = m_arr[m_arr > 0]
        loss_months = m_arr[m_arr < 0]
        avg_winning_month = float(np.mean(win_months)) if len(win_months) else 0.0
        avg_losing_month = float(np.mean(loss_months)) if len(loss_months) else 0.0
        avg_monthly_max_dd = float(np.mean(monthly_max_dds_list)) if monthly_max_dds_list else 0.0
    else:
        avg_monthly_return = 0.0
        median_monthly_return = 0.0
        best_month = 0.0
        worst_month = 0.0
        pct_profitable_months = 0.0
        avg_winning_month = 0.0
        avg_losing_month = 0.0
        avg_monthly_max_dd = 0.0

    # Yearly table
    yearly_groups = eq_df.groupby("year")
    yearly_table: dict[str, dict[str, Any]] = {}
    for yr, group in yearly_groups:
        yr_str = str(yr)
        y_start = float(group["equity"].iloc[0])
        y_end = float(group["equity"].iloc[-1])
        y_ret = (y_end / y_start) - 1.0 if y_start > 0 else 0.0
        y_pnl = y_end - y_start
        yearly_table[yr_str] = {
            "return_pct": round(y_ret * 100.0, 2),
            "net_pnl": round(y_pnl, 2),
        }

    # Trade statistics
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "total_trades": 0,
            "error": "no_trades",
            "total_net_pnl": round(total_net_pnl, 2),
            "total_gross_pnl": 0.0,
            "total_costs": 0.0,
            "cost_gross_ratio": 0.0,
            "total_return_pct": round(total_return_pct * 100.0, 2),
            "final_equity": round(final_equity, 2),
            "cagr": round(cagr * 100.0, 2),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "calmar": round(calmar, 4),
            "omega": round(omega, 4),
            "max_drawdown": round(max_drawdown, 4),
            "average_drawdown": round(average_drawdown, 4),
            "max_drawdown_bars": max_dd_bars,
            "max_drawdown_days": round(max_dd_days, 1),
            "time_under_water": round(time_under_water, 4),
            "ulcer_index": round(ulcer_index, 4),
            "recovery_factor": round(recovery_factor, 4),
            "exposure_pct": round(exposure_pct, 4),
            "buy_and_hold_return_pct": round(bnh_return_pct * 100.0, 2),
            "beta": round(beta, 4),
            "monthly_table": monthly_table,
            "yearly_table": yearly_table,
            "yearly_pnl": {yr: data["net_pnl"] for yr, data in yearly_table.items()},
            "status": "completed",
        }

    net_pnls = np.array([t.net_pnl for t in trades])
    gross_pnls = np.array([t.gross_pnl for t in trades])
    spread_costs = np.array([t.spread_cost for t in trades])
    slippage_costs = np.array([t.slippage_cost for t in trades])
    commissions = np.array([t.commission for t in trades])
    swap_costs = np.array([t.swap_cost for t in trades])
    total_costs = spread_costs + slippage_costs + commissions + swap_costs

    winners = net_pnls[net_pnls > 0]
    losers = net_pnls[net_pnls < 0]
    n_win = len(winners)
    n_loss = len(losers)
    win_rate = (n_win / n_trades) if n_trades > 0 else 0.0

    gross_wins = float(winners.sum()) if len(winners) else 0.0
    gross_losses = float(np.abs(losers.sum())) if len(losers) else 0.0
    profit_factor = (gross_wins / gross_losses) if gross_losses > 1e-12 else (999.0 if gross_wins > 0 else 0.0)

    avg_win = float(winners.mean()) if len(winners) else 0.0
    avg_loss = float(losers.mean()) if len(losers) else 0.0
    payoff = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-12 else 0.0

    largest_win = float(winners.max()) if len(winners) else 0.0
    largest_loss = float(losers.min()) if len(losers) else 0.0
    expectancy_usd = float(net_pnls.mean())
    expectancy_r = float(np.mean([t.pnl_r for t in trades]))

    # Consecutive win/loss streaks
    max_cons_wins = 0
    max_cons_losses = 0
    cur_wins = 0
    cur_losses = 0
    for pnl in net_pnls:
        if pnl > 0:
            cur_wins += 1
            cur_losses = 0
            if cur_wins > max_cons_wins:
                max_cons_wins = cur_wins
        elif pnl < 0:
            cur_losses += 1
            cur_wins = 0
            if cur_losses > max_cons_losses:
                max_cons_losses = cur_losses
        else:
            cur_wins = 0
            cur_losses = 0

    # Holding time
    holding_bars = np.array([t.bars_held for t in trades])
    avg_holding_bars = float(np.mean(holding_bars))
    holding_hours = np.array([(t.exit_time - t.entry_time).total_seconds() / 3600.0 for t in trades])
    avg_holding_hours = float(np.mean(holding_hours))

    # Trades per month
    months_span = max(0.1, total_days / 30.4375)
    trades_per_month = n_trades / months_span

    # Long vs Short breakdown
    long_trades = [t for t in trades if t.direction == "LONG"]
    short_trades = [t for t in trades if t.direction == "SHORT"]

    def _side_stats(sub_trades: list[Trade]) -> dict[str, Any]:
        if not sub_trades:
            return {"trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "profit_factor": 0.0}
        sub_pnls = np.array([t.net_pnl for t in sub_trades])
        sub_wins = sub_pnls[sub_pnls > 0]
        sub_losses = np.abs(sub_pnls[sub_pnls < 0])
        sub_gw = float(sub_wins.sum())
        sub_gl = float(sub_losses.sum())
        pf = (sub_gw / sub_gl) if sub_gl > 1e-12 else (999.0 if sub_gw > 0 else 0.0)
        return {
            "trades": len(sub_trades),
            "win_rate": round(len(sub_wins) / len(sub_trades), 4),
            "net_pnl": round(float(sub_pnls.sum()), 2),
            "profit_factor": round(pf, 4),
        }

    long_stats = _side_stats(long_trades)
    short_stats = _side_stats(short_trades)

    # Exit reason breakdown
    exit_reasons: dict[str, dict[str, Any]] = {}
    for t in trades:
        r = t.exit_reason
        if r not in exit_reasons:
            exit_reasons[r] = {"count": 0, "net_pnl": 0.0}
        exit_reasons[r]["count"] += 1
        exit_reasons[r]["net_pnl"] += t.net_pnl
    for r in exit_reasons:
        exit_reasons[r]["pct_of_trades"] = round(exit_reasons[r]["count"] / n_trades, 4)
        exit_reasons[r]["net_pnl"] = round(exit_reasons[r]["net_pnl"], 2)

    # MAE / MFE
    avg_mae = float(np.mean([t.mae for t in trades]))
    avg_mfe = float(np.mean([t.mfe for t in trades]))

    # Cost-to-gross ratio
    tot_costs = float(total_costs.sum())
    tot_gross_wins = float(gross_pnls[gross_pnls > 0].sum()) if np.any(gross_pnls > 0) else 0.0
    cost_gross_ratio = (tot_costs / tot_gross_wins) if tot_gross_wins > 1e-12 else 1.0

    # Add yearly trades to yearly_table
    for t in trades:
        yr_str = str(t.entry_time.year)
        if yr_str in yearly_table:
            yearly_table[yr_str]["trades"] = yearly_table[yr_str].get("trades", 0) + 1

    return {
        "total_trades": n_trades,
        "winners": n_win,
        "losers": n_loss,
        "win_rate": round(win_rate, 4),
        "total_net_pnl": round(total_net_pnl, 2),
        "total_gross_pnl": round(float(gross_pnls.sum()), 2),
        "total_costs": round(tot_costs, 2),
        "spread_cost_total": round(float(spread_costs.sum()), 2),
        "slippage_cost_total": round(float(slippage_costs.sum()), 2),
        "commission_total": round(float(commissions.sum()), 2),
        "swap_cost_total": round(float(swap_costs.sum()), 2),
        "cost_gross_ratio": round(cost_gross_ratio, 4),
        "profit_factor": round(profit_factor, 4),
        "expectancy": round(expectancy_usd, 2),
        "expectancy_usd": round(expectancy_usd, 2),
        "expectancy_r": round(expectancy_r, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff": round(payoff, 4),
        "payoff_ratio": round(payoff, 4),
        "largest_win": round(largest_win, 2),
        "largest_loss": round(largest_loss, 2),
        "max_consecutive_wins": max_cons_wins,
        "max_consecutive_losses": max_cons_losses,
        "avg_holding_bars": round(avg_holding_bars, 1),
        "avg_holding_hours": round(avg_holding_hours, 1),
        "exposure_pct": round(exposure_pct, 4),
        "trades_per_month": round(trades_per_month, 2),
        "trades_per_year": round(trades_per_month * 12.0, 1),
        "years_covered": round(years, 3),
        "long_trades": long_stats["trades"],
        "short_trades": short_stats["trades"],
        "long_win_rate": long_stats["win_rate"],
        "short_win_rate": short_stats["win_rate"],
        "profit_concentration_top5pct": round(float(np.sort(winners)[::-1][:max(1, int(len(trades) * 0.05))].sum()) / max(float(winners.sum()), 1e-9), 4) if len(winners) > 0 else 0.0,
        "yearly_pnl": {yr: data["net_pnl"] for yr, data in yearly_table.items()},
        "total_return_pct": round(total_return_pct * 100.0, 2),
        "final_equity": round(final_equity, 2),
        "cagr": round(cagr * 100.0, 2),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "omega": round(omega, 4),
        "max_drawdown": round(max_drawdown, 4),
        "average_drawdown": round(average_drawdown, 4),
        "max_drawdown_bars": max_dd_bars,
        "max_drawdown_days": round(max_dd_days, 1),
        "time_under_water": round(time_under_water, 4),
        "ulcer_index": round(ulcer_index, 4),
        "recovery_factor": round(recovery_factor, 4),
        "long_breakdown": long_stats,
        "short_breakdown": short_stats,
        "exit_reasons": exit_reasons,
        "avg_mae": round(avg_mae, 2),
        "avg_mfe": round(avg_mfe, 2),
        "monthly_table": monthly_table,
        "average_monthly_return_pct": round(avg_monthly_return * 100.0, 2),
        "median_monthly_return_pct": round(median_monthly_return * 100.0, 2),
        "best_month_pct": round(best_month * 100.0, 2),
        "worst_month_pct": round(worst_month * 100.0, 2),
        "pct_profitable_months": round(pct_profitable_months, 4),
        "avg_winning_month_pct": round(avg_winning_month * 100.0, 2),
        "avg_losing_month_pct": round(avg_losing_month * 100.0, 2),
        "average_monthly_max_drawdown": round(avg_monthly_max_dd, 4),
        "yearly_table": yearly_table,
        "buy_and_hold_return_pct": round(bnh_return_pct * 100.0, 2),
        "beta": round(beta, 4),
        "status": "completed",
    }
