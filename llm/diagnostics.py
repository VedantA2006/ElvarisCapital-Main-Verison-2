"""
llm/diagnostics.py – Train-only performance diagnostics for strategy improvement.

Generates concise, informative markdown summaries of backtest results strictly
from the TRAIN split to guide the LLM's structural refinement loop.
CRITICAL: Never reads, exposes, or references validation or holdout data.
"""

from __future__ import annotations

from typing import Any


def build_train_diagnostics(metrics: dict[str, Any], trades_summary: dict[str, Any] | None = None) -> str:
    """Build a compact markdown report summarizing train-only performance and weaknesses."""
    sharpe = metrics.get("sharpe", 0.0)
    sortino = metrics.get("sortino", 0.0)
    profit_factor = metrics.get("profit_factor", 0.0)
    max_dd = metrics.get("max_drawdown_pct", metrics.get("max_drawdown", 0.0))
    win_rate = metrics.get("win_rate", 0.0)
    trades_count = metrics.get("total_trades", metrics.get("trades", 0))
    cost_ratio = metrics.get("cost_gross_ratio", 0.0)

    lines = [
        "### In-Sample (Train) Performance Diagnostics",
        "",
        "| Metric | Value | Metric | Value |",
        "|---|---|---|---|",
        f"| **Sharpe Ratio** | {sharpe:.2f} | **Sortino Ratio** | {sortino:.2f} |",
        f"| **Profit Factor** | {profit_factor:.2f} | **Max Drawdown** | {max_dd:.1f}% |",
        f"| **Win Rate** | {win_rate:.1f}% | **Total Trades** | {trades_count} |",
        f"| **Cost / Gross Ratio** | {cost_ratio:.1%} | **Data Split** | Train (In-Sample) |",
        "",
    ]

    summary = trades_summary or {}

    # 1. Session Breakdown
    by_session = summary.get("by_session", {})
    if by_session:
        lines.append("#### Performance by Market Session")
        lines.append("| Session | Trades | Win Rate | Net PnL ($) |")
        lines.append("|---|---|---|---|")
        for sess, stats in by_session.items():
            t_cnt = stats.get("trades", 0) if isinstance(stats, dict) else 0
            pnl = stats.get("pnl", 0) if isinstance(stats, dict) else stats
            wr = stats.get("win_rate", 0.0) if isinstance(stats, dict) else 0.0
            lines.append(f"| {sess.title()} | {t_cnt} | {wr:.1f}% | ${pnl:,.2f} |")
        lines.append("")

    # 2. Long vs Short Breakdown
    ls = summary.get("long_short_split", {})
    if ls:
        lines.append("#### Directional Breakdown")
        lines.append(
            f"- Long: {ls.get('long_trades', 0)} trades, {ls.get('long_win_rate', 0.0):.1f}% win rate\n"
            f"- Short: {ls.get('short_trades', 0)} trades, {ls.get('short_win_rate', 0.0):.1f}% win rate"
        )
        lines.append("")

    # 3. Exit Reason Breakdown
    exits = summary.get("by_exit_reason", {})
    if exits:
        lines.append("#### Exit Reason Breakdown")
        lines.append(", ".join(f"{k.upper()}: {v} trades" for k, v in exits.items()))
        lines.append("")

    # 4. Weekday Breakdown
    wd = summary.get("by_weekday", {})
    if wd:
        lines.append("#### Weekday Net PnL")
        lines.append(", ".join(f"{d.title()}: ${val:,.0f}" for d, val in wd.items()))
        lines.append("")

    # 5. Weakness Diagnostics
    weaknesses = []
    if profit_factor < 1.3:
        weaknesses.append("Low profit factor indicates edge is marginal after realistic trading costs.")
    if max_dd > 15.0:
        weaknesses.append(f"Excessive drawdown ({max_dd:.1f}%); consider tighter stop or volatility scaling.")
    if win_rate < 40.0:
        weaknesses.append("Low win rate requires larger payoff ratio; examine false breakout frequency.")
    if by_session:
        # Check if any session is heavily negative
        for sess, stats in by_session.items():
            pnl = stats.get("pnl", 0) if isinstance(stats, dict) else stats
            if pnl < 0:
                weaknesses.append(f"{sess.title()} session is net negative (${pnl:,.2f}); consider session filter.")

    if weaknesses:
        lines.append("#### Observed Structural Weaknesses")
        for w in weaknesses:
            lines.append(f"- {w}")

    return "\n".join(lines)
