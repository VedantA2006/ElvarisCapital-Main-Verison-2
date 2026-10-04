"""
llm/diagnostics.py – Train-only performance diagnostics for strategy improvement.

Generates concise, informative markdown summaries of backtest results strictly
from the TRAIN split to guide the LLM's structural refinement loop.
CRITICAL: Never reads, exposes, or references validation or holdout data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


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


# ─── Trade-level diagnostics (F7.3) ─────────────────────────────────────────

_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_DEFAULT_SESSIONS = {
    "asia": {"tz": "Asia/Tokyo", "start": 8, "end": 17},
    "london": {"tz": "Europe/London", "start": 8, "end": 17},
    "new_york": {"tz": "America/New_York", "start": 8, "end": 17},
}


def _get(t: Any, name: str, default: Any = None) -> Any:
    if isinstance(t, dict):
        return t.get(name, default)
    return getattr(t, name, default)


def _to_utc(ts: Any) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def classify_session(ts_utc: pd.Timestamp, sessions: dict[str, Any]) -> str:
    """DST-correct primary session of a UTC timestamp (London+NY overlap reported separately)."""
    open_ = set()
    for name, s in sessions.items():
        local_hour = ts_utc.tz_convert(s["tz"]).hour
        if s["start"] <= local_hour < s["end"]:
            open_.add(name)
    if {"london", "new_york"} <= open_:
        return "overlap_london_ny"
    for name in ("london", "new_york", "asia"):
        if name in open_:
            return name
    return next(iter(sorted(open_)), "off_hours")


def _group_table(title: str, df: pd.DataFrame, key: str, order: list | None = None) -> list[str]:
    g = df.groupby(key)["net_pnl"].agg(["count", "sum", lambda s: (s > 0).mean() * 100])
    g.columns = ["trades", "net", "win"]
    if order is not None:
        g = g.reindex([k for k in order if k in g.index])
    out = [f"#### {title}", "| Bucket | Trades | Net PnL ($) | Win % |", "|---|---|---|---|"]
    for k, row in g.iterrows():
        out.append(f"| {k} | {int(row.trades)} | {row.net:,.0f} | {row.win:.0f}% |")
    return out + [""]


def build_train_diagnostics_from_trades(
    trades: list[Any],
    metrics: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
    split: str = "train",
) -> str:
    """Compact train-only diagnostics computed from the raw trade list.

    Covers: performance by year, month, session, weekday and hour; exit reasons;
    MAE/MFE; holding time; long vs short; cost impact; losing streaks; worst
    drawdown period. Small tables only, never raw trades.
    """
    if split != "train":
        raise ValueError(f"Diagnostics may only be built from the train split, got {split!r}")

    metrics = metrics or {}
    sessions = (cfg or {}).get("sessions") or _DEFAULT_SESSIONS
    sessions = {k: v for k, v in sessions.items() if isinstance(v, dict) and "tz" in v}

    lines = [build_train_diagnostics(metrics, None), ""]
    if not trades:
        lines.append("No trades were produced on the train split.")
        return "\n".join(lines)

    rows = []
    for t in trades:
        entry = _to_utc(_get(t, "entry_time"))
        exit_ = _to_utc(_get(t, "exit_time", entry))
        costs = sum(float(_get(t, k, 0.0) or 0.0)
                    for k in ("spread_cost", "slippage_cost", "commission", "swap_cost"))
        net = float(_get(t, "net_pnl", 0.0))
        rows.append({
            "entry": entry, "exit": exit_, "net_pnl": net,
            "gross_pnl": float(_get(t, "gross_pnl", net + costs)), "costs": costs,
            "direction": str(_get(t, "direction", "LONG")).upper(),
            "exit_reason": str(_get(t, "exit_reason", "unknown")),
            "bars_held": int(_get(t, "bars_held", 0) or 0),
            "mae": float(_get(t, "mae", 0.0) or 0.0), "mfe": float(_get(t, "mfe", 0.0) or 0.0),
        })
    df = pd.DataFrame(rows).sort_values("exit").reset_index(drop=True)
    df["year"] = df["entry"].dt.year
    df["month"] = df["entry"].dt.month.map(lambda m: _MONTHS[m - 1])
    df["weekday"] = df["entry"].dt.weekday.map(lambda d: _WEEKDAYS[d])
    df["hour"] = df["entry"].dt.hour
    df["session"] = df["entry"].map(lambda ts: classify_session(ts, sessions))

    lines += _group_table("By Year", df, "year")
    lines += _group_table("By Month (all years)", df, "month", _MONTHS)
    lines += _group_table("By Session", df, "session")
    lines += _group_table("By Weekday", df, "weekday", _WEEKDAYS)

    by_hour = df.groupby("hour")["net_pnl"].sum()
    lines += ["#### By Hour (UTC) net PnL",
              " | ".join(f"{h:02d}h:{v:+,.0f}" for h, v in by_hour.items()), ""]

    ex = df.groupby("exit_reason")["net_pnl"].agg(["count", "sum"])
    lines += ["#### Exit Reason Breakdown",
              ", ".join(f"{k}: {int(r['count'])} trades / {r['sum']:+,.0f}$" for k, r in ex.iterrows()), ""]

    win, loss = df[df.net_pnl > 0], df[df.net_pnl <= 0]
    lines += ["#### MAE / MFE ($ per trade)",
              f"- MAE median {df.mae.median():.1f} (winners {win.mae.median() if len(win) else 0:.1f}, "
              f"losers {loss.mae.median() if len(loss) else 0:.1f})",
              f"- MFE median {df.mfe.median():.1f} (winners {win.mfe.median() if len(win) else 0:.1f}, "
              f"losers {loss.mfe.median() if len(loss) else 0:.1f})", ""]

    q = df.bars_held.quantile([0.25, 0.5, 0.75])
    lines += ["#### Holding Time (bars)",
              f"- p25 {q[0.25]:.0f}, median {q[0.5]:.0f}, p75 {q[0.75]:.0f}, max {df.bars_held.max()}; "
              f"winners median {win.bars_held.median() if len(win) else 0:.0f}, "
              f"losers median {loss.bars_held.median() if len(loss) else 0:.0f}", ""]

    lines += ["#### Long vs Short"]
    for d in ("LONG", "SHORT"):
        sub = df[df.direction == d]
        if len(sub):
            lines.append(f"- {d.title()}: {len(sub)} trades, net {sub.net_pnl.sum():+,.0f}$, "
                         f"win {(sub.net_pnl > 0).mean() * 100:.0f}%")
    lines.append("")

    gross_abs = df.gross_pnl.clip(lower=0).sum()
    lines += ["#### Cost Impact",
              f"- Gross {df.gross_pnl.sum():+,.0f}$, costs {df.costs.sum():,.0f}$, net {df.net_pnl.sum():+,.0f}$, "
              f"costs / gross profit {df.costs.sum() / gross_abs:.0%}" if gross_abs > 0 else
              f"- Costs {df.costs.sum():,.0f}$ with no gross profit", ""]

    streak = best = 0
    for v in df.net_pnl:
        streak = streak + 1 if v <= 0 else 0
        best = max(best, streak)
    lines += ["#### Losing Streak", f"- Longest run of consecutive losing trades: {best}", ""]

    equity = df.net_pnl.cumsum()
    dd = equity - equity.cummax()
    trough = int(dd.idxmin())
    peak = int(equity.iloc[: trough + 1].idxmax()) if trough > 0 else 0
    lines += ["#### Worst Drawdown Period (closed-trade equity)",
              f"- {dd.min():,.0f}$ from {df.exit.iloc[peak]:%Y-%m-%d} to {df.exit.iloc[trough]:%Y-%m-%d}", ""]

    weak = []
    sess_net = df.groupby("session")["net_pnl"].sum()
    for s, v in sess_net.items():
        if v < 0:
            weak.append(f"{s.replace('_', ' ').title()} session is net negative ({v:,.0f}$, "
                        f"{int((df.session == s).sum())} trades); consider a session filter.")
    for d in ("LONG", "SHORT"):
        v = df.loc[df.direction == d, "net_pnl"].sum()
        if (df.direction == d).any() and v < 0:
            weak.append(f"{d.title()} side is net negative ({v:,.0f}$).")
    if weak:
        lines += ["#### Diagnosed Weaknesses"] + [f"- {w}" for w in weak]

    return "\n".join(lines)
