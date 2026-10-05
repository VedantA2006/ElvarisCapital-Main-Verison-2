"""
dashboard/tui.py – Terminal User Interface (TUI) for QuantForge using Rich.

Mirrors the Overview page:
- Engine execution state, heartbeat, worker ID, and desired state
- Trial counters (total, candidates, lookahead rejections, robustness rejections)
- Gate funnel breakdown
- LLM panel (active key label, model, tokens used, failed calls)
- Top candidates leaderboard
- Non-blocking loop reading and displaying live engine status from MongoDB
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.config import mongo_db_name
from engine.supervisor import EngineSupervisor
from storage.mongo import get_db


def build_engine_panel(db: Any) -> Panel:
    state_doc = db["engine_state"].find_one({"_id": "current_state"}) or {}
    ctrl_doc = db["engine_control"].find_one({"_id": "desired_state"}) or {}

    st = state_doc.get("state", "STOPPED")
    color = "green" if st == "RUNNING" else ("yellow" if "PAUSE" in st or "WAIT" in st else "red")

    desired = ctrl_doc.get("desired_state", "stopped")
    last_hb = float(state_doc.get("last_heartbeat", 0))
    hb_age = round(time.time() - last_hb, 1) if last_hb > 0 else 999.9

    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="left")
    grid.add_column(justify="right")

    grid.add_row(f"[bold {color}]● {st}[/]", f"Desired: [bold]{desired}[/]")
    grid.add_row(f"Worker: {state_doc.get('worker_id', 'none')}", f"Heartbeat Age: {hb_age}s")
    grid.add_row(f"Reason: {state_doc.get('reason', 'None')}", f"Cycle: {state_doc.get('current_cycle_id', 'none')}")

    return Panel(grid, title="[bold]Engine State[/]", border_style=color)


def build_counters_panel(db: Any) -> Panel:
    trials_col = db["trials"]
    cand_col = db["candidates"]

    total = trials_col.count_documents({})
    cands = cand_col.count_documents({})

    lookahead_rej = trials_col.count_documents({"status": "rejected", "rejected_at": {"$in": ["delay", "determinism", "truncation"]}})
    robust_rej = trials_col.count_documents({"status": "rejected", "rejected_at": {"$in": ["monte_carlo", "walk_forward", "regime_and_year"]}})

    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column()
    grid.add_column(justify="right")

    grid.add_row("Total Ideated Trials:", f"[bold cyan]{total:,}[/]")
    grid.add_row("Promoted Candidates:", f"[bold green]{cands:,}[/]")
    grid.add_row("Lookahead Rejections:", f"[bold yellow]{lookahead_rej:,}[/]")
    grid.add_row("Robustness Rejections:", f"[bold red]{robust_rej:,}[/]")

    return Panel(grid, title="[bold]Trial Discovery Counters[/]", border_style="cyan")


def build_llm_panel(db: Any) -> Panel:
    usage_doc = db["llm_usage"].find_one({"_id": "global"}) or {}
    total_tokens = usage_doc.get("total_tokens", 0)
    total_calls = usage_doc.get("total_calls", 0)
    failed_calls = usage_doc.get("failed_calls", 0)

    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column()
    grid.add_column(justify="right")

    grid.add_row("Active Key Label:", "[bold blue]key_1[/]")
    grid.add_row("Total Tokens:", f"[bold]{total_tokens:,}[/]")
    grid.add_row("Total API Calls:", f"{total_calls:,}")
    grid.add_row("Failures / Retries:", f"[yellow]{failed_calls:,}[/]")

    return Panel(grid, title="[bold]LLM & Token Telemetry[/]", border_style="blue")


def build_leaderboard_table(db: Any) -> Table:
    table = Table(title="Top 5 Candidate Strategies", expand=True)
    table.add_column("Strategy ID", style="bold")
    table.add_column("Status", style="yellow")
    table.add_column("TF")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Sharpe", justify="right")
    table.add_column("PF", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Trades", justify="right")

    cands = list(db["candidates"].find(
        {},
        projection={"_id": 0, "strategy_id": 1, "timeframe": 1, "robustness_score": 1, "train_metrics": 1},
    ).sort("robustness_score", -1).limit(5))

    if not cands:
        table.add_row("No candidates discovered yet", "—", "—", "—", "—", "—", "—", "—")
        return table

    for c in cands:
        m = c.get("train_metrics", {})
        table.add_row(
            c.get("strategy_id", "strat"),
            "CANDIDATE",
            c.get("timeframe", "1h"),
            f"{c.get('robustness_score', 0):.1f}",
            f"{m.get('sharpe', 0):.2f}",
            f"{m.get('profit_factor', 0):.2f}",
            f"{m.get('max_drawdown', 0) * 100:.1f}%",
            str(m.get("total_trades", 0)),
        )

    return table


def render_tui_frame(console: Console, db: Any) -> None:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top_row", size=6),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )

    layout["header"].update(
        Panel(
            Text("QuantForge v2.0 Research Terminal Console", justify="center", style="bold white on blue"),
            border_style="blue",
        )
    )

    layout["top_row"].split_row(
        Layout(build_engine_panel(db)),
        Layout(build_counters_panel(db)),
        Layout(build_llm_panel(db)),
    )

    layout["body"].update(Panel(build_leaderboard_table(db), border_style="grey50"))

    layout["footer"].update(
        Panel(
            Text("Controls: [Ctrl+C] Exit TUI | Engine state is supervised autonomously in background", justify="center", style="dim"),
            border_style="grey37",
        )
    )

    console.clear()
    console.print(layout)


def run_tui(cfg: dict, refresh_seconds: float = 2.0) -> int:
    console = Console()
    db = get_db(mongo_db_name(cfg))
    console.print("[bold green]Starting QuantForge Terminal Dashboard...[/]")

    try:
        while True:
            render_tui_frame(console, db)
            time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]TUI exited.[/]")
        return 0
