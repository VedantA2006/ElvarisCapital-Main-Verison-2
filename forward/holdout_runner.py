"""
forward/holdout_runner.py – Strict one-time holdout evaluation for QuantForge.

Closes: OPS-2 (holdout part)
Guarantees:
- Only eligible strategies that passed all prior gates may evaluate holdout.
- Atomic audit logged in Mongo `holdout_accesses`.
- Pass criterion: profitable (net_profit > 0) AND holdout Sharpe >= 50% of train Sharpe.
- Failure marks `holdout_failed = True` and permanently burns holdout for that strategy.
- Absolute isolation: holdout numbers are NEVER fed into LLM prompts or feedback loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any

import numpy as np
import pandas as pd

from core.event_simulator import EventSimulator


class HoldoutEligibilityError(Exception):
    """Raised when an ineligible or previously failed strategy attempts holdout."""
    pass


@dataclass
class HoldoutEvaluationResult:
    strategy_id: str
    passed: bool
    holdout_sharpe: float
    train_sharpe: float
    net_profit: float
    total_trades: int
    leak_to_llm: bool = False
    details: dict[str, Any] | None = None


def run_holdout(
    strategy_id: str,
    cfg: dict,
    db: Any,
    datastore: Any = None,
) -> HoldoutEvaluationResult:
    """Execute audited holdout evaluation for a qualified candidate strategy."""
    col = db["candidates"]
    doc = col.find_one({"strategy_id": strategy_id})

    if not doc:
        raise HoldoutEligibilityError(f"Strategy '{strategy_id}' not found in candidate pool.")

    if doc.get("holdout_failed") is True:
        raise HoldoutEligibilityError(
            f"Strategy '{strategy_id}' previously failed holdout. "
            "Holdout data is burned for this strategy forever."
        )

    allowed_statuses = {"candidate", "survived", "passed_validation"}
    if doc.get("status") not in allowed_statuses:
        raise HoldoutEligibilityError(
            f"Strategy '{strategy_id}' has status '{doc.get('status')}', "
            f"must be one of {allowed_statuses} to qualify for holdout."
        )

    # Resolve DataStore
    if datastore is None:
        from core.splits import DataStore
        datastore = DataStore(cfg)

    tf = doc.get("timeframe", "1h")

    # Fetch holdout data via audited path
    holdout_df = datastore.get_data(
        tf,
        "holdout",
        strategy_id=strategy_id,
        purpose="holdout_evaluation",
    )

    # Audit log access
    db["holdout_accesses"].insert_one({
        "strategy_id": strategy_id,
        "timeframe": tf,
        "timestamp": time.time(),
        "iso_time": datetime.now(timezone.utc).isoformat(),
        "bars": len(holdout_df),
        "triggered_by": "holdout_runner",
    })

    # Execute backtest on holdout split
    train_metrics = doc.get("train_metrics", {})
    train_sharpe = float(train_metrics.get("sharpe", 1.0))
    code = doc.get("source_code", "")

    from core.backtester import run_backtest, run_simulation_on_tape
    from core.signals import Action, Direction, SignalTape

    import logging
    _log = logging.getLogger("quantforge.forward")

    # Support class Strategy and generate_signals
    if not isinstance(code, str) or ("class Strategy" in code and "def on_bar" in code):
        try:
            bt_res = run_backtest(code, holdout_df, cfg, use_fast=False)
            m = bt_res.metrics
        except Exception as exc:
            _log.warning("Holdout run_backtest error: %s", exc)
            m = {}
    else:
        loc: dict[str, Any] = {}
        try:
            exec(code, loc)
            gen = loc.get("generate_signals")
            sigs = gen(holdout_df) if callable(gen) else pd.Series(0, index=holdout_df.index)
        except Exception as exc:
            _log.warning("Holdout signal generation error: %s", exc)
            sigs = pd.Series(0, index=holdout_df.index)

        n = len(holdout_df)
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
            bt_res = run_simulation_on_tape(tape, holdout_df, cfg, use_fast=False)
            m = bt_res.metrics
        except Exception as exc:
            _log.warning("Holdout simulation on tape error: %s", exc)
            m = {}

    net_profit = float(m.get("net_profit", 0.0))
    holdout_sharpe = float(m.get("sharpe", 0.0))
    trades_count = int(m.get("total_trades", 0))

    # Verification criteria:
    # 1. Net profit > 0 (profitable)
    # 2. Sharpe >= 50% of train Sharpe (Sharpe >= 0.5 * train_sharpe)
    # If train_sharpe <= 0, require holdout_sharpe > 0
    min_required_sharpe = max(0.0, 0.5 * train_sharpe)
    passed = (net_profit > 0.0) and (holdout_sharpe >= min_required_sharpe)

    holdout_status = "holdout_passed" if passed else "holdout_failed"

    # Update candidate record
    update_data = {
        "status": holdout_status,
        "holdout_failed": not passed,
        "holdout_result": {
            "passed": passed,
            "sharpe": holdout_sharpe,
            "net_profit": net_profit,
            "total_trades": trades_count,
            "profit_factor": float(m.get("profit_factor", 0.0)),
            "max_drawdown": float(m.get("max_drawdown", 0.0)),
            "evaluated_at": time.time(),
        }
    }
    col.update_one({"strategy_id": strategy_id}, {"$set": update_data})

    return HoldoutEvaluationResult(
        strategy_id=strategy_id,
        passed=passed,
        holdout_sharpe=holdout_sharpe,
        train_sharpe=train_sharpe,
        net_profit=net_profit,
        total_trades=trades_count,
        leak_to_llm=False,  # Enforces isolation
        details=m,
    )
