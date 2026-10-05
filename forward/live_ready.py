"""
forward/live_ready.py – Non-negotiable LIVE-READY gate verification.

Closes: OPS-3
Guarantees:
- Hard-coded minimums in code: MIN_FORWARD_DAYS = 60, MIN_FORWARD_TRADES = 50.
- Config or runtime overrides CANNOT lower these floors.
- Any attempt to label a strategy 'live_ready' before satisfying these floors raises CandidateUnprovenError.
- Until all criteria pass, strategy is explicitly labeled "CANDIDATE (unproven)" or "FORWARD TESTING".
"""

from __future__ import annotations

import time
from typing import Any

# Non-negotiable hard floors defined in code
MIN_FORWARD_DAYS: int = 60
MIN_FORWARD_TRADES: int = 50


class CandidateUnprovenError(Exception):
    """Raised when an unproven strategy attempts premature live promotion."""
    pass


def evaluate_live_ready(
    strategy_id: str,
    db: Any,
    cfg: dict | None = None,
) -> dict[str, Any]:
    """Verify that a candidate satisfies the non-negotiable LIVE-READY gate."""
    col = db["candidates"]
    doc = col.find_one({"strategy_id": strategy_id})

    if not doc:
        raise CandidateUnprovenError(f"Strategy '{strategy_id}' not found in candidate pool.")

    fwd = doc.get("forward_metrics", {})
    days = int(fwd.get("calendar_days", 0))
    trades = int(fwd.get("total_trades", 0))
    is_degraded = bool(fwd.get("is_degraded", False))
    pf = float(fwd.get("profit_factor", 0.0))
    sharpe = float(fwd.get("sharpe", 0.0))

    # Hard-coded floor validation (config is intentionally ignored if lower)
    if days < MIN_FORWARD_DAYS:
        raise CandidateUnprovenError(
            f"Strategy '{strategy_id}' cannot be marked live_ready: forward test has run "
            f"for {days} calendar days, but non-negotiable requirement is >= {MIN_FORWARD_DAYS} calendar days."
        )

    if trades < MIN_FORWARD_TRADES:
        raise CandidateUnprovenError(
            f"Strategy '{strategy_id}' cannot be marked live_ready: strategy has executed "
            f"{trades} forward trades, but non-negotiable requirement is >= {MIN_FORWARD_TRADES} trades."
        )

    if is_degraded:
        raise CandidateUnprovenError(
            f"Strategy '{strategy_id}' cannot be marked live_ready: marked DEGRADED by decay monitor."
        )

    if pf <= 1.0 or sharpe <= 0.0:
        raise CandidateUnprovenError(
            f"Strategy '{strategy_id}' cannot be marked live_ready: forward performance is unprofitable "
            f"(Profit Factor: {pf:.2f}, Sharpe: {sharpe:.2f})."
        )

    # Successfully passes non-negotiable LIVE-READY gate
    now = time.time()
    col.update_one(
        {"strategy_id": strategy_id},
        {"$set": {"status": "live_ready", "live_ready_at": now}},
    )

    return {
        "strategy_id": strategy_id,
        "status": "live_ready",
        "calendar_days": days,
        "total_trades": trades,
        "sharpe": sharpe,
        "profit_factor": pf,
        "promoted_at": now,
    }
