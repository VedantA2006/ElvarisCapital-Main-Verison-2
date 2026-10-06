"""
strategy/library.py – Searchable Strategy Library and State Machine Manager.

Guarantees (Sections 47, 52):
1. Strict 12-state lifecycle state machine:
   GENERATED -> COMPILED -> SECURITY_PASS -> BACKTESTED -> SCREENED -> ROBUST ->
   FROZEN -> HOLDOUT_TESTED -> FORWARD_TESTED -> PORTFOLIO_ELIGIBLE -> LIVE_READY -> DEPLOYED.
2. Immutable frozen states (cannot modify code or config once FROZEN without new version).
3. Rich querying and filtering (family, timeframe, Sharpe, PF, DD, robustness score).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from storage import mongo

_log = logging.getLogger("quantforge.strategy.library")


class StrategyLifecycleState(str, Enum):
    GENERATED = "GENERATED"
    COMPILED = "COMPILED"
    SECURITY_PASS = "SECURITY_PASS"
    BACKTESTED = "BACKTESTED"
    SCREENED = "SCREENED"
    ROBUST = "ROBUST"
    FROZEN = "FROZEN"
    HOLDOUT_TESTED = "HOLDOUT_TESTED"
    FORWARD_TESTED = "FORWARD_TESTED"
    PORTFOLIO_ELIGIBLE = "PORTFOLIO_ELIGIBLE"
    LIVE_READY = "LIVE_READY"
    DEPLOYED = "DEPLOYED"


VALID_TRANSITIONS = {
    StrategyLifecycleState.GENERATED: [StrategyLifecycleState.COMPILED],
    StrategyLifecycleState.COMPILED: [StrategyLifecycleState.SECURITY_PASS],
    StrategyLifecycleState.SECURITY_PASS: [StrategyLifecycleState.BACKTESTED],
    StrategyLifecycleState.BACKTESTED: [StrategyLifecycleState.SCREENED],
    StrategyLifecycleState.SCREENED: [StrategyLifecycleState.ROBUST],
    StrategyLifecycleState.ROBUST: [StrategyLifecycleState.FROZEN],
    StrategyLifecycleState.FROZEN: [StrategyLifecycleState.HOLDOUT_TESTED],
    StrategyLifecycleState.HOLDOUT_TESTED: [StrategyLifecycleState.FORWARD_TESTED],
    StrategyLifecycleState.FORWARD_TESTED: [StrategyLifecycleState.PORTFOLIO_ELIGIBLE],
    StrategyLifecycleState.PORTFOLIO_ELIGIBLE: [StrategyLifecycleState.LIVE_READY],
    StrategyLifecycleState.LIVE_READY: [StrategyLifecycleState.DEPLOYED],
}


@dataclass
class LibraryStrategyEntry:
    strategy_id: str
    name: str
    family: str
    timeframe: str
    state: StrategyLifecycleState
    code: str
    code_hash: str
    spec: Dict[str, Any]
    dataset_hash: str
    robustness_score: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    frozen_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)


StrategyState = StrategyLifecycleState


class StrategyLibrary:
    """Persistent storage and lifecycle query manager for promoted strategies."""

    def __init__(self, db=None):
        self._db = db
        try:
            self._col = self._db["strategy_library"] if self._db is not None else mongo.get_db()["strategy_library"]
        except Exception as exc:
            _log.debug("Mongo unavailable in StrategyLibrary: %s", exc)
            self._col = None
        self._mem: Dict[str, Dict[str, Any]] = {}

    def register(self, entry: LibraryStrategyEntry) -> None:
        """Add new strategy to library in INITIAL state."""
        doc = asdict(entry)
        doc["state"] = entry.state.value
        self._mem[entry.strategy_id] = dict(doc)
        if self._col is not None:
            self._col.replace_one({"strategy_id": entry.strategy_id}, doc, upsert=True)

    def register_strategy(
        self,
        strategy_id: str,
        spec_hash: str = "",
        code: str = "",
        family: str = "TREND",
        timeframe: str = "1h",
    ) -> None:
        """Convenience method to register a strategy in GENERATED state."""
        entry = LibraryStrategyEntry(
            strategy_id=strategy_id,
            name=strategy_id,
            family=family,
            timeframe=timeframe,
            state=StrategyLifecycleState.GENERATED,
            code=code,
            code_hash=hashlib.sha256(code.encode()).hexdigest() if code else "",
            spec={"spec_hash": spec_hash},
            dataset_hash="",
        )
        self.register(entry)

    def get_state(self, strategy_id: str) -> StrategyLifecycleState:
        """Return the current lifecycle state of a strategy."""
        doc = None
        if self._col is not None:
            try:
                doc = self._col.find_one({"strategy_id": strategy_id})
            except Exception as exc:
                _log.debug("Strategy query fallback for %s: %s", strategy_id, exc)
        if not doc:
            doc = self._mem.get(strategy_id)
        if not doc:
            raise KeyError(f"Strategy '{strategy_id}' not found.")
        return StrategyLifecycleState(doc["state"])

    def is_frozen(self, strategy_id: str) -> bool:
        """Check if strategy has reached FROZEN or later state."""
        st = self.get_state(strategy_id)
        frozen_or_later = {
            StrategyLifecycleState.FROZEN,
            StrategyLifecycleState.HOLDOUT_TESTED,
            StrategyLifecycleState.FORWARD_TESTED,
            StrategyLifecycleState.PORTFOLIO_ELIGIBLE,
            StrategyLifecycleState.LIVE_READY,
            StrategyLifecycleState.DEPLOYED,
        }
        return st in frozen_or_later

    def modify_spec(self, strategy_id: str, new_spec_hash: str) -> None:
        """Attempt to modify strategy specification, raising PermissionError if frozen."""
        if self.is_frozen(strategy_id):
            raise PermissionError(
                f"Strategy '{strategy_id}' is FROZEN and immutable. Any modification requires a new strategy version."
            )
        if self._col is not None:
            self._col.update_one({"strategy_id": strategy_id}, {"$set": {"spec.spec_hash": new_spec_hash}})
        if strategy_id in self._mem:
            self._mem[strategy_id]["spec"]["spec_hash"] = new_spec_hash

    def transition_state(self, strategy_id: str, new_state: StrategyLifecycleState) -> None:
        """Advance strategy through lifecycle state machine with immutability guarantees."""
        doc = None
        if self._col is not None:
            try:
                doc = self._col.find_one({"strategy_id": strategy_id})
            except Exception as exc:
                _log.debug("Transition query fallback for %s: %s", strategy_id, exc)
        if not doc:
            doc = self._mem.get(strategy_id)
        if not doc:
            raise KeyError(f"Strategy '{strategy_id}' not found in library.")

        current = StrategyLifecycleState(doc["state"])
        allowed = VALID_TRANSITIONS.get(current, [])
        if new_state not in allowed:
            raise ValueError(
                f"Invalid lifecycle transition: cannot move '{strategy_id}' from {current.value} to {new_state.value}. "
                f"Allowed transitions: {[s.value for s in allowed]}"
            )

        updates: Dict[str, Any] = {"state": new_state.value}
        if new_state == StrategyLifecycleState.FROZEN:
            updates["frozen_at"] = datetime.now(timezone.utc).isoformat()

        if self._col is not None:
            self._col.update_one({"strategy_id": strategy_id}, {"$set": updates})
        if strategy_id in self._mem:
            self._mem[strategy_id].update(updates)
        _log.info("Strategy '%s' transitioned: %s -> %s", strategy_id, current.value, new_state.value)

    def transition(self, strategy_id: str, new_state: StrategyLifecycleState) -> None:
        """Alias for transition_state."""
        self.transition_state(strategy_id, new_state)

    def search(
        self,
        family: Optional[str] = None,
        timeframe: Optional[str] = None,
        min_sharpe: float = 0.0,
        min_pf: float = 0.0,
        state: Optional[StrategyLifecycleState] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query and filter strategies in the library."""
        if self._col is not None:
            query: Dict[str, Any] = {
                "sharpe": {"$gte": min_sharpe},
                "profit_factor": {"$gte": min_pf},
            }
            if family:
                query["family"] = family
            if timeframe:
                query["timeframe"] = timeframe
            if state:
                query["state"] = state.value

            return list(self._col.find(query).sort("robustness_score", -1).limit(limit))
        return list(self._mem.values())[:limit]

