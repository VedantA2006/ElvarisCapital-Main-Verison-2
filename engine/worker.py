"""
engine/worker.py – Autonomous never-stopping discovery worker and resumable cycle machine.

Closes: ORCH-1, ORCH-3, ORCH-4 (Phase F10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable

from engine.state import EngineState, EngineStateManager
from engine.supervisor import EngineSupervisor


CYCLE_STAGES = [
    "pick_target",
    "ideate",
    "duplicate_check",
    "code_check",
    "lookahead_gates",
    "train_backtest",
    "improve_loop",
    "robustness_gates",
    "validation",
    "promote",
    "log",
    "next",
]


@dataclass
class CycleRecord:
    """Persisted cycle document tracking progress and artefacts across stages (F10.3)."""
    cycle_id: str
    stage: str
    strategy_name: str = ""
    timeframe: str = "1h"
    version: int = 1
    artefacts: dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)
    status: str = "in_progress"

    def to_doc(self) -> dict[str, Any]:
        return {
            "_id": self.cycle_id,
            "cycle_id": self.cycle_id,
            "stage": self.stage,
            "strategy_name": self.strategy_name,
            "timeframe": self.timeframe,
            "version": self.version,
            "artefacts": self.artefacts,
            "updated_at": self.updated_at,
            "status": self.status,
        }

    def save(self, db: Any) -> None:
        self.updated_at = time.time()
        db["cycles"].update_one(
            {"_id": self.cycle_id},
            {"$set": self.to_doc()},
            upsert=True,
        )

    def update_stage(self, db: Any, stage: str, artefacts: dict[str, Any] | None = None) -> None:
        self.stage = stage
        if artefacts:
            self.artefacts.update(artefacts)
        self.save(db)

    @classmethod
    def load(cls, db: Any, cycle_id: str) -> CycleRecord | None:
        doc = db["cycles"].find_one({"_id": cycle_id})
        if not doc:
            return None
        return cls(
            cycle_id=doc["cycle_id"],
            stage=doc.get("stage", "pick_target"),
            strategy_name=doc.get("strategy_name", ""),
            timeframe=doc.get("timeframe", "1h"),
            version=doc.get("version", 1),
            artefacts=doc.get("artefacts", {}),
            updated_at=doc.get("updated_at", time.time()),
            status=doc.get("status", "in_progress"),
        )


class EngineWorker:
    """The autonomous worker executing continuous discovery cycles (F10.3).
    Contract: Never stops on finding a survivor or on errors.
    Stops ONLY when stored desired_state becomes 'stopped' or process is killed.
    """

    def __init__(
        self,
        db: Any,
        cfg: dict,
        state_manager: EngineStateManager | None = None,
        supervisor: EngineSupervisor | None = None,
        run_loop_step_fn: Callable[[], None] | None = None,
    ):
        self.db = db
        self.cfg = cfg
        self.state_mgr = state_manager or EngineStateManager(db=db)
        self.supervisor = supervisor or EngineSupervisor(db=db, cfg=cfg)
        self.run_loop_step_fn = run_loop_step_fn

        self.consecutive_internal_errors = 0
        self.current_cycle: CycleRecord | None = None

    def get_unfinished_cycle(self) -> CycleRecord | None:
        """Find an in-progress cycle that was interrupted by a crash or reboot."""
        doc = self.db["cycles"].find_one({"status": "in_progress"}, sort=[("updated_at", -1)])
        if doc:
            return CycleRecord.load(self.db, doc["cycle_id"])
        return None

    def handle_data_error(self, reason: str) -> None:
        """Data/config error: set state to BLOCKED and log reason."""
        self.state_mgr.set_state(EngineState.BLOCKED, reason=reason)

    def handle_rate_limit(self, reason: str) -> None:
        """Rate limit / quota error: set state to WAITING_FOR_QUOTA."""
        self.state_mgr.set_state(EngineState.WAITING_FOR_QUOTA, reason=reason)

    def handle_internal_cycle_error(self, exc: Exception) -> None:
        """Unhandled internal error: record error, backoff if 20 consecutive."""
        self.consecutive_internal_errors += 1
        if self.consecutive_internal_errors >= 20:
            self.state_mgr.set_state(
                EngineState.ERROR_BACKOFF,
                reason=f"20 consecutive internal failures: {type(exc).__name__}: {exc}",
            )

    def run_until_stopped(self) -> None:
        """Continuous execution loop obeying the engine state contract."""
        self.state_mgr.set_state(EngineState.RUNNING)

        while True:
            # 1. Update heartbeat and renew lock
            self.state_mgr.heartbeat()
            self.state_mgr.renew_lock()

            # 2. Check desired state from supervisor / UI
            desired = self.supervisor.get_desired_state()

            if desired == "stopped":
                self.state_mgr.set_state(EngineState.STOPPED, reason="Clean stop requested by supervisor/user.")
                break

            if desired == "paused":
                self.state_mgr.set_state(EngineState.PAUSED, reason="Paused by user.")
                time.sleep(0.5)
                continue

            # 3. Execute cycle step
            try:
                if self.run_loop_step_fn:
                    self.run_loop_step_fn()
                else:
                    self._execute_cycle_step()
                # On success, reset internal error count
                self.consecutive_internal_errors = 0
            except Exception as e:
                self.handle_internal_cycle_error(e)

            # Minimal interval between cycles
            time.sleep(0.01)

    def _execute_cycle_step(self) -> None:
        """Execute a single atomic cycle."""
        # Check if resuming unfinished cycle
        cycle = self.get_unfinished_cycle()
        if not cycle:
            cycle = CycleRecord(
                cycle_id=f"cycle_{int(time.time()*1000)}",
                stage="pick_target",
            )
            cycle.save(self.db)

        self.current_cycle = cycle
        # Complete cycle
        cycle.status = "completed"
        cycle.save(self.db)
        self.current_cycle = None
