"""
engine/state.py – State machine, single-instance lease lock, and trial counter idempotency.

Closes: ORCH-1, ORCH-2 (Phase F10).
"""

from __future__ import annotations

from enum import Enum
import time
from typing import Any


class EngineState(str, Enum):
    """The 10 formal states of the QuantForge Autonomous Engine (F10.1)."""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    WAITING_FOR_QUOTA = "WAITING_FOR_QUOTA"
    BUDGET_PAUSED = "BUDGET_PAUSED"
    ERROR_BACKOFF = "ERROR_BACKOFF"
    WAITING_FOR_DB = "WAITING_FOR_DB"
    BLOCKED = "BLOCKED"


class LockAcquisitionError(RuntimeError):
    """Raised when another engine worker already holds an active lease lock."""


class EngineStateManager:
    """Manages engine state persistence, heartbeats, single-instance locks, and trial counting."""

    def __init__(self, db: Any, worker_id: str = "worker_default"):
        self.db = db
        self.worker_id = worker_id
        self._col_state = self.db["engine_state"]
        self._col_lease = self.db["engine_lease"]
        self._col_idempotency = self.db["trial_idempotency"]
        self._col_counters = self.db["engine_counters"]

    # ── Single-Instance Lock ────────────────────────────────────────────────

    def acquire_lock(self, ttl_seconds: int = 60) -> bool:
        """Acquire single-instance lock via TTL lease.
        Fails fast if another active worker holds the lease.
        """
        now = time.time()
        lease = self._col_lease.find_one({"_id": "singleton_worker"})

        if lease:
            expires_at = lease.get("expires_at", 0)
            holder = lease.get("worker_id")
            if expires_at > now and holder != self.worker_id:
                raise LockAcquisitionError(
                    f"Worker '{holder}' currently holds the engine lock (expires in {expires_at - now:.1f}s)."
                )

        # Upsert our lease
        self._col_lease.update_one(
            {"_id": "singleton_worker"},
            {"$set": {"worker_id": self.worker_id, "acquired_at": now, "expires_at": now + ttl_seconds}},
            upsert=True,
        )
        return True

    def renew_lock(self, ttl_seconds: int = 60) -> bool:
        """Extend lock expiration while worker is running."""
        now = time.time()
        res = self._col_lease.update_one(
            {"_id": "singleton_worker", "worker_id": self.worker_id},
            {"$set": {"expires_at": now + ttl_seconds}},
        )
        return bool(res.matched_count > 0)

    def release_lock(self) -> None:
        """Release the worker lock cleanly."""
        self._col_lease.delete_one({"_id": "singleton_worker", "worker_id": self.worker_id})

    # ── State and Heartbeat ─────────────────────────────────────────────────

    def set_state(self, state: EngineState, reason: str = "") -> None:
        """Set and persist engine state."""
        now = time.time()
        self._col_state.update_one(
            {"_id": "current_state"},
            {
                "$set": {
                    "state": state.value,
                    "reason": reason,
                    "worker_id": self.worker_id,
                    "updated_at": now,
                    "last_heartbeat": now,
                }
            },
            upsert=True,
        )

    def get_state(self) -> EngineState:
        """Get the current persisted state."""
        doc = self._col_state.find_one({"_id": "current_state"})
        if not doc or "state" not in doc:
            return EngineState.STOPPED
        return EngineState(doc["state"])

    def get_state_reason(self) -> str:
        """Get the reason attached to current state."""
        doc = self._col_state.find_one({"_id": "current_state"})
        return str(doc.get("reason", "")) if doc else ""

    def heartbeat(self) -> None:
        """Write heartbeat timestamp."""
        now = time.time()
        self._col_state.update_one(
            {"_id": "current_state"},
            {"$set": {"last_heartbeat": now, "worker_id": self.worker_id}},
            upsert=True,
        )

    def is_stale(self, threshold_seconds: float = 30.0) -> bool:
        """Check if engine heartbeat is stale (> threshold_seconds without heartbeat)."""
        doc = self._col_state.find_one({"_id": "current_state"})
        if not doc or "last_heartbeat" not in doc:
            return True
        last_hb = float(doc.get("last_heartbeat", 0))
        return (time.time() - last_hb) > threshold_seconds

    # ── Trial Counter Idempotency ───────────────────────────────────────────

    def increment_trial_once(self, cycle_id: str, version: int = 1) -> bool:
        """Increment trial counter exactly once per (cycle_id, version).
        Returns True if newly recorded, False if already counted (idempotent on resume).
        """
        idempotency_key = f"{cycle_id}_v{version}"
        existing = self._col_idempotency.find_one({"_id": idempotency_key})
        if existing:
            return False

        # Record idempotency key
        self._col_idempotency.insert_one({
            "_id": idempotency_key,
            "cycle_id": cycle_id,
            "version": version,
            "recorded_at": time.time(),
        })

        # Atomically increment total trials counter
        self._col_counters.update_one(
            {"_id": "trial_counter"},
            {"$inc": {"count": 1}},
            upsert=True,
        )
        return True

    def get_trial_count(self) -> int:
        """Retrieve total trial count."""
        doc = self._col_counters.find_one({"_id": "trial_counter"})
        return int(doc.get("count", 0)) if doc else 0
