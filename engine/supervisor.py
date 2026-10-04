"""
engine/supervisor.py – Worker process supervision, lifecycle control, and crash recovery.

Closes: ORCH-2, ORCH-3 (Phase F10).
"""

from __future__ import annotations

import time
from typing import Any


class EngineSupervisor:
    """Controls the engine process lifecycle according to engine_control documents."""

    def __init__(self, db: Any, cfg: dict | None = None):
        self.db = db
        self.cfg = cfg or {}
        self._col_control = self.db["engine_control"]

    def set_desired_state(
        self,
        state: str,
        requested_by: str = "user",
        force_kill_at: float | None = None,
    ) -> None:
        """Store the desired engine state ('running', 'paused', 'stopped')."""
        valid_states = {"running", "paused", "stopped"}
        state_lower = state.strip().lower()
        if state_lower not in valid_states:
            raise ValueError(f"Invalid desired state '{state}'. Expected one of {valid_states}.")

        now = time.time()
        self._col_control.update_one(
            {"_id": "engine_control"},
            {
                "$set": {
                    "desired_state": state_lower,
                    "requested_by": requested_by,
                    "requested_at": now,
                    "force_kill_at": force_kill_at,
                }
            },
            upsert=True,
        )

    def get_desired_state(self) -> str:
        """Read the currently requested engine state."""
        doc = self._col_control.find_one({"_id": "engine_control"})
        if not doc or "desired_state" not in doc:
            return "stopped"
        return str(doc["desired_state"])
