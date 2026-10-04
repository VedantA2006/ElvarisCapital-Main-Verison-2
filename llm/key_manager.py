"""
llm/key_manager.py – API Key lifecycle manager with two-key failover and persistence.

Manages keys by label (e.g. 'key_1', 'key_2'). Real API secrets are NEVER stored
in Mongo, logs, or exception messages. State is persisted in Mongo `key_status`
so cooldowns and budget tracking survive process restarts.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from storage import mongo

_log = logging.getLogger("quantforge.llm.key_manager")


class KeyStatus(str, Enum):
    ACTIVE = "active"
    COOLING_DOWN = "cooling_down"
    INVALID = "invalid"
    BUDGET_EXHAUSTED = "budget_exhausted"


class AllKeysUnavailableError(RuntimeError):
    def __init__(self, message: str, reset_at: datetime | None = None):
        super().__init__(message)
        self.reset_at = reset_at


def _ensure_utc(dt: Any) -> datetime | None:
    """Ensure datetime is timezone-aware in UTC."""
    if dt is None:
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


class KeyManager:
    """Manages API keys with persistent state, cooldowns, and round-robin / failover."""

    def __init__(self, cfg: dict, db=None):
        self._cfg = cfg.get("llm", {})
        self._db = db if db is not None else mongo.get_db()
        self._col = self._db["key_status"]
        self._default_cooldown = self._cfg.get("cooldown_default_seconds", 60)
        self._alternate_keys = self._cfg.get("alternate_keys", False)
        self._max_calls_per_day = self._cfg.get("max_calls_per_day", 500)
        self._max_tokens_per_day = self._cfg.get("max_tokens_per_day", 2_000_000)

        # Discover keys from environment
        self._keys: dict[str, str] = {}  # label -> secret
        self._order: list[str] = []
        for i in range(1, 10):
            secret = os.environ.get(f"LLM_API_KEY_{i}", "").strip()
            if secret:
                label = f"key_{i}"
                self._keys[label] = secret
                self._order.append(label)

        if not self._keys:
            # Fallback to OPENAI_API_KEY or generic LLM_API_KEY
            fallback = os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("LLM_API_KEY", "").strip()
            if fallback:
                self._keys["key_1"] = fallback
                self._order.append("key_1")
            else:
                raise RuntimeError("No LLM_API_KEY_* found in environment.")

        self._current_idx = 0
        self._sync_db_state()

    def _sync_db_state(self) -> None:
        """Initialize or reload persisted key state in MongoDB."""
        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")

        for label in self._order:
            existing = self._col.find_one({"label": label})
            if existing is None:
                doc = {
                    "label": label,
                    "status": KeyStatus.ACTIVE.value,
                    "reset_at": None,
                    "calls_today": 0,
                    "tokens_today": 0,
                    "date": today_str,
                    "consecutive_errors": 0,
                    "last_used_at": None,
                    "updated_at": now,
                }
                self._col.insert_one(doc)
            else:
                # If date rolled over, reset today's counts
                if existing.get("date") != today_str:
                    update: dict[str, Any] = {
                        "calls_today": 0,
                        "tokens_today": 0,
                        "date": today_str,
                        "updated_at": now,
                    }
                    if existing.get("status") == KeyStatus.BUDGET_EXHAUSTED.value:
                        update["status"] = KeyStatus.ACTIVE.value
                        update["reset_at"] = None
                    self._col.update_one({"label": label}, {"$set": update})

    def redact(self, text: str) -> str:
        """Replace all known secret keys with [REDACTED]."""
        if not text:
            return ""
        out = text
        for secret in self._keys.values():
            if secret and len(secret) >= 6:
                out = out.replace(secret, "[REDACTED]")
        return out

    def get_key_status(self, label: str) -> dict[str, Any]:
        """Fetch current status document for key label."""
        doc = self._col.find_one({"label": label})
        if not doc:
            return {"label": label, "status": KeyStatus.INVALID.value, "reset_at": None}
        if doc.get("reset_at"):
            doc["reset_at"] = _ensure_utc(doc["reset_at"])
        return doc

    def get_active_key(self, now: datetime | None = None) -> tuple[str, str]:
        """Return the next (label, secret) ready for use.
        
        If all keys are unavailable (cooling down or budget exhausted),
        raises AllKeysUnavailableError with the earliest reset time.
        """
        current_time = _ensure_utc(now) or datetime.now(timezone.utc)
        today_str = current_time.strftime("%Y-%m-%d")

        # Refresh candidate statuses (clearing expired cooldowns)
        available_labels: list[str] = []
        earliest_reset: datetime | None = None

        for label in self._order:
            doc = self._col.find_one({"label": label})
            if not doc:
                continue

            status = doc.get("status", KeyStatus.ACTIVE.value)
            reset_at = _ensure_utc(doc.get("reset_at"))

            # Check for date rollover
            if doc.get("date") != today_str:
                self._col.update_one(
                    {"label": label},
                    {"$set": {"calls_today": 0, "tokens_today": 0, "date": today_str, "status": KeyStatus.ACTIVE.value, "reset_at": None}}
                )
                status = KeyStatus.ACTIVE.value
                reset_at = None

            # Check expired cooldown
            if status == KeyStatus.COOLING_DOWN.value and reset_at:
                if current_time >= reset_at:
                    self._col.update_one(
                        {"label": label},
                        {"$set": {"status": KeyStatus.ACTIVE.value, "reset_at": None, "updated_at": current_time}}
                    )
                    status = KeyStatus.ACTIVE.value
                    reset_at = None

            if status == KeyStatus.ACTIVE.value:
                available_labels.append(label)
            elif status in (KeyStatus.COOLING_DOWN.value, KeyStatus.BUDGET_EXHAUSTED.value):
                if reset_at:
                    if earliest_reset is None or reset_at < earliest_reset:
                        earliest_reset = reset_at

        if not available_labels:
            msg = f"All {len(self._order)} API keys are unavailable."
            _log.warning(msg)
            raise AllKeysUnavailableError(msg, reset_at=earliest_reset)

        # Select key according to alternate_keys policy
        if self._alternate_keys:
            chosen = available_labels[self._current_idx % len(available_labels)]
            self._current_idx = (self._current_idx + 1) % len(available_labels)
        else:
            chosen = available_labels[0]

        return chosen, self._keys[chosen]

    def mark_rate_limited(self, label: str, retry_after_s: float | None = None, now: datetime | None = None) -> datetime:
        """Mark a key as cooling down until now + delay."""
        current_time = _ensure_utc(now) or datetime.now(timezone.utc)
        delay = retry_after_s if (retry_after_s and retry_after_s > 0) else self._default_cooldown
        reset_at = current_time + timedelta(seconds=delay)

        self._col.update_one(
            {"label": label},
            {
                "$set": {
                    "status": KeyStatus.COOLING_DOWN.value,
                    "reset_at": reset_at,
                    "updated_at": current_time,
                },
                "$inc": {"consecutive_errors": 1}
            }
        )
        _log.warning("Key %s rate limited. Cooling down until %s (%ds)", label, reset_at.isoformat(), delay)
        return reset_at

    def mark_invalid(self, label: str, reason: str = "") -> None:
        """Mark a key as permanently invalid (e.g. 401 Unauthorized)."""
        now = datetime.now(timezone.utc)
        self._col.update_one(
            {"label": label},
            {
                "$set": {
                    "status": KeyStatus.INVALID.value,
                    "invalid_reason": reason[:200],
                    "updated_at": now,
                }
            }
        )
        _log.error("Key %s marked INVALID: %s", label, reason)

    def mark_budget_exhausted(self, label: str, now: datetime | None = None) -> datetime:
        """Mark key budget exhausted until next midnight UTC."""
        current_time = _ensure_utc(now) or datetime.now(timezone.utc)
        tomorrow = (current_time + timedelta(days=1)).date()
        reset_at = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0, tzinfo=timezone.utc)

        self._col.update_one(
            {"label": label},
            {
                "$set": {
                    "status": KeyStatus.BUDGET_EXHAUSTED.value,
                    "reset_at": reset_at,
                    "updated_at": current_time,
                }
            }
        )
        _log.warning("Key %s daily budget reached. Inactive until %s", label, reset_at.isoformat())
        return reset_at

    def record_success(self, label: str, tokens: int, now: datetime | None = None) -> None:
        """Record successful call: increments counters and checks daily budget."""
        current_time = _ensure_utc(now) or datetime.now(timezone.utc)
        today_str = current_time.strftime("%Y-%m-%d")

        doc = self._col.find_one_and_update(
            {"label": label},
            {
                "$inc": {"calls_today": 1, "tokens_today": max(0, tokens)},
                "$set": {
                    "last_used_at": current_time,
                    "updated_at": current_time,
                    "consecutive_errors": 0,
                    "date": today_str,
                }
            },
            return_document=mongo.ReturnDocument.AFTER,
        )

        if doc:
            calls = doc.get("calls_today", 0)
            toks = doc.get("tokens_today", 0)
            if calls >= self._max_calls_per_day or toks >= self._max_tokens_per_day:
                self.mark_budget_exhausted(label, now=current_time)

    def get_next_available_time(self, now: datetime | None = None) -> datetime | None:
        """Calculate the earliest reset time across cooling keys."""
        current_time = _ensure_utc(now) or datetime.now(timezone.utc)
        earliest = None
        for label in self._order:
            doc = self._col.find_one({"label": label})
            if not doc:
                continue
            r = _ensure_utc(doc.get("reset_at"))
            if r and (earliest is None or r < earliest):
                earliest = r
        return earliest
