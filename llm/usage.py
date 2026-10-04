"""
llm/usage.py – Token tracking, cost estimation, and call telemetry for LLM operations.

Every LLM request inserts a complete audit record into Mongo `llm_calls` and
atomically updates running aggregates in `engine_counters`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from storage import mongo

_log = logging.getLogger("quantforge.llm.usage")

# Optional tiktoken encoder for precise token estimations when provider omits usage
_tiktoken_enc = None
try:
    import tiktoken
    try:
        _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
    except Exception as _enc_err:
        _log.debug("tiktoken encoding initialisation failed: %s", _enc_err)
        _tiktoken_enc = None
except ImportError as _imp_err:
    _log.debug("tiktoken import not available: %s", _imp_err)
    _tiktoken_enc = None


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string using tiktoken or characters/4 fallback."""
    if not text:
        return 0
    if _tiktoken_enc is not None:
        try:
            return len(_tiktoken_enc.encode(text, disallowed_special=()))
        except Exception as _enc_err:
            _log.debug("tiktoken encoding failed: %s", _enc_err)
    return max(1, len(text) // 4)


class UsageTracker:
    """Persists call-level audit records and maintains atomic running metrics."""

    def __init__(self, db=None, cfg: dict | None = None):
        self._db = db if db is not None else mongo.get_db()
        self._calls_col = self._db["llm_calls"]
        self._counters_col = self._db["engine_counters"]
        self._cfg = (cfg or {}).get("llm", {})
        # Price config per 1M tokens: e.g. {"prompt": 0.50, "completion": 1.50}
        self._prices = self._cfg.get("prices_per_million", {})

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        """Calculate USD cost if prices are configured."""
        if not self._prices:
            return None
        p_price = float(self._prices.get("prompt", 0.0))
        c_price = float(self._prices.get("completion", 0.0))
        cost = (prompt_tokens / 1_000_000.0) * p_price + (completion_tokens / 1_000_000.0) * c_price
        return round(cost, 6)

    def log_call(
        self,
        key_label: str,
        model: str,
        purpose: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        status: str,
        error_class: str | None = None,
        strategy_id: str | None = None,
        cycle_id: str | None = None,
        estimated: bool = False,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Record a completed or failed LLM request and update running counters."""
        current_time = now or datetime.now(timezone.utc)
        today_str = current_time.strftime("%Y-%m-%d")
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = self.calculate_cost(prompt_tokens, completion_tokens)

        doc = {
            "timestamp": current_time,
            "key_label": key_label,
            "model": model,
            "purpose": purpose,
            "strategy_id": strategy_id,
            "cycle_id": cycle_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "status": status,
            "error_class": error_class,
            "estimated": estimated,
            "cost_usd": cost_usd,
        }

        # 1. Insert audit record
        self._calls_col.insert_one(doc)

        # 2. Update atomic counters in engine_counters
        inc_fields = {
            "total_calls": 1,
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            f"by_key.{key_label}.calls": 1,
            f"by_key.{key_label}.total_tokens": total_tokens,
            f"by_model.{model}.calls": 1,
            f"by_model.{model}.total_tokens": total_tokens,
            f"by_purpose.{purpose}.calls": 1,
            f"by_purpose.{purpose}.total_tokens": total_tokens,
            f"by_day.{today_str}.calls": 1,
            f"by_day.{today_str}.total_tokens": total_tokens,
        }
        if cost_usd:
            inc_fields["total_cost_usd"] = cost_usd
            inc_fields[f"by_day.{today_str}.cost_usd"] = cost_usd

        self._counters_col.update_one(
            {"counter_id": "llm_usage"},
            {"$inc": inc_fields, "$set": {"last_call_at": current_time}},
            upsert=True,
        )

        return doc

    def get_usage_summary(self) -> dict[str, Any]:
        """Fetch running usage summary document."""
        doc = self._counters_col.find_one({"counter_id": "llm_usage"})
        if not doc:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "by_key": {},
                "by_model": {},
                "by_purpose": {},
                "by_day": {},
            }
        doc.pop("_id", None)
        return doc
