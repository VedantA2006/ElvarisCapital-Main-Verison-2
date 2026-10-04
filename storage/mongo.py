"""
storage/mongo.py – MongoDB connection, collection management, and helpers.

Collections and indexes as specified in Section 13.
Secrets come from .env only. Never log or expose the connection string.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone


from pymongo import MongoClient, ASCENDING, DESCENDING, IndexModel, ReturnDocument
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

_log = logging.getLogger("quantforge.mongo")


_client: MongoClient | None = None
_db: Database | None = None


def get_client() -> MongoClient:
    """Return a singleton MongoClient. Creates on first call."""
    global _client
    if _client is None:
        url = os.environ.get("MONGO_URL")
        if not url:
            raise RuntimeError("MONGO_URL not set in environment")
        _client = MongoClient(url, serverSelectionTimeoutMS=10_000)
        # Verify connectivity eagerly
        _client.admin.command("ping")
    return _client


def get_db(db_name: str | None = None) -> Database:
    """Return the quantforge database.

    Name resolution: explicit arg > QF_MONGO_DB env var > 'quantforge'.
    pytest sets QF_MONGO_DB to the test database so tests never touch
    production collections.
    """
    global _db
    name = db_name or os.environ.get("QF_MONGO_DB") or "quantforge"
    if _db is None or _db.name != name:
        _db = get_client()[name]
    return _db


def close() -> None:
    """Close the client connection."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None


# ─── Collection accessors ──────────────────────────────────────────────────

def col_strategies() -> Collection:
    return get_db()["strategies"]

def col_backtests() -> Collection:
    return get_db()["backtests"]

def col_gate_results() -> Collection:
    return get_db()["gate_results"]

def col_llm_calls() -> Collection:
    return get_db()["llm_calls"]

def col_key_status() -> Collection:
    return get_db()["key_status"]

def col_trials() -> Collection:
    """DEPRECATED: use col_runs() for run documents, col_counters() for counters."""
    return get_db()["trials"]

def col_runs() -> Collection:
    """Run documents (one per pipeline attempt)."""
    return get_db()["runs"]

def col_counters() -> Collection:
    """Atomic counters (trial count, etc.)."""
    return get_db()["counters"]

def col_coverage_map() -> Collection:
    return get_db()["coverage_map"]

def col_holdout_access() -> Collection:
    return get_db()["holdout_access"]

def col_logs() -> Collection:
    return get_db()["logs"]

def col_leaderboard() -> Collection:
    return get_db()["leaderboard"]

def col_forward_trades() -> Collection:
    return get_db()["forward_trades"]

def col_data_reports() -> Collection:
    return get_db()["data_reports"]

def col_session_map() -> Collection:
    return get_db()["session_map"]


# ─── Index setup ────────────────────────────────────────────────────────────

def ensure_indexes() -> None:
    """Create all required indexes. Safe to call multiple times (idempotent)."""

    col_strategies().create_indexes([
        IndexModel([("status", ASCENDING)]),
        IndexModel([("concept_family", ASCENDING)]),
        IndexModel([("timeframe", ASCENDING)]),
        IndexModel([("fingerprint", ASCENDING)]),
        IndexModel([("created_at", DESCENDING)]),
        IndexModel([("code_hash", ASCENDING)]),
    ])

    col_backtests().create_indexes([
        IndexModel([("strategy_id", ASCENDING)]),
        IndexModel([("strategy_id", ASCENDING), ("split", ASCENDING)]),
        IndexModel([("created_at", DESCENDING)]),
    ])

    col_gate_results().create_indexes([
        IndexModel([("strategy_id", ASCENDING)]),
        IndexModel([("strategy_id", ASCENDING), ("gate_name", ASCENDING)]),
    ])

    col_llm_calls().create_indexes([
        IndexModel([("timestamp", DESCENDING)]),
        IndexModel([("key_label", ASCENDING)]),
        IndexModel([("strategy_id", ASCENDING)]),
        IndexModel([("purpose", ASCENDING)]),
    ])

    col_key_status().create_indexes([
        IndexModel([("label", ASCENDING)], unique=True),
    ])

    # ── counters (atomic increment; not the same as run documents) ──
    col_counters().create_indexes([
        IndexModel([("counter_id", ASCENDING)], unique=True),
    ])

    # ── runs (one per pipeline attempt, formerly mixed into 'trials') ──
    col_runs().create_indexes([
        IndexModel([("trial_id", ASCENDING)], unique=True, sparse=True),
        IndexModel([("status", ASCENDING)]),
        IndexModel([("started_at", DESCENDING)]),
    ])

    # ── one-time migration: move legacy trial docs out of 'trials' ──────
    _migrate_legacy_trials()

    col_coverage_map().create_indexes([
        IndexModel([
            ("concept_family", ASCENDING),
            ("timeframe", ASCENDING),
            ("session", ASCENDING),
            ("regime_bias", ASCENDING),
        ], unique=True),
    ])

    # CRITICAL: holdout_access has a UNIQUE index on strategy_id.
    # This is the enforcement mechanism for the "one holdout access" rule.
    col_holdout_access().create_indexes([
        IndexModel([("strategy_id", ASCENDING)], unique=True),
    ])

    col_logs().create_indexes([
        IndexModel([("timestamp", DESCENDING)]),
        IndexModel([("level", ASCENDING)]),
        IndexModel([("cycle_id", ASCENDING)]),
        IndexModel([("strategy_id", ASCENDING)]),
    ])

    col_leaderboard().create_indexes([
        IndexModel([("robustness_score", DESCENDING)]),
        IndexModel([("timeframe", ASCENDING)]),
        IndexModel([("status", ASCENDING)]),
    ])

    col_forward_trades().create_indexes([
        IndexModel([("strategy_id", ASCENDING)]),
        IndexModel([("timestamp", DESCENDING)]),
    ])

    col_data_reports().create_indexes([
        IndexModel([("timeframe", ASCENDING)]),
        IndexModel([("timestamp", DESCENDING)]),
    ])


# ─── Legacy migration ──────────────────────────────────────────────────────

def _migrate_legacy_trials() -> None:
    """One-time migration: move counter docs and run docs out of the old 'trials'
    collection into 'counters' and 'runs'.  Safe to call multiple times (idempotent).
    """
    db = get_db()
    legacy = db["trials"]
    if legacy.estimated_document_count() == 0:
        return

    # Move counter documents → counters
    for doc in legacy.find({"counter_id": {"$exists": True}}):
        try:
            col_counters().insert_one(doc)
        except DuplicateKeyError:
            _log.debug("counter doc already migrated: %s", doc.get("counter_id"))
        legacy.delete_one({"_id": doc["_id"]})

    # Move run documents → runs
    for doc in legacy.find({"trial_id": {"$exists": True}}):
        try:
            col_runs().insert_one(doc)
        except DuplicateKeyError:
            _log.debug("run doc already migrated: %s", doc.get("trial_id"))
        legacy.delete_one({"_id": doc["_id"]})

    # Drop the old problematic unique index on counter_id if it still exists
    try:
        legacy.drop_index("counter_id_1")
    except Exception as exc:
        _log.debug("could not drop legacy counter_id index: %s", exc)

    _log.info("legacy trials migration complete")


# ─── Helpers ────────────────────────────────────────────────────────────────

def increment_trial_counter() -> int:
    """Atomically increment and return the global trial counter.

    Uses upsert + $inc so it works from a cold start.
    """
    result = col_counters().find_one_and_update(
        {"counter_id": "global"},
        {"$inc": {"count": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return int(result["count"])


def get_trial_count() -> int:
    """Read the current trial count without incrementing."""
    doc = col_counters().find_one({"counter_id": "global"})
    if doc is None:
        return 0
    return int(doc["count"])


def record_holdout_access(strategy_id: str, *, timeframe: str = "", purpose: str = "",
                          file_hash: str = "") -> None:
    """Record a holdout access. Raises PermissionError on a second attempt.

    Called ONLY by core.splits.DataStore.get_data(split="holdout").
    The unique index on strategy_id is the enforcement mechanism, so the
    check-and-insert is atomic even across processes.
    """
    try:
        col_holdout_access().insert_one({
            "strategy_id": strategy_id,
            "timeframe": timeframe,
            "purpose": purpose,
            "file_hash": file_hash,
            "timestamp": datetime.now(timezone.utc),
        })
    except DuplicateKeyError:
        raise PermissionError(
            f"Holdout already accessed for strategy '{strategy_id}'. "
            "Second access is forbidden."
        )


def has_holdout_access(strategy_id: str) -> bool:
    """Check whether holdout has already been accessed for a strategy."""
    return col_holdout_access().find_one({"strategy_id": strategy_id}) is not None


def utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)
