"""
storage/logger.py – Structured logging to MongoDB + rotating JSONL files.

Every log entry goes to both Mongo and a JSONL file.
Secrets are never included in log messages.
"""

from __future__ import annotations

import json
import logging

import sys
from datetime import datetime, timezone
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Any

from storage.mongo import col_logs, utcnow


class QuantForgeLogger:
    """Dual-destination structured logger (Mongo + JSONL)."""

    def __init__(
        self,
        jsonl_dir: str = "logs",
        max_file_size_mb: int = 50,
        rotate_count: int = 5,
        level: str = "INFO",
    ):
        self.jsonl_dir = Path(jsonl_dir)
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)

        # Python logger for console + JSONL
        self._logger = logging.getLogger("quantforge")
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.handlers.clear()

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        self._logger.addHandler(console)

        # Rotating JSONL handler
        jsonl_path = self.jsonl_dir / "quantforge.jsonl"
        file_handler = RotatingFileHandler(
            str(jsonl_path),
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=rotate_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(_JsonlFormatter())
        self._logger.addHandler(file_handler)

        # Flag to control Mongo writes (can be disabled for testing)
        self.mongo_enabled = True
        self.dropped_writes = 0

    def log(
        self,
        level: str,
        message: str,
        *,
        cycle_id: str | None = None,
        stage: str | None = None,
        strategy_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write a structured log entry to Mongo + JSONL + console."""
        doc = {
            "timestamp": utcnow(),
            "level": level.upper(),
            "message": message,
            "cycle_id": cycle_id,
            "stage": stage,
            "strategy_id": strategy_id,
        }
        if extra:
            doc["extra"] = extra

        # Console + JSONL via standard logging
        log_fn = getattr(self._logger, level.lower(), self._logger.info)
        # Build a compact message for console
        parts = [message]
        if cycle_id:
            parts.append(f"cycle={cycle_id}")
        if stage:
            parts.append(f"stage={stage}")
        if strategy_id:
            parts.append(f"strategy={strategy_id}")
        log_fn(" | ".join(parts), extra={"structured": doc})

        # Mongo (best effort — don't crash on Mongo failures)
        if self.mongo_enabled:
            try:
                col_logs().insert_one(doc)
            except Exception as exc:
                self.dropped_writes += 1
                # If Mongo is down, the JSONL file still has the entry
                self._logger.warning("Mongo log write failed (%s), dropped_writes=%d",
                                     exc, self.dropped_writes)

    def info(self, message: str, **kwargs: Any) -> None:
        self.log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self.log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self.log("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self.log("DEBUG", message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self.log("CRITICAL", message, **kwargs)


class _JsonlFormatter(logging.Formatter):
    """Format log records as single-line JSON for the JSONL file."""

    def format(self, record: logging.LogRecord) -> str:
        structured = getattr(record, "structured", None)
        if structured:
            # Serialize the structured doc
            doc = dict(structured)
            # Convert datetime to ISO string for JSON
            if "timestamp" in doc and isinstance(doc["timestamp"], datetime):
                doc["timestamp"] = doc["timestamp"].isoformat()
            return json.dumps(doc, default=str)
        # Fallback: plain text record
        return json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        })


# Module-level singleton
_logger_instance: QuantForgeLogger | None = None


def get_logger(**kwargs: Any) -> QuantForgeLogger:
    """Return the singleton logger, creating it on first call."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = QuantForgeLogger(**kwargs)
    return _logger_instance
