"""
core/config.py – Load config.yaml + .env, resolve paths, compute hashes.

Single source for configuration. Secrets are read from .env into the process
environment only; they are NEVER copied into the config dict, never hashed
into config_hash, and never logged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# quantforge/ package root (this file lives in quantforge/core/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SECRET_ENV_VARS = ("MONGO_URL", "LLM_API_KEY_1", "LLM_API_KEY_2")


def load_env(env_path: str | Path | None = None) -> None:
    """Load .env into os.environ (does not override already-set variables)."""
    path = Path(env_path) if env_path else PROJECT_ROOT / ".env"
    if path.exists():
        load_dotenv(path, override=False)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml and resolve relative paths against PROJECT_ROOT."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve the data dir relative to the project root
    base_dir = Path(cfg["data"]["base_dir"])
    if not base_dir.is_absolute():
        base_dir = (PROJECT_ROOT / base_dir).resolve()
    cfg["data"]["base_dir_resolved"] = str(base_dir)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    """Fail fast on config values that would break core guarantees."""
    s = cfg["splits"]
    total = s["train_ratio"] + s["validation_ratio"] + s["holdout_ratio"]
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    if min(s["train_ratio"], s["validation_ratio"], s["holdout_ratio"]) <= 0:
        raise ValueError("All split ratios must be > 0")
    if s["embargo_days"] < 0:
        raise ValueError("embargo_days must be >= 0")

    enabled = cfg["data"]["enabled_timeframes"]
    all_tfs = cfg["data"]["all_timeframes"]
    unknown = set(enabled) - set(all_tfs)
    if unknown:
        raise ValueError(f"enabled_timeframes contains unknown timeframes: {unknown}")

    for secret in SECRET_ENV_VARS:
        if _contains_value(cfg, os.environ.get(secret)):
            raise ValueError(f"A secret ({secret}) appears inside config.yaml. Secrets belong in .env only.")


def _contains_value(obj: Any, needle: str | None) -> bool:
    if not needle:
        return False
    if isinstance(obj, dict):
        return any(_contains_value(v, needle) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_value(v, needle) for v in obj)
    return isinstance(obj, str) and needle in obj


def config_hash(cfg: dict[str, Any]) -> str:
    """Deterministic SHA256 of the config (excluding derived/resolved paths)."""
    c = copy.deepcopy(cfg)
    c.get("data", {}).pop("base_dir_resolved", None)
    blob = json.dumps(c, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def code_hash(source: str) -> str:
    """SHA256 of strategy source code, normalised for line endings/trailing ws."""
    norm = "\n".join(line.rstrip() for line in source.replace("\r\n", "\n").split("\n")).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def mongo_db_name(cfg: dict[str, Any]) -> str:
    """DB name; QF_MONGO_DB env var overrides (used by pytest -> test DB)."""
    return os.environ.get("QF_MONGO_DB") or cfg["mongo"]["database"]
