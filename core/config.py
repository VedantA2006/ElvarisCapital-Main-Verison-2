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
        cfg = _safe_load_no_dupes(f.read())

    # Resolve the data dir relative to the project root
    base_dir = Path(cfg["data"]["base_dir"])
    if not base_dir.is_absolute():
        base_dir = (PROJECT_ROOT / base_dir).resolve()
    cfg["data"]["base_dir_resolved"] = str(base_dir)

    _validate_config(cfg)
    return cfg


class _NoDuplicateKeyLoader(yaml.SafeLoader):
    """YAML loader that raises on duplicate keys at any nesting level."""
    pass


def _no_duplicate_key_constructor(loader, node):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=False)
        if key in mapping:
            raise ValueError(
                f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=False)
    return mapping


_NoDuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _no_duplicate_key_constructor,
)


def _safe_load_no_dupes(text: str) -> dict:
    return yaml.load(text, Loader=_NoDuplicateKeyLoader)


def _validate_config(cfg: dict[str, Any]) -> None:
    """Fail fast on config values that would break core guarantees."""

    # ── Allowed top-level keys ──────────────────────────────────────────
    _ALLOWED_TOP = {
        "data", "sessions", "market_hours", "validation", "splits", "costs",
        "contract", "sizing", "strategy", "gates", "diversity", "llm",
        "forward", "sandbox", "robustness_weights", "mongo", "dashboard",
        "logging", "cross_asset",
    }
    unknown_top = set(cfg.keys()) - _ALLOWED_TOP
    if unknown_top:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown_top)}")

    # ── Allowed sub-keys in commonly-mistyped sections ──────────────────
    _TRAIN_KEYS = {"min_profit_factor", "min_sharpe", "max_drawdown", "min_expectancy"}
    if "train" in cfg.get("gates", {}):
        bad = set(cfg["gates"]["train"].keys()) - _TRAIN_KEYS
        if bad:
            raise ValueError(f"unknown keys in gates.train: {sorted(bad)}")

    # ── Split ratios ────────────────────────────────────────────────────
    s = cfg["splits"]
    total = s["train_ratio"] + s["validation_ratio"] + s["holdout_ratio"]
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    if min(s["train_ratio"], s["validation_ratio"], s["holdout_ratio"]) <= 0:
        raise ValueError("All split ratios must be > 0")
    if s["embargo_days"] < 0:
        raise ValueError("embargo_days must be >= 0")

    # ── Timeframes ──────────────────────────────────────────────────────
    enabled = cfg["data"]["enabled_timeframes"]
    all_tfs = cfg["data"]["all_timeframes"]
    unknown = set(enabled) - set(all_tfs)
    if unknown:
        raise ValueError(f"enabled_timeframes contains unknown timeframes: {unknown}")

    # ── Secrets check ───────────────────────────────────────────────────
    for secret in SECRET_ENV_VARS:
        if _contains_value(cfg, os.environ.get(secret)):
            raise ValueError(f"A secret ({secret}) appears inside config.yaml. Secrets belong in .env only.")

    # ── Range checks on critical numeric parameters ─────────────────────
    _range_checks = [
        (("sizing", "risk_per_trade"),    0.0001, 1.0,    "risk_per_trade must be in (0, 1]"),
        (("sizing", "initial_equity"),    1.0,    None,   "initial_equity must be > 0"),
        (("costs", "spread", "default"),  0.0,    None,   "spread.default must be >= 0"),
        (("gates", "dsr", "min_probability"), 0.0, 1.0,   "dsr.min_probability must be in [0, 1]"),
        (("sandbox", "wall_clock_timeout"), 1,    None,   "wall_clock_timeout must be >= 1"),
    ]
    for path, lo, hi, msg in _range_checks:
        node = cfg
        for k in path:
            if not isinstance(node, dict) or k not in node:
                break
            node = node[k]
        else:
            if not isinstance(node, (int, float)):
                raise ValueError(f"{'.'.join(path)} must be numeric, got {type(node).__name__}")
            if lo is not None and node < lo:
                raise ValueError(msg)
            if hi is not None and node > hi:
                raise ValueError(msg)

    # ── Type checks on numeric fields ───────────────────────────────────
    _type_checks = [
        (("gates", "train", "min_sharpe"), (int, float)),
        (("gates", "train", "min_profit_factor"), (int, float)),
        (("gates", "train", "max_drawdown"), (int, float)),
    ]
    for path, expected in _type_checks:
        node = cfg
        for k in path:
            if not isinstance(node, dict) or k not in node:
                break
            node = node[k]
        else:
            if not isinstance(node, expected):
                raise ValueError(f"{'.'.join(path)} must be {expected}, got {type(node).__name__}")


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
