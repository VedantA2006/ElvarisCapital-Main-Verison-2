"""
core/data_vault.py – Encrypted Data Vault for Validation and Holdout data (Phase F1b).

Guarantees (SBX-2 defense in depth):
1. Validation and holdout data can be encrypted in a .vault file.
2. The decryption key (VAULT_KEY) is held only by the engine / parent process.
3. Even if a sandboxed strategy executes pd.read_csv on the disk file, it only sees train data.
"""

from __future__ import annotations

import base64
import os
import pickle
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from cryptography.fernet import Fernet

from core.data_loader import data_path, TIMEFRAME_MINUTES


def generate_vault_key() -> str:
    """Generate a fresh base64 URL-safe 32-byte encryption key."""
    return Fernet.generate_key().decode("ascii")


def get_vault_key(cfg: dict | None = None) -> bytes:
    """Obtain the vault key from environment (VAULT_KEY) or config."""
    key = os.environ.get("VAULT_KEY")
    if not key and cfg is not None:
        key = cfg.get("data", {}).get("vault_key")
    if not key:
        raise ValueError("VAULT_KEY not found in environment or config")
    if isinstance(key, str):
        return key.strip().encode("ascii")
    return key


def vault_path(timeframe: str, cfg: dict) -> Path:
    """Path to the encrypted vault file for a timeframe."""
    src = data_path(timeframe, cfg)
    return src.with_suffix(".vault")


def encrypt_splits(
    df: pd.DataFrame,
    boundaries: Any,
    key: bytes | str,
) -> bytes:
    """Package and encrypt validation and holdout splits."""
    if isinstance(key, str):
        key = key.encode("ascii")
    f = Fernet(key)

    ts = df["timestamp"]
    val_lo, val_hi = boundaries.range_for("validation")
    hold_lo, hold_hi = boundaries.range_for("holdout")

    val_df = df.loc[(ts >= val_lo) & (ts < val_hi)].copy().reset_index(drop=True)
    hold_df = df.loc[(ts >= hold_lo) & (ts < hold_hi)].copy().reset_index(drop=True)

    payload = {
        "validation": val_df,
        "holdout": hold_df,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_end": str(boundaries.train_end),
        "validation_start": str(val_lo),
        "validation_end": str(val_hi),
        "holdout_start": str(hold_lo),
        "holdout_end": str(hold_hi),
    }

    raw = zlib.compress(pickle.dumps(payload, protocol=5))
    return f.encrypt(raw)


def decrypt_vault(vault_bytes: bytes, key: bytes | str) -> dict[str, Any]:
    """Decrypt the vault payload and restore DataFrames."""
    if isinstance(key, str):
        key = key.encode("ascii")
    f = Fernet(key)
    decrypted = f.decrypt(vault_bytes)
    decompressed = zlib.decompress(decrypted)
    return pickle.loads(decompressed)


def save_vault(
    timeframe: str,
    df: pd.DataFrame,
    boundaries: Any,
    cfg: dict,
    key: bytes | str | None = None,
) -> Path:
    """Encrypt and write the vault file to disk."""
    k = key if key is not None else get_vault_key(cfg)
    enc = encrypt_splits(df, boundaries, k)
    vp = vault_path(timeframe, cfg)
    vp.write_bytes(enc)
    return vp


def load_vault_split(
    timeframe: str,
    split: str,
    cfg: dict,
    key: bytes | str | None = None,
) -> pd.DataFrame:
    """Load and decrypt a single split from the vault file."""
    if split not in ("validation", "holdout"):
        raise ValueError(f"Vault only contains 'validation' and 'holdout' splits, requested '{split}'")
    vp = vault_path(timeframe, cfg)
    if not vp.exists():
        raise FileNotFoundError(f"Vault file not found: {vp}")
    k = key if key is not None else get_vault_key(cfg)
    payload = decrypt_vault(vp.read_bytes(), k)
    if split not in payload:
        raise KeyError(f"Split '{split}' not found in vault")
    return payload[split]


def strip_working_csv(timeframe: str, boundaries: Any, cfg: dict) -> Path:
    """Strip validation and holdout rows from the working CSV, keeping only train data."""
    csv_file = data_path(timeframe, cfg)
    df = pd.read_csv(csv_file)
    from core.data_loader import convert_to_utc
    df_utc, _ = convert_to_utc(df, cfg["data"]["broker_tz"])
    mask = df_utc["timestamp"] < boundaries.train_end
    train_only = df.loc[mask]
    train_only.to_csv(csv_file, index=False)
    return csv_file
