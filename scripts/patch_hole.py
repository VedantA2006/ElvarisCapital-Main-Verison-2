"""
Patch 1h and 4h CSVs by filling the September 2025 hole from 5m data.

- Backs up originals as .bak
- Aggregates 5m bars into 1h/4h for the missing window ONLY
- Appends, sorts, deduplicates by timestamp, and re-writes
- Logs every step for auditability
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from core.config import load_config, load_env
from core.data_loader import file_sha256

load_env()
cfg = load_config()
data_dir = Path(cfg["data"]["base_dir_resolved"])

# ── Load 5m data ──────────────────────────────────────────────────────────
m5_path = data_dir / cfg["data"]["files"]["5m"]
print(f"Loading 5m data from {m5_path}")
m5 = pd.read_csv(m5_path)
m5["timestamp"] = pd.to_datetime(m5["timestamp"])
print(f"  {len(m5)} bars, {m5['timestamp'].min()} -> {m5['timestamp'].max()}")

# ── Define the hole window ────────────────────────────────────────────────
HOLE_START = pd.Timestamp("2025-09-01 00:00:00")
HOLE_END   = pd.Timestamp("2025-10-01 00:00:00")

m5_hole = m5[(m5["timestamp"] >= HOLE_START) & (m5["timestamp"] < HOLE_END)].copy()
print(f"\n5m bars in hole window: {len(m5_hole)}")
assert len(m5_hole) > 5000, f"Expected ~6000 5m bars in Sep 2025, got {len(m5_hole)}"

# ── Aggregate into 1h and 4h ─────────────────────────────────────────────
def aggregate(src: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = (src.set_index("timestamp")
           .resample(freq, label="left", closed="left")
           .agg({"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"})
           .dropna()
           .reset_index())
    return agg

h1_fill = aggregate(m5_hole, "60min")
h4_fill = aggregate(m5_hole, "240min")
print(f"Aggregated: {len(h1_fill)} x 1h bars, {len(h4_fill)} x 4h bars")

# ── Patch each file ──────────────────────────────────────────────────────
for tf, fill_df, fname in [("1h", h1_fill, cfg["data"]["files"]["1h"]),
                            ("4h", h4_fill, cfg["data"]["files"]["4h"])]:
    path = data_dir / fname
    bak = path.with_suffix(".csv.bak")

    # Backup
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"\n[{tf}] Backed up {path.name} -> {bak.name}")
    else:
        print(f"\n[{tf}] Backup already exists: {bak.name}")

    orig_hash = file_sha256(path)
    orig = pd.read_csv(path)
    orig["timestamp"] = pd.to_datetime(orig["timestamp"])
    print(f"  Original: {len(orig)} rows, hash={orig_hash[:16]}...")

    # Remove any partial rows in the hole window (shouldn't exist, but be safe)
    in_hole = (orig["timestamp"] >= HOLE_START) & (orig["timestamp"] < HOLE_END)
    existing_in_hole = int(in_hole.sum())
    if existing_in_hole:
        print(f"  WARNING: {existing_in_hole} existing rows in hole window will be replaced")
        orig = orig[~in_hole]

    # Merge
    merged = pd.concat([orig, fill_df], ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)

    # Write back in the same naive-timestamp format
    out = merged.copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)

    new_hash = file_sha256(path)
    print(f"  Patched:  {len(merged)} rows (+{len(merged) - len(orig)}), hash={new_hash[:16]}...")
    print(f"  Hole coverage: {fill_df['timestamp'].min()} -> {fill_df['timestamp'].max()}")

    # Sanity: verify monotonic
    check = pd.read_csv(path)
    check["timestamp"] = pd.to_datetime(check["timestamp"])
    assert check["timestamp"].is_monotonic_increasing, f"{tf}: timestamps not monotonic after patch!"
    print(f"  [OK] Monotonic timestamp check passed")

print("\n=== Patch complete. Re-run 'python main.py verify-data' to confirm. ===")
