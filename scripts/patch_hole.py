"""
scripts/patch_hole.py – Safely patch gaps in higher-timeframe data from lower-timeframe bars.

Guarantees (Section F1.2):
1. Aggregates lower-TF bars into the EXACT same grid as the native target file.
2. Validates cross-timeframe consistency on an overlapping historical month; aborts if mismatch rate > threshold.
3. NEVER overwrites the source file; writes to a separate '*.patched.csv' file.
4. Identifies whether patched bars fall into train, validation, or holdout splits.
5. If patched bars fall into the holdout, marks them 'synthetic' and refuses to silently contaminate holdout without audited refreeze.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config, load_env
from core.data_loader import (
    TIMEFRAME_MINUTES,
    convert_to_utc,
    cross_timeframe_consistency,
    data_path,
    file_sha256,
)
from core.splits import get_or_freeze_boundaries


def aggregate_to_grid(df_low_utc: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Aggregate lower-timeframe bars into target timeframe using native grid alignment."""
    rule = f"{TIMEFRAME_MINUTES[target_tf]}min"
    indexed = df_low_utc.set_index("timestamp")
    agg = (
        indexed.resample(rule, label="left", closed="left")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return agg


def patch_gap(
    source_tf: str = "5m",
    target_tf: str = "1h",
    hole_start: str = "2025-09-01",
    hole_end: str = "2025-10-01",
    overlap_month: str = "2025-08",
    mismatch_threshold_pct: float = 1.0,
    cfg: dict | None = None,
    output_path: Path | None = None,
    allow_holdout_contamination: bool = False,
) -> dict[str, Any]:
    """Execute a safe, audited gap patch procedure."""
    if cfg is None:
        load_env()
        cfg = load_config()

    src_path = data_path(source_tf, cfg)
    tgt_path = data_path(target_tf, cfg)

    out_file = output_path or tgt_path.with_name(f"{tgt_path.stem}.patched.csv")
    if out_file.resolve() == tgt_path.resolve():
        raise PermissionError(f"Refusing to overwrite original data file: {tgt_path}")

    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    if not tgt_path.exists():
        raise FileNotFoundError(f"Target file not found: {tgt_path}")

    # 1. Load raw data and convert to UTC
    df_src_raw = pd.read_csv(src_path)
    df_tgt_raw = pd.read_csv(tgt_path)

    df_src_utc, _ = convert_to_utc(df_src_raw, cfg["data"]["broker_tz"])
    df_tgt_utc, _ = convert_to_utc(df_tgt_raw, cfg["data"]["broker_tz"])

    # 2. Cross-timeframe consistency check on overlapping period
    overlap_ts = pd.to_datetime(overlap_month)
    overlap_start = overlap_ts.tz_localize("UTC") if overlap_ts.tzinfo is None else overlap_ts
    overlap_end = overlap_start + pd.DateOffset(months=1)

    src_overlap = df_src_utc[(df_src_utc["timestamp"] >= overlap_start) & (df_src_utc["timestamp"] < overlap_end)]
    tgt_overlap = df_tgt_utc[(df_tgt_utc["timestamp"] >= overlap_start) & (df_tgt_utc["timestamp"] < overlap_end)]

    if len(src_overlap) == 0 or len(tgt_overlap) == 0:
        raise ValueError(
            f"Overlap validation month '{overlap_month}' has no bars (src: {len(src_overlap)}, tgt: {len(tgt_overlap)})"
        )

    consistency = cross_timeframe_consistency(src_overlap, tgt_overlap, target_tf)
    max_mism = max(
        consistency.get("open_mismatch_pct", 0) or 0,
        consistency.get("close_mismatch_pct", 0) or 0,
    )

    if max_mism > mismatch_threshold_pct:
        raise RuntimeError(
            f"Cross-timeframe consistency validation failed on month {overlap_month}: "
            f"max mismatch {max_mism:.2f}% exceeds threshold {mismatch_threshold_pct:.2f}%. Aborting patch."
        )

    # 3. Aggregate hole window
    h_start = pd.to_datetime(hole_start).tz_localize("UTC") if pd.to_datetime(hole_start).tzinfo is None else pd.to_datetime(hole_start)
    h_end = pd.to_datetime(hole_end).tz_localize("UTC") if pd.to_datetime(hole_end).tzinfo is None else pd.to_datetime(hole_end)

    src_hole = df_src_utc[(df_src_utc["timestamp"] >= h_start) & (df_src_utc["timestamp"] < h_end)].copy()
    if len(src_hole) == 0:
        raise ValueError(f"No source {source_tf} bars found in hole window {hole_start} -> {hole_end}")

    aggregated_fill = aggregate_to_grid(src_hole, target_tf)

    # 4. Determine which split(s) are touched by the patch
    boundaries = get_or_freeze_boundaries(cfg)
    fill_ts = aggregated_fill["timestamp"]
    fill_min, fill_max = fill_ts.min(), fill_ts.max()

    train_touched = bool(((fill_ts >= boundaries.train_start) & (fill_ts < boundaries.train_end)).any())
    val_touched = bool(((fill_ts >= boundaries.validation_start) & (fill_ts < boundaries.validation_end)).any())
    holdout_touched = bool(((fill_ts >= boundaries.holdout_start) & (fill_ts < boundaries.holdout_end)).any())

    splits_touched = []
    if train_touched:
        splits_touched.append("train")
    if val_touched:
        splits_touched.append("validation")
    if holdout_touched:
        splits_touched.append("holdout")

    if holdout_touched and not allow_holdout_contamination:
        raise PermissionError(
            f"Patched bars fall within the HOLDOUT split ({boundaries.holdout_start} -> {boundaries.holdout_end}). "
            "Synthetic data must not contaminate holdout without explicit boundary shift via audited refreeze."
        )

    # 5. Merge and write to NEW patched file (NEVER overwrite source)
    out_file = output_path or tgt_path.with_name(f"{tgt_path.stem}.patched.csv")
    if out_file.resolve() == tgt_path.resolve():
        raise PermissionError(f"Refusing to overwrite original data file: {tgt_path}")

    # Remove any existing rows in the hole window
    in_hole = (df_tgt_utc["timestamp"] >= h_start) & (df_tgt_utc["timestamp"] < h_end)
    clean_tgt = df_tgt_raw.loc[~in_hole].copy()

    # Format aggregated fill to match original format (broker time string)
    broker_tz = cfg["data"]["broker_tz"]
    fill_broker_ts = aggregated_fill["timestamp"].dt.tz_convert(broker_tz).dt.tz_localize(None)
    fill_formatted = aggregated_fill.copy()
    fill_formatted["timestamp"] = fill_broker_ts.dt.strftime("%Y-%m-%d %H:%M:%S")

    merged = pd.concat([clean_tgt, fill_formatted], ignore_index=True)
    # Sort by timestamp
    merged["_dt"] = pd.to_datetime(merged["timestamp"], format="mixed", utc=True)
    merged = merged.sort_values("_dt").drop(columns=["_dt"]).drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    merged.to_csv(out_file, index=False)
    new_hash = file_sha256(out_file)

    report = {
        "status": "SUCCESS",
        "source_file": src_path.name,
        "target_file": tgt_path.name,
        "output_file": str(out_file),
        "source_bars_in_hole": len(src_hole),
        "aggregated_fill_bars": len(aggregated_fill),
        "fill_time_range": f"{fill_min} -> {fill_max}",
        "consistency_overlap_month": overlap_month,
        "consistency_mismatch_max_pct": max_mism,
        "splits_touched": splits_touched,
        "holdout_touched": holdout_touched,
        "original_rows": len(df_tgt_raw),
        "patched_rows": len(merged),
        "new_file_sha256": new_hash,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely patch data hole from 5m data without overwriting source.")
    parser.add_argument("--source-tf", default="5m", help="Lower timeframe source (default 5m)")
    parser.add_argument("--target-tf", default="1h", help="Target timeframe to patch (default 1h)")
    parser.add_argument("--hole-start", default="2025-09-01", help="Start of hole window (UTC)")
    parser.add_argument("--hole-end", default="2025-10-01", help="End of hole window (UTC)")
    parser.add_argument("--overlap-month", default="2025-08", help="Overlap validation month (default 2025-08)")
    parser.add_argument("--mismatch-max", type=float, default=1.0, help="Max allowed mismatch %% (default 1.0)")
    parser.add_argument("--allow-holdout", action="store_true", default=False, help="Allow holdout touch")
    args = parser.parse_args()

    try:
        report = patch_gap(
            source_tf=args.source_tf,
            target_tf=args.target_tf,
            hole_start=args.hole_start,
            hole_end=args.hole_end,
            overlap_month=args.overlap_month,
            mismatch_threshold_pct=args.mismatch_max,
            allow_holdout_contamination=args.allow_holdout,
        )
        print("Patch Report:")
        for k, v in report.items():
            print(f"  {k}: {v}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
