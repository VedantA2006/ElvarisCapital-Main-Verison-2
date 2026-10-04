"""
scripts/verify_alignment.py – Inspect and report the native bar grid of data files.

Detects:
1. Set of start minutes-of-day in UTC.
2. Set of start minutes-of-day in NY time (America/New_York, DST-aware).
3. Grid cadence and interval consistency.
4. Daily break and weekend boundaries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config, load_env
from core.data_loader import TIMEFRAME_MINUTES, convert_to_utc, data_path, file_sha256


def verify_file_alignment(timeframe: str, cfg: dict) -> dict[str, Any]:
    """Analyze the native grid of a given timeframe's data file."""
    path = data_path(timeframe, cfg)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    fhash = file_sha256(path)
    df_raw = pd.read_csv(path)
    df_utc, conv_issues = convert_to_utc(df_raw, cfg["data"]["broker_tz"])

    ts_utc = df_utc["timestamp"].dropna()
    ts_ny = ts_utc.dt.tz_convert("America/New_York")

    # Minutes of day: 0..1439
    mins_utc = sorted(set(ts_utc.dt.hour * 60 + ts_utc.dt.minute))
    mins_ny = sorted(set(ts_ny.dt.hour * 60 + ts_ny.dt.minute))

    # Formatted HH:MM sets
    utc_hours = sorted(set(ts_utc.dt.strftime("%H:%M")))
    ny_hours = sorted(set(ts_ny.dt.strftime("%H:%M")))

    expected_interval = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    diffs = ts_utc.diff().dropna()
    normal_bars = int((diffs == expected_interval).sum())
    gaps = int((diffs > expected_interval).sum())
    negative_or_zero = int((diffs <= pd.Timedelta(0)).sum())

    # Check for off-grid bar starts (e.g. seconds != 0 or minutes not multiple of timeframe)
    off_grid = int(((ts_utc.dt.second != 0) | ((ts_utc.dt.minute % TIMEFRAME_MINUTES[timeframe]) != 0)).sum())

    report = {
        "timeframe": timeframe,
        "filename": path.name,
        "sha256": fhash,
        "row_count": len(df_raw),
        "valid_timestamps": len(ts_utc),
        "time_range_utc": f"{ts_utc.min()} -> {ts_utc.max()}",
        "time_range_ny": f"{ts_ny.min()} -> {ts_ny.max()}",
        "start_minutes_utc": mins_utc,
        "start_minutes_ny": mins_ny,
        "active_hours_utc": utc_hours,
        "active_hours_ny": ny_hours,
        "expected_interval_minutes": TIMEFRAME_MINUTES[timeframe],
        "normal_cadence_bars": normal_bars,
        "gaps_count": gaps,
        "negative_or_zero_deltas": negative_or_zero,
        "off_grid_bars": off_grid,
        "conversion_issues": conv_issues,
        "is_grid_aligned": off_grid == 0 and negative_or_zero == 0,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify native bar grid alignment of XAUUSD data files.")
    parser.add_argument("--timeframe", "-t", choices=["5m", "15m", "1h", "4h"], default=None,
                        help="Timeframe to inspect. Defaults to all configured files.")
    args = parser.parse_args()

    load_env()
    cfg = load_config()

    timeframes = [args.timeframe] if args.timeframe else cfg["data"]["files"].keys()
    print("=" * 80)
    print("QuantForge Bar Grid Alignment Report")
    print("=" * 80)

    all_aligned = True
    for tf in timeframes:
        try:
            rep = verify_file_alignment(tf, cfg)
        except Exception as exc:
            print(f"[{tf}] Error checking file: {exc}")
            all_aligned = False
            continue

        status = "ALIGNED" if rep["is_grid_aligned"] else "MISALIGNED"
        print(f"\n--- Timeframe: {tf} ({rep['filename']}) [{status}] ---")
        print(f"  SHA256:          {rep['sha256'][:16]}...")
        print(f"  Total Rows:      {rep['row_count']}")
        print(f"  Range (UTC):     {rep['time_range_utc']}")
        print(f"  Range (NY):      {rep['time_range_ny']}")
        print(f"  Cadence:         {rep['expected_interval_minutes']}m interval")
        print(f"  Normal bars:     {rep['normal_cadence_bars']}, Gaps: {rep['gaps_count']}, Off-grid: {rep['off_grid_bars']}")
        print(f"  NY Start Hours:  {', '.join(rep['active_hours_ny'][:10])}{' ...' if len(rep['active_hours_ny']) > 10 else ''}")
        print(f"  UTC Start Hours: {', '.join(rep['active_hours_utc'][:10])}{' ...' if len(rep['active_hours_utc']) > 10 else ''}")

        if not rep["is_grid_aligned"]:
            all_aligned = False

    print("\n" + "=" * 80)
    print(f"Overall Grid Alignment: {'PASSED' if all_aligned else 'FAILED'}")
    print("=" * 80)
    return 0 if all_aligned else 1


if __name__ == "__main__":
    sys.exit(main())
