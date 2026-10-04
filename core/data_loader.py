"""
core/data_loader.py – Load, validate, hash and label XAUUSD bar data.

Principles (Section 3):
- NEVER silently fix data. Every anomaly is reported; critical ones halt.
- Every dataset gets a SHA256 file hash (stored with every backtest).
- Timestamps are converted to tz-aware UTC from the configured broker tz.
- Sessions are computed in each market's local time -> DST-correct.

NOTE: Strategy/backtest code must NOT call `load_and_validate` directly.
All split-aware access goes through `core.splits.DataStore.get_data`, which
enforces the holdout lock. A test (tests/test_phase1_foundations.py) scans
the codebase to enforce this.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TIMEFRAME_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


# ─── Hashing ────────────────────────────────────────────────────────────────

def file_sha256(filepath: str | Path) -> str:
    """SHA256 of the raw file bytes."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_sha256(df: pd.DataFrame) -> str:
    """Deterministic SHA256 of a bar DataFrame's content (OHLCV + timestamps)."""
    h = hashlib.sha256()
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC")
    h.update(ts.astype("int64").to_numpy().tobytes())
    for col in ("open", "high", "low", "close", "volume"):
        h.update(np.ascontiguousarray(df[col].to_numpy(dtype=np.float64)).tobytes())
    return h.hexdigest()


# ─── Loading ────────────────────────────────────────────────────────────────

def data_path(timeframe: str, cfg: dict) -> Path:
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    return Path(cfg["data"]["base_dir_resolved"]) / cfg["data"]["files"][timeframe]


def load_raw_data(timeframe: str, cfg: dict) -> tuple[pd.DataFrame, str]:
    """Read the CSV exactly as stored. Returns (df, file_hash). No fixing."""
    path = data_path(timeframe, cfg)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns {sorted(missing)}")
    return df, file_sha256(path)


def convert_to_utc(df: pd.DataFrame, broker_tz: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse timestamps and convert broker tz -> UTC.

    Ambiguous/non-existent local times (DST transitions) become NaT and are
    REPORTED (count returned), not silently dropped; NaT timestamps are a
    critical validation error downstream.
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], format="mixed", errors="coerce")
    unparseable = int(ts.isna().sum())
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(broker_tz, ambiguous="NaT", nonexistent="NaT")
    ts = ts.dt.tz_convert("UTC")
    dst_nat = int(ts.isna().sum()) - unparseable
    out["timestamp"] = ts
    return out, {"unparseable_timestamps": unparseable, "dst_ambiguous_or_nonexistent": dst_nat}


# ─── Validation ─────────────────────────────────────────────────────────────

def _classify_gaps(ts: pd.Series, timeframe: str, cfg: dict) -> dict[str, Any]:
    """Classify every gap (> 1 bar interval) as daily_break / weekend / unexplained."""
    mh = cfg["market_hours"]
    interval = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    diffs = ts.diff()
    gap_idx = np.where(diffs > interval)[0]

    counts = {"daily_break": 0, "weekend": 0, "unexplained": 0}
    unexplained: list[dict[str, Any]] = []
    long_holes: list[dict[str, Any]] = []
    break_max = pd.Timedelta(minutes=mh["daily_break_max_minutes"])
    weekend_max = pd.Timedelta(hours=mh["weekend_max_hours"])
    ny_tz = mh["tz"]

    for i in gap_idx:
        prev_ts, next_ts = ts.iloc[i - 1], ts.iloc[i]
        gap = next_ts - prev_ts
        missing_start = (prev_ts + interval).tz_convert(ny_tz)  # first missing bar open
        missing_end = next_ts.tz_convert(ny_tz)                 # exclusive
        prev_ny = prev_ts.tz_convert(ny_tz)

        # Daily break: missing window overlaps [17:00, 17:00 + break_max) NY local
        brk_start = missing_start.normalize() + pd.Timedelta(hours=mh["daily_break_hour"])
        brk_end = brk_start + break_max
        is_daily = (gap - interval) <= break_max + interval and missing_start < brk_end and missing_end > brk_start - interval

        is_weekend = prev_ny.weekday() == 4 and missing_end.weekday() in (6, 0) and gap <= weekend_max

        if is_daily:
            counts["daily_break"] += 1
        elif is_weekend:
            counts["weekend"] += 1
        else:
            counts["unexplained"] += 1
            rec = {
                "after": str(prev_ts),
                "before": str(next_ts),
                "gap_hours": round(gap.total_seconds() / 3600, 2),
                "missing_bars": int(gap / interval) - 1,
            }
            unexplained.append(rec)
            if gap > pd.Timedelta(days=5):
                long_holes.append(rec)

    unexplained_sorted = sorted(unexplained, key=lambda r: -r["gap_hours"])
    return {
        "counts": counts,
        "unexplained_examples": unexplained_sorted[:25],
        "long_holes": long_holes,
    }


def _detect_spikes(df: pd.DataFrame, n_atr: float, atr_period: int) -> dict[str, Any]:
    """Flag (a) close-to-close jumps and (b) isolated wicks > n_atr * prior ATR.

    ATR is computed from PRIOR bars only (shifted by 1) so the bar under test
    does not inflate its own threshold. Spikes are warnings, not fixed.
    """
    h, lo, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    tr = np.nanmax(np.vstack([h - lo, np.abs(h - prev_c), np.abs(lo - prev_c)]), axis=0)
    atr_prior = pd.Series(tr).rolling(atr_period, min_periods=atr_period).mean().shift(1).to_numpy()

    jump = np.abs(c - prev_c)
    jump_mask = np.nan_to_num(jump > n_atr * atr_prior, nan=False).astype(bool)

    next_c = np.roll(c, -1)
    next_c[-1] = np.nan
    up_wick = h - np.fmax(prev_c, next_c)
    dn_wick = np.fmin(prev_c, next_c) - lo
    wick_mask = np.nan_to_num((np.fmax(up_wick, dn_wick) > n_atr * atr_prior), nan=False).astype(bool)

    def examples(mask: np.ndarray, kind: str) -> list[dict[str, Any]]:
        idx = np.where(mask)[0][:15]
        return [{
            "timestamp": str(df["timestamp"].iloc[i]),
            "kind": kind,
            "close": float(c[i]), "high": float(h[i]), "low": float(lo[i]),
            "prior_atr": round(float(atr_prior[i]), 4),
        } for i in idx]

    return {
        "threshold_atr_multiple": n_atr,
        "close_jump_count": int(jump_mask.sum()),
        "isolated_wick_count": int(wick_mask.sum()),
        "examples": examples(jump_mask, "close_jump") + examples(wick_mask, "isolated_wick"),
    }


def validate_data(df: pd.DataFrame, timeframe: str, cfg: dict,
                  conversion_issues: dict[str, int] | None = None) -> dict[str, Any]:
    """Run every Section 3 check. Returns a JSON/Mongo-serialisable report."""
    vcfg = cfg["validation"]
    critical: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    info: dict[str, Any] = {}

    report: dict[str, Any] = {
        "timeframe": timeframe,
        "total_rows": int(len(df)),
        "critical_errors": critical,
        "warnings": warnings,
        "info": info,
    }
    if len(df) == 0:
        critical.append({"type": "empty_dataset"})
        return report

    if conversion_issues:
        for k, v in conversion_issues.items():
            if v:
                critical.append({"type": k, "count": int(v)})

    ts = df["timestamp"]
    valid_ts = ts.dropna()
    report["date_range_start"] = str(valid_ts.iloc[0]) if len(valid_ts) else None
    report["date_range_end"] = str(valid_ts.iloc[-1]) if len(valid_ts) else None

    # 1. Duplicates
    dup = ts.duplicated(keep=False) & ts.notna()
    if dup.any():
        critical.append({"type": "duplicate_timestamps", "count": int(ts[dup].nunique()),
                         "examples": [str(t) for t in ts[dup].unique()[:10]]})

    # 2. Out-of-order
    ooo = np.where(ts.diff() <= pd.Timedelta(0))[0]
    ooo = [i for i in ooo if not dup.iloc[i]]
    if ooo:
        critical.append({"type": "out_of_order_rows", "count": len(ooo),
                         "examples": [str(ts.iloc[i]) for i in ooo[:10]]})

    # 3. NaNs
    for col in REQUIRED_COLUMNS[1:]:
        n = int(df[col].isna().sum())
        if n:
            critical.append({"type": "nan_values", "column": col, "count": n})

    # 4. Zero / negative prices, negative volume
    for col in PRICE_COLUMNS:
        n = int((df[col] <= 0).sum())
        if n:
            critical.append({"type": "non_positive_price", "column": col, "count": n})
    n = int((df["volume"] < 0).sum())
    if n:
        critical.append({"type": "negative_volume", "count": n})
    zero_vol = int((df["volume"] == 0).sum())
    if zero_vol:
        warnings.append({"type": "zero_volume_bars", "count": zero_vol})

    # 5. High < Low ; 6. Open/Close outside [low, high]
    n = int((df["high"] < df["low"]).sum())
    if n:
        critical.append({"type": "high_less_than_low", "count": n,
                         "examples": [str(t) for t in ts[df["high"] < df["low"]].head(10)]})
    for col in ("open", "close"):
        bad = (df[col] > df["high"]) | (df[col] < df["low"])
        if bad.any():
            critical.append({"type": f"{col}_outside_high_low", "count": int(bad.sum()),
                             "examples": [str(t) for t in ts[bad].head(10)]})

    # 7. Grid alignment (bar opens must sit on the timeframe grid)
    minutes = TIMEFRAME_MINUTES[timeframe]
    mins_of_day = valid_ts.dt.hour * 60 + valid_ts.dt.minute
    misaligned = (mins_of_day % minutes != 0) | (valid_ts.dt.second != 0)
    if misaligned.any():
        critical.append({"type": "misaligned_bar_timestamps", "count": int(misaligned.sum()),
                         "examples": [str(t) for t in valid_ts[misaligned].head(10)]})

    # 8. Gaps (only meaningful when timestamps are clean)
    if not critical:
        gaps = _classify_gaps(valid_ts.reset_index(drop=True), timeframe, cfg)
        info["gaps"] = gaps["counts"]
        if gaps["counts"]["unexplained"]:
            warnings.append({"type": "unexplained_gaps", "count": gaps["counts"]["unexplained"],
                             "note": "holidays / early closes / feed outages; NOT filled",
                             "largest": gaps["unexplained_examples"]})
        if gaps["long_holes"]:
            critical.append({"type": "data_hole_over_5_days", "count": len(gaps["long_holes"]),
                             "examples": gaps["long_holes"][:10]})

        # 9. Spikes
        spikes = _detect_spikes(df, vcfg["spike_atr_threshold"], vcfg["atr_period"])
        if spikes["close_jump_count"] or spikes["isolated_wick_count"]:
            warnings.append({"type": "price_spikes", **spikes})

    info["price_min"] = float(df["close"].min())
    info["price_max"] = float(df["close"].max())
    info["volume_median"] = float(df["volume"].median())
    if len(valid_ts) > 1:
        info["years_covered"] = round((valid_ts.iloc[-1] - valid_ts.iloc[0]).total_seconds() / (365.25 * 86400), 3)
    report["passed"] = not critical
    return report


def cross_timeframe_consistency(low_df: pd.DataFrame, high_df: pd.DataFrame,
                                high_tf: str, tol: float = 1e-6) -> dict[str, Any]:
    """Compare native higher-TF bars against bars aggregated from a lower TF.

    Reported as information/warnings (vendor bars can legitimately differ at
    session boundaries). Used to decide how the Phase 3 `htf()` helper builds
    higher-timeframe bars.
    """
    rule = f"{TIMEFRAME_MINUTES[high_tf]}min"
    agg = (low_df.set_index("timestamp")
           .resample(rule, label="left", closed="left")
           .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
           .dropna())
    nat = high_df.set_index("timestamp")[["open", "high", "low", "close"]]
    joined = nat.join(agg, rsuffix="_agg", how="inner")
    out: dict[str, Any] = {"high_tf": high_tf, "compared_bars": int(len(joined))}
    for col in ("open", "high", "low", "close"):
        mism = (joined[col] - joined[f"{col}_agg"]).abs() > tol
        out[f"{col}_mismatch_pct"] = round(float(mism.mean()) * 100, 4) if len(joined) else None
    out["native_bars_without_lower_tf_data"] = int(len(nat) - len(joined))
    return out


# ─── Sessions ───────────────────────────────────────────────────────────────

SESSION_PRIORITY = ("overlap_london_ny", "new_york", "london", "asia")


def add_session_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add DST-correct session flags + a primary 'session' label (by bar open)."""
    out = df.copy()
    ts = out["timestamp"]
    for name, s in cfg["sessions"].items():
        local = ts.dt.tz_convert(s["tz"])
        hour = local.dt.hour + local.dt.minute / 60.0
        if s["start"] < s["end"]:
            mask = (hour >= s["start"]) & (hour < s["end"])
        else:
            mask = (hour >= s["start"]) | (hour < s["end"])
        out[f"in_{name}"] = mask.to_numpy()
    out["in_overlap_london_ny"] = out["in_london"] & out["in_new_york"]

    label = np.full(len(out), "off_hours", dtype=object)
    for name in reversed(SESSION_PRIORITY):  # lowest priority first, overwritten by higher
        label[out[f"in_{name}"].to_numpy()] = name
    out["session"] = label
    return out


def session_map_utc(cfg: dict, year: int = 2025) -> dict[str, Any]:
    """UTC hours of each session for a northern-summer and a winter reference day."""
    result: dict[str, Any] = {}
    for season, day in (("summer", f"{year}-07-15"), ("winter", f"{year}-01-15")):
        result[season] = {}
        for name, s in cfg["sessions"].items():
            start = pd.Timestamp(f"{day} {s['start']:02d}:00", tz=s["tz"]).tz_convert("UTC")
            end = pd.Timestamp(f"{day} {s['end']:02d}:00", tz=s["tz"]).tz_convert("UTC")
            result[season][name] = {"start_utc": start.strftime("%H:%M"), "end_utc": end.strftime("%H:%M")}
    result["note"] = ("Sessions are evaluated per bar in local market time; US and EU DST "
                      "switch on different dates, so in Mar/Oct-Nov the overlap shifts by 1h.")
    return result


# ─── Full pipeline ──────────────────────────────────────────────────────────

class DataValidationError(RuntimeError):
    """Raised when critical data errors are found (Section 3.3: stop and tell the user)."""

    def __init__(self, timeframe: str, report: dict[str, Any]):
        self.report = report
        lines = [f"CRITICAL data errors in {timeframe}:"]
        for e in report["critical_errors"]:
            lines.append(f"  - {e}")
        super().__init__("\n".join(lines))


def load_and_validate(timeframe: str, cfg: dict) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Load -> UTC -> validate. Returns (df, file_hash, report).

    The returned frame is in FILE ORDER (never re-sorted, never de-duplicated);
    if the file is out of order the report says so and we halt.
    Raises DataValidationError on critical errors when configured to halt.
    """
    raw, fhash = load_raw_data(timeframe, cfg)
    df, conv = convert_to_utc(raw, cfg["data"]["broker_tz"])
    for col in REQUIRED_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    report = validate_data(df, timeframe, cfg, conversion_issues=conv)
    report["file_hash"] = fhash
    report["file_name"] = data_path(timeframe, cfg).name
    report["broker_tz"] = cfg["data"]["broker_tz"]
    report["created_at"] = datetime.now(timezone.utc)

    if report["critical_errors"] and cfg["validation"].get("critical_errors_halt", True):
        raise DataValidationError(timeframe, report)
    df = df.reset_index(drop=True)
    df.attrs["file_hash"] = fhash
    df.attrs["timeframe"] = timeframe
    return df, fhash, report
