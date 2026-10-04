"""
core/splits.py – Chronological train / validation / holdout splits + holdout lock.

Design (conservative choices, documented in docs/METHODOLOGY.md):

1. Boundaries are CALENDAR-based and COMMON to all timeframes (computed on the
   intersection of all configured data files, including disabled ones), so a
   1h bar and a 4h bar at the same instant always belong to the same split,
   and enabling 5m/15m later cannot move the boundaries.
2. Boundaries are FROZEN in Mongo (`split_boundaries`) the first time they are
   computed. If new data is appended later, the extra bars fall AFTER the
   holdout ("post_holdout", reserved for forward testing) - they never shift
   holdout bars into train/validation. Changing ratios/embargo in config after
   freezing raises instead of silently re-splitting.
3. Embargo: `embargo_days` of bars between train|validation and
   validation|holdout belong to NO split.
4. Holdout: only via `DataStore.get_data(tf, "holdout", strategy_id=...)`,
   which writes `holdout_access` (unique index on strategy_id) BEFORE
   returning any data. A second call for the same strategy raises
   HoldoutLockError. If the process dies after the access is recorded the
   holdout is still considered burned (conservative).
5. Warm-up: `warmup_bars` may prepend bars from strictly EARLIER data (marked
   `is_warmup=True`; the backtester must never trade on them). Past data is not
   lookahead; the embargo exists so that positions/labels never straddle splits.
6. Disabled timeframes (Section 0C) cannot be served at all.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from pymongo.errors import DuplicateKeyError

from core.data_loader import (
    TIMEFRAME_MINUTES,
    add_session_labels,
    convert_to_utc,
    data_path,
    frame_sha256,
    load_and_validate,
)

VALID_SPLITS = ("train", "validation", "holdout")


class HoldoutLockError(PermissionError):
    """Raised on any second holdout access, or holdout access without strategy_id."""


class TimeframeDisabledError(PermissionError):
    """Raised when a timeframe not in enabled_timeframes is requested."""


class SplitConfigChangedError(RuntimeError):
    """Raised if config split settings differ from the frozen boundaries."""


@dataclass(frozen=True)
class SplitBoundaries:
    data_start: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp          # exclusive
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp     # exclusive
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp        # exclusive (bars opening at/after are post_holdout)
    train_ratio: float
    validation_ratio: float
    holdout_ratio: float
    embargo_days: float

    def range_for(self, split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        if split == "train":
            return self.train_start, self.train_end
        if split == "validation":
            return self.validation_start, self.validation_end
        if split == "holdout":
            return self.holdout_start, self.holdout_end
        raise ValueError(f"Unknown split '{split}'. Valid: {VALID_SPLITS}")

    def to_doc(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: (v.to_pydatetime() if isinstance(v, pd.Timestamp) else v) for k, v in d.items()}

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> "SplitBoundaries":
        kw = {}
        for f in cls.__dataclass_fields__:
            v = doc[f]
            if isinstance(v, datetime):
                v = pd.Timestamp(v)
                v = v.tz_localize("UTC") if v.tzinfo is None else v.tz_convert("UTC")
            kw[f] = v
        return cls(**kw)


def compute_boundaries(data_start: pd.Timestamp, data_end: pd.Timestamp, splits_cfg: dict) -> SplitBoundaries:
    """Pure function: chronological 60/20/20 (configurable) with embargo.

    data_end is EXCLUSIVE (close time of the last bar). Internal boundaries are
    floored to 00:00 UTC so every timeframe's bar grid aligns with them.
    """
    if data_end <= data_start:
        raise ValueError("data_end must be after data_start")
    span = data_end - data_start
    embargo = pd.Timedelta(days=splits_cfg["embargo_days"])
    tr, va = splits_cfg["train_ratio"], splits_cfg["validation_ratio"]

    train_end = (data_start + span * tr).floor("D")
    validation_start = train_end + embargo
    validation_end = (data_start + span * (tr + va)).floor("D")
    holdout_start = validation_end + embargo

    if not (data_start < train_end < validation_start < validation_end < holdout_start < data_end):
        raise ValueError("Data range too short for the configured splits + embargo")

    return SplitBoundaries(
        data_start=data_start, train_start=data_start, train_end=train_end,
        validation_start=validation_start, validation_end=validation_end,
        holdout_start=holdout_start, holdout_end=data_end,
        train_ratio=tr, validation_ratio=va, holdout_ratio=splits_cfg["holdout_ratio"],
        embargo_days=float(splits_cfg["embargo_days"]),
    )


def _file_time_range(timeframe: str, cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First bar open and last bar CLOSE (open + interval) in UTC, read cheaply."""
    path = data_path(timeframe, cfg)
    with open(path, "rb") as f:
        f.readline()                     # header
        first = f.readline().decode().split(",")[0]
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4096))
        last = [ln for ln in f.read().decode().splitlines() if ln.strip()][-1].split(",")[0]
    tmp = pd.DataFrame({"timestamp": [first, last]})
    conv, _ = convert_to_utc(tmp, cfg["data"]["broker_tz"])
    start, last_open = conv["timestamp"].iloc[0], conv["timestamp"].iloc[1]
    return start, last_open + pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])


def common_data_range(cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Intersection of the time ranges of ALL configured data files."""
    starts, ends = [], []
    for tf in cfg["data"]["files"]:
        if data_path(tf, cfg).exists():
            s, e = _file_time_range(tf, cfg)
            starts.append(s)
            ends.append(e)
    if not starts:
        raise FileNotFoundError("No data files found")
    return max(starts), min(ends)


# ─── Frozen boundary persistence ───────────────────────────────────────────

def get_or_freeze_boundaries(cfg: dict, collection=None) -> SplitBoundaries:
    """Return frozen boundaries from Mongo, freezing them on first call."""
    if collection is None:
        from storage.mongo import get_db
        collection = get_db()["split_boundaries"]

    symbol = cfg["data"]["symbol"]
    doc = collection.find_one({"_id": symbol})
    s = cfg["splits"]
    if doc is None:
        start, end = common_data_range(cfg)
        b = compute_boundaries(start, end, s)
        try:
            collection.insert_one({"_id": symbol, **b.to_doc(),
                                   "frozen_at": datetime.now(timezone.utc)})
        except DuplicateKeyError:
            doc = collection.find_one({"_id": symbol})  # another process froze first
        else:
            return b

    b = SplitBoundaries.from_doc(doc)
    for key, cfg_key in (("train_ratio", "train_ratio"), ("validation_ratio", "validation_ratio"),
                         ("holdout_ratio", "holdout_ratio"), ("embargo_days", "embargo_days")):
        if abs(float(getattr(b, key)) - float(s[cfg_key])) > 1e-12:
            raise SplitConfigChangedError(
                f"config.splits.{cfg_key}={s[cfg_key]} differs from frozen value {getattr(b, key)}. "
                "Re-splitting would leak holdout data. Refusing.")
    return b


# ─── DataStore: the ONLY split-aware data access path ──────────────────────

class DataStore:
    """Serves split data. The holdout is only reachable through get_data()."""

    def __init__(self, cfg: dict, boundaries: SplitBoundaries | None = None, holdout_recorder=None):
        self._cfg = cfg
        self._boundaries = boundaries or get_or_freeze_boundaries(cfg)
        self._frames: dict[str, pd.DataFrame] = {}
        self._file_hashes: dict[str, str] = {}
        if holdout_recorder is None:
            from storage.mongo import record_holdout_access
            holdout_recorder = record_holdout_access
        self._record_holdout = holdout_recorder

    @property
    def boundaries(self) -> SplitBoundaries:
        return self._boundaries

    def _check_tf(self, timeframe: str) -> None:
        if timeframe not in self._cfg["data"]["enabled_timeframes"]:
            raise TimeframeDisabledError(
                f"Timeframe '{timeframe}' is disabled (enabled: {self._cfg['data']['enabled_timeframes']}). "
                "See Section 0C.")

    def _frame(self, timeframe: str) -> pd.DataFrame:
        if timeframe not in self._frames:
            df, fhash, _ = load_and_validate(timeframe, self._cfg)
            df = add_session_labels(df, self._cfg)
            self._frames[timeframe] = df
            self._file_hashes[timeframe] = fhash
        return self._frames[timeframe]

    def split_info(self, timeframe: str) -> dict[str, Any]:
        """Boundaries + bar counts (no prices). Safe for dashboards, NOT for LLM prompts."""
        self._check_tf(timeframe)
        df = self._frame(timeframe)
        out: dict[str, Any] = {"timeframe": timeframe, "file_hash": self._file_hashes[timeframe]}
        for split in VALID_SPLITS:
            lo, hi = self._boundaries.range_for(split)
            out[split] = {"start": str(lo), "end_exclusive": str(hi),
                          "bars": int(((df["timestamp"] >= lo) & (df["timestamp"] < hi)).sum())}
        out["post_holdout_bars"] = int((df["timestamp"] >= self._boundaries.holdout_end).sum())
        return out

    def get_data(self, timeframe: str, split: str, *, strategy_id: str | None = None,
                 warmup_bars: int = 0, purpose: str = "") -> pd.DataFrame:
        """Return a COPY of the requested split.

        train / validation: free access.
        holdout: requires strategy_id; access is recorded BEFORE data is returned
                 and a second access for that strategy raises HoldoutLockError.
        """
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Valid: {VALID_SPLITS}")
        self._check_tf(timeframe)
        if warmup_bars < 0:
            raise ValueError("warmup_bars must be >= 0")

        if split == "holdout":
            if not strategy_id or not isinstance(strategy_id, str):
                raise HoldoutLockError("Holdout access requires a non-empty strategy_id.")
            df_full = self._frame(timeframe)   # load/validate first so a data error doesn't burn access
            try:
                self._record_holdout(strategy_id, timeframe=timeframe, purpose=purpose,
                                     file_hash=self._file_hashes[timeframe])
            except PermissionError as exc:
                raise HoldoutLockError(str(exc)) from exc
        else:
            df_full = self._frame(timeframe)

        lo, hi = self._boundaries.range_for(split)
        ts = df_full["timestamp"]
        in_split = (ts >= lo) & (ts < hi)
        idx = in_split.to_numpy().nonzero()[0]
        if len(idx) == 0:
            raise ValueError(f"No bars in split '{split}' for {timeframe}")
        first = int(idx[0])
        start = max(0, first - warmup_bars)

        out = df_full.iloc[start: int(idx[-1]) + 1].copy()
        out["is_warmup"] = (out["timestamp"] < lo).to_numpy()
        out = out.reset_index(drop=True)
        core = out.loc[~out["is_warmup"]]
        out.attrs = {
            "timeframe": timeframe,
            "split": split,
            "file_hash": self._file_hashes[timeframe],
            "slice_hash": frame_sha256(core),
            "split_start": str(lo),
            "split_end_exclusive": str(hi),
            "warmup_bars": int(out["is_warmup"].sum()),
        }
        return out
