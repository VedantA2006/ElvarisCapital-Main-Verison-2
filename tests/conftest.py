"""Shared pytest fixtures. Tests ALWAYS use the test Mongo database."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import load_config, load_env  # noqa: E402

load_env()
_BASE_CFG = load_config()
os.environ["QF_MONGO_DB"] = _BASE_CFG["mongo"]["test_database"]


@pytest.fixture(scope="session")
def base_cfg() -> dict:
    return copy.deepcopy(_BASE_CFG)


@pytest.fixture(scope="session")
def test_db():
    """Clean test database (dropped at session start). Refuses to touch non-test DBs."""
    from storage import mongo
    db = mongo.get_db()
    assert db.name.endswith("_test"), f"Refusing to use non-test DB {db.name}"
    mongo.get_client().drop_database(db.name)
    mongo.ensure_indexes()
    yield db


# ─── Synthetic market data on a realistic XAUUSD calendar ──────────────────

def market_open(ts_utc: pd.DatetimeIndex) -> np.ndarray:
    """XAUUSD open: Sun 18:00 -> Fri 17:00 New York, closed 17:00-18:00 NY daily."""
    ny = ts_utc.tz_convert("America/New_York")
    wd, hr = ny.weekday, ny.hour
    closed = (wd == 5) | ((wd == 4) & (hr >= 17)) | ((wd == 6) & (hr < 18)) | (hr == 17)
    return ~np.asarray(closed)


def make_bars(start: str, end: str, freq: str = "60min", seed: int = 0, price: float = 1800.0) -> pd.DataFrame:
    """Random-walk OHLCV bars, UTC, only during market hours. Internally consistent."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")
    idx = idx[market_open(idx)]
    n = len(idx)
    rets = rng.normal(0, 0.002, n)
    close = price * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = price
    open_[1:] = close[:-1]
    span = np.abs(rng.normal(0, 0.0015, n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    vol = rng.uniform(0.5, 3.0, n)
    return pd.DataFrame({"timestamp": idx, "open": open_, "high": high,
                         "low": low, "close": close, "volume": vol})


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write in the same naive 'YYYY-mm-dd HH:MM:SS' format as the real files."""
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    out["timestamp"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)


@pytest.fixture
def synth_cfg(tmp_path, base_cfg):
    """Config pointing at a temp data dir with synthetic 1h and 4h files (3 years)."""
    h1 = make_bars("2021-01-04", "2024-01-01", "60min", seed=1)
    h4 = (h1.set_index("timestamp").resample("240min", label="left", closed="left")
          .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
          .dropna().reset_index())
    write_csv(h1, tmp_path / "H1.csv")
    write_csv(h4, tmp_path / "H4.csv")
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["base_dir_resolved"] = str(tmp_path)
    cfg["data"]["files"] = {"1h": "H1.csv", "4h": "H4.csv"}
    cfg["data"]["all_timeframes"] = ["1h", "4h"]
    cfg["data"]["enabled_timeframes"] = ["1h", "4h"]
    cfg["data"]["symbol"] = f"TEST_{tmp_path.name}"
    return cfg
