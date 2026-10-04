"""Shared pytest fixtures. Tests ALWAYS use the test Mongo database.

Mongo backend for tests is chosen by the QF_TEST_MONGO environment variable:
    QF_TEST_MONGO=mock  (default)  in-memory mongomock, no server needed
    QF_TEST_MONGO=real             the real server from MONGO_URL, database
                                   `mongo.test_database` (must end in _test)
"""

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

QF_TEST_MONGO = os.environ.get("QF_TEST_MONGO", "mock").strip().lower()
if QF_TEST_MONGO not in ("mock", "real"):
    raise RuntimeError(f"QF_TEST_MONGO must be 'mock' or 'real', got {QF_TEST_MONGO!r}")


def _install_mock_client() -> None:
    """Point storage.mongo's singleton at a brand-new in-memory client."""
    import mongomock
    from pymongo.operations import IndexModel
    orig_create_indexes = mongomock.collection.Collection.create_indexes

    def _patched_create_indexes(self, indexes, session=None):
        for index in indexes:
            if not isinstance(index, IndexModel):
                raise TypeError(f"{index} is not an instance of pymongo.operations.IndexModel")
        return [
            self.create_index(
                list(index.document['key'].items()),
                session=session,
                expireAfterSeconds=index.document.get('expireAfterSeconds'),
                unique=index.document.get('unique', False),
                sparse=index.document.get('sparse', False),
                name=index.document.get('name'))
            for index in indexes
        ]

    mongomock.collection.Collection.create_indexes = _patched_create_indexes
    from storage import mongo
    mongo._client = mongomock.MongoClient()
    mongo._db = None


# Install the mock BEFORE any test module imports code that calls get_client(),
# so no test can reach a real server by accident in mock mode.
if QF_TEST_MONGO == "mock":
    _install_mock_client()


def _reset_test_db():
    from storage import mongo
    db = mongo.get_db()
    assert db.name.endswith("_test"), f"Refusing to use non-test DB {db.name}"
    mongo.get_client().drop_database(db.name)
    mongo.ensure_indexes()
    return db


@pytest.fixture(scope="session")
def base_cfg() -> dict:
    return copy.deepcopy(_BASE_CFG)


@pytest.fixture(scope="session")
def test_db():
    """Clean test database (dropped at session start). Refuses to touch non-test DBs."""
    yield _reset_test_db()


@pytest.fixture
def fresh_db():
    """A clean test database for ONE test (function scope, full isolation)."""
    yield _reset_test_db()


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
