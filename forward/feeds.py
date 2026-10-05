"""
forward/feeds.py – Pluggable closed-bar data feeds for forward paper testing.

Includes:
1. DataFeed base interface
2. SimulatedLiveFeed: sequential bar generator for testing and replay
3. CSVFolderFeed: directory watcher yielding closed bars as CSVs arrive
4. BrokerFeedStub: explicit stub for MT5 / broker WebSocket feeds
"""

from __future__ import annotations

import glob
import os
import time
from typing import Any, Generator, Iterator
import pandas as pd


class DataFeed:
    """Base interface for closed-bar market data feeds."""

    def has_next(self) -> bool:
        raise NotImplementedError

    def next_bar(self) -> dict[str, Any]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while self.has_next():
            yield self.next_bar()


class SimulatedLiveFeed(DataFeed):
    """Simulates real-time incremental arrival of closed bars from a DataFrame."""

    def __init__(self, df: pd.DataFrame, bar_delay_ms: float = 0.0):
        self._df = df.copy().reset_index(drop=True)
        self._cursor = 0
        self._bar_delay_ms = bar_delay_ms

    def has_next(self) -> bool:
        return self._cursor < len(self._df)

    def next_bar(self) -> dict[str, Any]:
        if not self.has_next():
            raise StopIteration("No more bars in simulated feed.")
        row = self._df.iloc[self._cursor].to_dict()
        self._cursor += 1
        if self._bar_delay_ms > 0:
            time.sleep(self._bar_delay_ms / 1000.0)
        return row

    def reset(self) -> None:
        self._cursor = 0


class CSVFolderFeed(DataFeed):
    """Watches a directory for newly deposited closed bar CSV files."""

    def __init__(self, folder_path: str, processed_dir: str | None = None):
        self.folder_path = folder_path
        self.processed_dir = processed_dir or os.path.join(folder_path, "processed")
        os.makedirs(self.folder_path, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._current_df: pd.DataFrame | None = None
        self._cursor = 0

    def _poll_new_file(self) -> str | None:
        csvs = sorted(glob.glob(os.path.join(self.folder_path, "*.csv")))
        return csvs[0] if csvs else None

    def has_next(self) -> bool:
        if self._current_df is not None and self._cursor < len(self._current_df):
            return True
        f = self._poll_new_file()
        if f:
            self._current_df = pd.read_csv(f)
            self._cursor = 0
            # Move file to processed
            target = os.path.join(self.processed_dir, os.path.basename(f))
            try:
                os.replace(f, target)
            except Exception as exc:
                import logging
                logging.getLogger("quantforge.forward").debug("Failed moving CSV file: %s", exc)
            return len(self._current_df) > 0
        return False

    def next_bar(self) -> dict[str, Any]:
        if not self.has_next() or self._current_df is None:
            raise StopIteration("No more bars in CSV folder.")
        row = self._current_df.iloc[self._cursor].to_dict()
        self._cursor += 1
        return row


class BrokerFeedStub(DataFeed):
    """Clearly marked stub for MT5 / broker API live streaming."""

    def __init__(self, broker_name: str = "MT5_Demo"):
        self.broker_name = broker_name
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> dict[str, str]:
        # Stubs MT5 initialization
        return {
            "status": "stub_mode",
            "broker": self.broker_name,
            "message": "BrokerFeedStub active. Connect real broker gateway when live.",
        }

    def has_next(self) -> bool:
        return False

    def next_bar(self) -> dict[str, Any]:
        raise NotImplementedError("BrokerFeedStub requires real broker connection.")
