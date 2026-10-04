"""Canary 06: pd.read_csv price file read (SBX-2).
Expected gate: audit hook / sandbox_error (or compile_smoke)
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class CsvReadStrategy:
    """Attempts to read price data directly from disk."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        df_all = pd.read_csv("XAUUSD_1h.csv")
        return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
