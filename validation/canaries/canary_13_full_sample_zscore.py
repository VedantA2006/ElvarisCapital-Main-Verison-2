"""Canary 13: Full-sample z-score using whole-series statistics.
Expected gate: Architectural guarantee – child process never receives the full series.
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class FullSampleZScoreStrategy:
    """Relies on seeing the entire dataset to compute global mean and std."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        # In a naive engine, bars is the whole series.
        # In QuantForge, len(bars) <= max_lookback_bars (e.g. 1500) and represents only past bars.
        global_mean = bars['close'].mean()
        global_std = bars['close'].std()
        if global_std > 0:
            z = (bars['close'].iloc[-1] - global_mean) / global_std
            if z < -2.0:
                return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
        return None
