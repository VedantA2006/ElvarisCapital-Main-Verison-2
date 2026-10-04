"""Canary 19: Take-profit smaller than spread scalp.
Expected gate: signal validation (tp_distance < min_tp_spread_multiple * spread)
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class TpSmallerSpreadStrategy:
    """Attempts to scalp a take profit smaller than the transaction spread."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        if len(bars) < 5:
            return None
        # TP of $0.05 is far smaller than London spread ($0.25)
        return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=0.05)
