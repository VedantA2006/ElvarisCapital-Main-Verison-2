"""Canary 01: shift(-1) lookahead cheat.
Expected gate: static_policy
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class ShiftNegativeStrategy:
    """Deliberate lookahead cheat using close.shift(-1)."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        if len(bars) < 5:
            return None
        # Blatant lookahead: next bar's close
        future_close = bars['close'].shift(-1).iloc[-1]
        current_close = bars['close'].iloc[-1]
        if future_close > current_close:
            return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
        return None
