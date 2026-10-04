"""Canary 03: rolling(5, center=True) lookahead cheat.
Expected gate: static_policy
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class RollingCenterStrategy:
    """Deliberate lookahead cheat using center=True."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        if len(bars) < 10:
            return None
        centered_mean = bars['close'].rolling(5, center=True).mean()
        if bars['close'].iloc[-1] > centered_mean.iloc[-1]:
            return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
        return None
