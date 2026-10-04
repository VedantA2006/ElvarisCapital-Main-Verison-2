"""Canary 05: .to_numpy().base future peek (LH-1).
Expected gate: static_policy (and sandbox gives no parent buffer)
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class NumpyBaseStrategy:
    """Attempts to inspect numpy parent buffer via .base attribute."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        if len(bars) < 5:
            return None
        arr = bars['close'].to_numpy()
        parent = arr.base
        if parent is not None and len(parent) > len(bars):
            return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
        return None
