"""Canary 04: j = i + 1; df.iloc[j] variable indexing cheat.
Expected gate: static_policy (or truncation)
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class IlocVariableStrategy:
    """Deliberate variable-indexed iloc access to peek ahead."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        if len(bars) < 5:
            return None
        i = len(bars) - 1
        j = i + 1
        val = bars['close'].iloc[j]
        if val > bars['close'].iloc[-1]:
            return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
        return None
