"""Canary 15: Pivot/swing used before it is confirmed.
Expected gate: truncation (unconfirmed future swing lookahead)
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class UnconfirmedPivotStrategy:
    """Deliberately marks a swing high at bar i by peeking ahead at i+1 and i+2."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        i = len(bars) - 1
        arr = bars['close'].to_numpy()
        base = arr.base
        if base is not None and hasattr(base, "shape"):
            n_rows = base.shape[1] if base.ndim > 1 else base.shape[0]
            if i + 2 < n_rows:
                # Peeks ahead 2 bars to confirm current bar is swing high
                c_now = bars['close'].iloc[-1]
                c_next1 = base[3, i + 1] if base.ndim > 1 else base[i + 1]
                c_next2 = base[3, i + 2] if base.ndim > 1 else base[i + 2]
                if c_now > c_next1 and c_now > c_next2:
                    return Signal(action=Action.ENTER_SHORT, sl_distance=10.0, tp_distance=20.0)
        return None
