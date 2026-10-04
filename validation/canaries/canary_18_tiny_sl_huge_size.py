"""Canary 18: sl_distance = 0.01 microscopic stop (BT-14).
Expected gate: signal validation (sl_distance < min_sl_usd)
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class TinySlStrategy:
    """Attempts to place a 1-cent stop distance to game position sizing (BT-14 exploit)."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        if len(bars) < 5:
            return None
        # Microscopic stop distance $0.01 vs min required $1.00
        return Signal(action=Action.ENTER_LONG, sl_distance=0.01, tp_distance=10.0)
