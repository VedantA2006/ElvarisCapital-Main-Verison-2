"""Control 01: Simple Moving Average crossover.
Clean, honest, past-only trend following strategy.
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class SmaCrossControl:
    """Fast SMA (10) crosses Slow SMA (30)."""
    PARAMS = {
        "fast_period": {"default": 10, "min": 5, "max": 25},
        "slow_period": {"default": 30, "min": 20, "max": 60},
    }

    def __init__(self, params=None):
        params = params or {}
        self.fast_period = int(params.get("fast_period", 10))
        self.slow_period = int(params.get("slow_period", 30))

    def on_bar(self, bars):
        if len(bars) < self.slow_period + 2:
            return None

        closes = bars['close']
        fast = closes.rolling(self.fast_period).mean()
        slow = closes.rolling(self.slow_period).mean()

        f_now, f_prev = fast.iloc[-1], fast.iloc[-2]
        s_now, s_prev = slow.iloc[-1], slow.iloc[-2]

        if f_prev <= s_prev and f_now > s_now:
            return Signal(action=Action.ENTER_LONG, sl_distance=15.0, tp_distance=30.0)
        elif f_prev >= s_prev and f_now < s_now:
            return Signal(action=Action.ENTER_SHORT, sl_distance=15.0, tp_distance=30.0)

        return None
