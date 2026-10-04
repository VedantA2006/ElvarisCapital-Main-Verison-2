"""Control 02: RSI Mean Reversion.
Clean, honest, past-only mean reversion strategy.
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class RsiMeanReversionControl:
    """14-period RSI: buy below 30, sell above 70."""
    PARAMS = {
        "period": {"default": 14, "min": 7, "max": 28},
        "oversold": {"default": 30.0, "min": 20.0, "max": 40.0},
        "overbought": {"default": 70.0, "min": 60.0, "max": 80.0},
    }

    def __init__(self, params=None):
        params = params or {}
        self.period = int(params.get("period", 14))
        self.oversold = float(params.get("oversold", 30.0))
        self.overbought = float(params.get("overbought", 70.0))

    def on_bar(self, bars):
        if len(bars) < self.period + 5:
            return None

        closes = bars['close']
        delta = closes.diff()
        gain = delta.clip(lower=0.0).rolling(self.period).mean()
        loss = (-delta.clip(upper=0.0)).rolling(self.period).mean()

        l_val = loss.iloc[-1]
        g_val = gain.iloc[-1]
        if l_val == 0:
            rsi = 100.0
        else:
            rs = g_val / l_val
            rsi = 100.0 - (100.0 / (1.0 + rs))

        if rsi < self.oversold:
            return Signal(action=Action.ENTER_LONG, sl_distance=12.0, tp_distance=24.0)
        elif rsi > self.overbought:
            return Signal(action=Action.ENTER_SHORT, sl_distance=12.0, tp_distance=24.0)

        return None
