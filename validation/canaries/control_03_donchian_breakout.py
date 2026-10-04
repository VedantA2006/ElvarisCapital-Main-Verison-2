"""Control 03: Donchian Channel Breakout.
Clean, honest, past-only breakout strategy.
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class DonchianBreakoutControl:
    """20-bar Donchian channel breakout on prior bars' high/low."""
    PARAMS = {
        "channel_period": {"default": 20, "min": 10, "max": 50},
    }

    def __init__(self, params=None):
        params = params or {}
        self.period = int(params.get("channel_period", 20))

    def on_bar(self, bars):
        if len(bars) < self.period + 2:
            return None

        # Exclude current bar to avoid lookahead: use bars[-period-1 : -1]
        prior_highs = bars['high'].iloc[-self.period - 1 : -1]
        prior_lows = bars['low'].iloc[-self.period - 1 : -1]

        upper = prior_highs.max()
        lower = prior_lows.min()
        close = bars['close'].iloc[-1]

        if close > upper:
            return Signal(action=Action.ENTER_LONG, sl_distance=15.0, tp_distance=30.0)
        elif close < lower:
            return Signal(action=Action.ENTER_SHORT, sl_distance=15.0, tp_distance=30.0)

        return None
