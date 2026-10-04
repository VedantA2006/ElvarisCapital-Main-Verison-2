"""Control 05: ATR Volatility Filter Trend Strategy.
Clean, honest, past-only volatility filtered trend strategy.
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class AtrVolFilterControl:
    """Trend entry only when 14-bar ATR exceeds its 20-bar baseline."""
    PARAMS = {
        "atr_period": {"default": 14, "min": 7, "max": 28},
        "trend_period": {"default": 20, "min": 10, "max": 40},
    }

    def __init__(self, params=None):
        params = params or {}
        self.atr_period = int(params.get("atr_period", 14))
        self.trend_period = int(params.get("trend_period", 20))

    def on_bar(self, bars):
        min_bars = self.trend_period + self.atr_period + 5
        if len(bars) < min_bars:
            return None

        high = bars['high']
        low = bars['low']
        close = bars['close']

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        atr_baseline = atr.rolling(self.trend_period).mean()
        sma = close.rolling(self.trend_period).mean()

        current_atr = atr.iloc[-1]
        baseline = atr_baseline.iloc[-1]
        c = close.iloc[-1]
        m = sma.iloc[-1]

        # Only enter if volatility is elevated above baseline
        if current_atr > baseline:
            if c > m:
                return Signal(action=Action.ENTER_LONG, sl_distance=max(5.0, current_atr * 1.5), tp_distance=max(10.0, current_atr * 3.0))
            elif c < m:
                return Signal(action=Action.ENTER_SHORT, sl_distance=max(5.0, current_atr * 1.5), tp_distance=max(10.0, current_atr * 3.0))

        return None
