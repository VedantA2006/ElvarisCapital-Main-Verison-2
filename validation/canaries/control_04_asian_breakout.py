"""Control 04: Asian-range Breakout.
Clean, honest, past-only session breakout strategy.
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class AsianBreakoutControl:
    """Breakout of Asian session range during London open."""
    PARAMS = {
        "sl_mult": {"default": 1.0, "min": 0.5, "max": 2.0},
    }

    def __init__(self, params=None):
        params = params or {}
        self.sl_mult = float(params.get("sl_mult", 1.0))

    def on_bar(self, bars):
        if len(bars) < 24:
            return None

        # Look back up to 24 bars for the most recent completed Asia session
        sessions = bars['session'] if 'session' in bars.columns else pd.Series(['unknown'] * len(bars))
        current_session = str(sessions.iloc[-1]).lower()

        # Only trade during London session
        if current_session != "london":
            return None

        # Find recent Asia bars in the past 24 bars excluding current bar
        recent_bars = bars.iloc[-25:-1]
        recent_sess = sessions.iloc[-25:-1].astype(str).str.lower()
        asia_bars = recent_bars[recent_sess == "asia"]

        if len(asia_bars) < 3:
            return None

        asia_high = asia_bars['high'].max()
        asia_low = asia_bars['low'].min()
        close = bars['close'].iloc[-1]
        rng = max(5.0, asia_high - asia_low)

        if close > asia_high:
            return Signal(action=Action.ENTER_LONG, sl_distance=rng * self.sl_mult, tp_distance=rng * 2.0)
        elif close < asia_low:
            return Signal(action=Action.ENTER_SHORT, sl_distance=rng * self.sl_mult, tp_distance=rng * 2.0)

        return None
