import numpy as np
import pandas as pd

class Strategy:
    PARAMS = {
        "ema_period": {"default": 20, "min": 10, "max": 50},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        from core.indicators import previous_day_hl, ema, atr, is_session

        # Session filter: only trade during London or NY
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Indicators
        prev_high, prev_low = previous_day_hl(bars)
        ema_series = ema(bars, period=self.params["ema_period"])
        atr_series = atr(bars, period=self.params["atr_period"])

        # Current bar values
        ph_val = float(prev_high.iloc[-1])
        pl_val = float(prev_low.iloc[-1])
        c_val = float(bars["close"].iloc[-1])
        h_val = float(bars["high"].iloc[-1])
        l_val = float(bars["low"].iloc[-1])
        ema_val = float(ema_series.iloc[-1])
        atr_val = float(atr_series.iloc[-1])

        # Safety checks
        if np.isnan(ph_val) or np.isnan(pl_val) or np.isnan(ema_val) or np.isnan(atr_val) or atr_val <= 0:
            return None

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(7.5, float(atr_val * self.params["tp_mult"]))

        # Long setup: Sweep of Previous Day Low
        # Wick goes below yesterday's low, but closes back above it (rejection)
        # Must be in an uptrend context (close > EMA) to ensure we are catching a pullback sweep, not a breakdown
        if l_val < pl_val and c_val > pl_val and c_val > ema_val:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short setup: Sweep of Previous Day High
        # Wick goes above yesterday's high, but closes back below it (rejection)
        # Must be in a downtrend context (close < EMA)
        if h_val > ph_val and c_val < ph_val and c_val < ema_val:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None
