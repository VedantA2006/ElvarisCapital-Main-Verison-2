import numpy as np
import pandas as pd

class Strategy:
    PARAMS = {
        "adx_period": {"default": 14, "min": 7, "max": 28},
        "adx_thresh": {"default": 22.0, "min": 15.0, "max": 35.0},
        "ema_period": {"default": 50, "min": 20, "max": 100},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.5}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        from core.indicators import macd, adx, atr, ema, is_session

        # Calculate indicators
        macd_res = macd(bars, fast_period=12, slow_period=26, signal_period=9)
        adx_res = adx(bars, period=self.params["adx_period"])
        atr_series = atr(bars, period=self.params["atr_period"])
        ema_series = ema(bars, period=self.params["ema_period"])
        session_mask = is_session(bars, "london") | is_session(bars, "ny")

        # Extract current and previous values safely
        hist_curr = float(macd_res.hist.iloc[-1])
        hist_prev = float(macd_res.hist.iloc[-2])
        adx_val = float(adx_res.adx_line.iloc[-1])
        plus_di = float(adx_res.plus_di.iloc[-1])
        minus_di = float(adx_res.minus_di.iloc[-1])
        atr_val = float(atr_series.iloc[-1])
        ema_val = float(ema_series.iloc[-1])
        close_val = float(bars["close"].iloc[-1])
        in_session = bool(session_mask.iloc[-1])

        # Safety checks
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        if np.isnan(adx_val) or np.isnan(hist_curr) or np.isnan(hist_prev):
            return None

        # Session filter to avoid Asian dead zones
        if not in_session:
            return None

        # Trend strength filter
        if adx_val < self.params["adx_thresh"]:
            return None

        # Distance calculations ensuring strictly positive USD points
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(7.5, float(atr_val * self.params["tp_mult"]))

        # Long Entry: Uptrend (+DI > -DI), price above EMA, MACD histogram crosses above 0
        if plus_di > minus_di and close_val > ema_val:
            if hist_prev <= 0.0 and hist_curr > 0.0:
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short Entry: Downtrend (-DI > +DI), price below EMA, MACD histogram crosses below 0
        if minus_di > plus_di and close_val < ema_val:
            if hist_prev >= 0.0 and hist_curr < 0.0:
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None