import numpy as np
import pandas as pd
from core.indicators import premium_discount, macd, ema, atr

class Strategy:
    PARAMS = {
        "ema_period": {"default": 21, "min": 10, "max": 50},
        "swing_n": {"default": 5, "min": 3, "max": 15},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_rr": {"default": 1.5, "min": 1.0, "max": 3.0},
        "cooldown": {"default": 3, "min": 1, "max": 8}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999

    def on_bar(self, bars: pd.DataFrame):
        if len(bars) < 60:
            return None

        # Enforce cooldown to prevent overtrading and ensure statistical independence
        if len(bars) - self.last_trade_bar <= self.params["cooldown"]:
            return None

        ema_period = int(self.params["ema_period"])
        swing_n = int(self.params["swing_n"])

        # Calculate indicators
        pd_series = premium_discount(bars, swing_n=swing_n)
        macd_line, signal_line, hist = macd(bars, 12, 26, 9)
        ema_series = ema(bars, ema_period)
        atr_series = atr(bars, 14)

        # Extract current values safely
        pd_val = float(pd_series.iloc[-1])
        hist_val = float(hist.iloc[-1])
        ema_val = float(ema_series.iloc[-1])
        atr_val = float(atr_series.iloc[-1])
        close_val = float(bars["close"].iloc[-1])

        # Safety checks for NaNs
        if np.isnan(pd_val) or np.isnan(hist_val) or np.isnan(ema_val) or np.isnan(atr_val):
            return None
        if atr_val <= 0:
            return None

        # Calculate distances
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(sl_dist * self.params["tp_rr"]))

        # Long Entry: Discount zone + Bullish Momentum + Above EMA
        if pd_val < 0.45 and hist_val > 0 and close_val > ema_val:
            self.last_trade_bar = len(bars)
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short Entry: Premium zone + Bearish Momentum + Below EMA
        if pd_val > 0.55 and hist_val < 0 and close_val < ema_val:
            self.last_trade_bar = len(bars)
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None