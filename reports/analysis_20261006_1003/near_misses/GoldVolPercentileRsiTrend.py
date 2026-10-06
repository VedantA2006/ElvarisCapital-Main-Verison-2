import numpy as np
import pandas as pd
from core.indicators import vol_percentile, rsi, ema, atr, adx, is_session

class Strategy:
    PARAMS = {
        "vol_lookback": {"default": 100, "min": 50, "max": 200},
        "vol_thresh": {"default": 30.0, "min": 15.0, "max": 50.0},
        "rsi_period": {"default": 14, "min": 7, "max": 21},
        "ema_period": {"default": 21, "min": 10, "max": 50},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -10

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Cooldown to prevent overtrading and ensure sample quality
        if len(bars) - self.last_trade_bar < 4:
            return None

        # Session Filter: London or NY
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Indicators
        vp = vol_percentile(bars, lookback=int(self.params["vol_lookback"]))
        r = rsi(bars, period=int(self.params["rsi_period"]))
        e = ema(bars, period=int(self.params["ema_period"]))
        a = atr(bars, period=14)
        adx_res = adx(bars, period=14)

        # Current and previous values
        vp_val = float(vp.iloc[-1])
        r_curr = float(r.iloc[-1])
        r_prev = float(r.iloc[-2])
        e_curr = float(e.iloc[-1])
        close_curr = float(bars["close"].iloc[-1])
        atr_val = float(a.iloc[-1])
        adx_val = float(adx_res.adx_line.iloc[-1])

        # Safety checks for NaNs
        if np.isnan(vp_val) or np.isnan(r_curr) or np.isnan(r_prev) or np.isnan(e_curr) or np.isnan(atr_val) or np.isnan(adx_val):
            return None
        if atr_val <= 0:
            return None

        # Core Logic Filters
        # 1. Volatility must be compressed (below threshold)
        if vp_val >= self.params["vol_thresh"]:
            return None

        # 2. Must have some baseline trend structure to avoid pure chop
        if adx_val < 20.0:
            return None

        # Distance calculations
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(7.5, float(atr_val * self.params["tp_mult"]))

        # Long Entry: RSI crosses above 50, Price > EMA
        if r_curr > 50.0 and r_prev <= 50.0 and close_curr > e_curr:
            self.last_trade_bar = len(bars)
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short Entry: RSI crosses below 50, Price < EMA
        if r_curr < 50.0 and r_prev >= 50.0 and close_curr < e_curr:
            self.last_trade_bar = len(bars)
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None