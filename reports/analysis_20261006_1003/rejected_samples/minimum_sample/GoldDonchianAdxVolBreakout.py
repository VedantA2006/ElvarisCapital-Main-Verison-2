import numpy as np
import pandas as pd
from core.indicators import donchian, adx, atr, is_session

class Strategy:
    PARAMS = {
        "donchian_period": {"default": 20, "min": 10, "max": 40},
        "adx_period": {"default": 14, "min": 7, "max": 21},
        "adx_thresh": {"default": 20.0, "min": 15.0, "max": 35.0},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999
        self.last_direction = None

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Cooldown to prevent overtrading and allow structure to develop
        current_bar_idx = len(bars) - 1
        if (current_bar_idx - self.last_trade_bar) < 3:
            return None

        # Session filter: only trade during London or NY for liquidity
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Indicators
        d_period = int(self.params["donchian_period"])
        a_period = int(self.params["adx_period"])
        atr_period = int(self.params["atr_period"])

        dc = donchian(bars, period=d_period)
        adx_res = adx(bars, period=a_period)
        atr_series = atr(bars, period=atr_period)

        close_curr = float(bars["close"].iloc[-1])
        
        # Use PREVIOUS bar's Donchian bounds to avoid look-ahead bias on the current forming/closed bar
        dc_upper_prev = float(dc.upper.iloc[-2])
        dc_lower_prev = float(dc.lower.iloc[-2])

        adx_val = float(adx_res.adx_line.iloc[-1])
        plus_di = float(adx_res.plus_di.iloc[-1])
        minus_di = float(adx_res.minus_di.iloc[-1])
        atr_val = float(atr_series.iloc[-1])

        # Safety checks for NaNs
        if np.isnan(adx_val) or np.isnan(atr_val) or np.isnan(dc_upper_prev) or np.isnan(dc_lower_prev):
            return None
        if atr_val <= 0:
            return None

        # Calculate SL and TP distances
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(7.5, float(atr_val * self.params["tp_mult"]))

        # Trend strength filter
        if adx_val < self.params["adx_thresh"]:
            return None

        # Long Entry: Breakout above previous upper Donchian + Bullish DI cross
        if close_curr > dc_upper_prev and plus_di > minus_di:
            self.last_trade_bar = current_bar_idx
            self.last_direction = "long"
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short Entry: Breakout below previous lower Donchian + Bearish DI cross
        if close_curr < dc_lower_prev and minus_di > plus_di:
            self.last_trade_bar = current_bar_idx
            self.last_direction = "short"
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None