import numpy as np
import pandas as pd

class Strategy:
    PARAMS = {
        "adx_period": {"default": 14, "min": 7, "max": 21},
        "donchian_period": {"default": 16, "min": 8, "max": 30},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "adx_thresh": {"default": 22.0, "min": 15.0, "max": 35.0},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 5.0}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Cooldown to prevent overtrading and ensure minimum sample quality
        if len(bars) - self.last_trade_bar < 3:
            return None

        from core.indicators import adx, donchian, atr, is_session

        # Session filter: only trade during London or NY
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Calculate indicators
        adx_res = adx(bars, period=self.params["adx_period"])
        donch_res = donchian(bars, period=self.params["donchian_period"])
        atr_series = atr(bars, period=self.params["atr_period"])

        # Extract current values
        adx_val = float(adx_res.adx_line.iloc[-1])
        plus_di = float(adx_res.plus_di.iloc[-1])
        minus_di = float(adx_res.minus_di.iloc[-1])
        atr_val = float(atr_series.iloc[-1])
        close_curr = float(bars["close"].iloc[-1])

        # Use PREVIOUS bar's Donchian levels to avoid any lookahead bias
        donch_upper_prev = float(donch_res.upper.iloc[-2])
        donch_lower_prev = float(donch_res.lower.iloc[-2])

        # Safety checks for NaNs and valid ATR
        if np.isnan(adx_val) or np.isnan(atr_val) or atr_val <= 0:
            return None
        if np.isnan(donch_upper_prev) or np.isnan(donch_lower_prev):
            return None

        # Trend strength filter
        if adx_val < self.params["adx_thresh"]:
            return None

        # Calculate SL and TP distances
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(7.5, float(atr_val * self.params["tp_mult"]))

        # Long Entry: Bullish DI cross + Breakout above previous Donchian Upper
        if plus_di > minus_di and close_curr > donch_upper_prev:
            self.last_trade_bar = len(bars)
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short Entry: Bearish DI cross + Breakout below previous Donchian Lower
        if minus_di > plus_di and close_curr < donch_lower_prev:
            self.last_trade_bar = len(bars)
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None