import numpy as np
import pandas as pd
from core.indicators import ema, atr, supertrend

PARAMS = {
    "fast_len": {"default": 21, "min": 5, "max": 50},
    "slow_len": {"default": 55, "min": 20, "max": 200},
    "atr_len": {"default": 14, "min": 5, "max": 30},
    "sl_mult": {"default": 1.5, "min": 0.5, "max": 4.0},
    "tp_mult": {"default": 2.5, "min": 1.0, "max": 6.0},
    "st_period": {"default": 10, "min": 5, "max": 20}
}

class Strategy:
    def __init__(self, params):
        self.params = params
        self.position = None

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        close = bars["close"].values
        high = bars["high"].values
        low = bars["low"].values

        fast_len = int(self.params["fast_len"])
        slow_len = int(self.params["slow_len"])
        atr_len = int(self.params["atr_len"])
        st_period = int(self.params["st_period"])

        fast_ema = ema(close, fast_len)
        slow_ema = ema(close, slow_len)
        atr_val = atr(high, low, close, atr_len)[-1]
        st_dir = supertrend(high, low, close, st_period, 3.0)[-1]

        if np.isnan(fast_ema[-1]) or np.isnan(slow_ema[-1]) or np.isnan(atr_val) or atr_val <= 0:
            return None

        # Require at least 2 closed bars to confirm trend direction safely without lookahead
        if np.isnan(fast_ema[-2]) or np.isnan(slow_ema[-2]):
            return None

        curr_fast = fast_ema[-1]
        prev_fast = fast_ema[-2]
        curr_slow = slow_ema[-1]
        prev_slow = slow_ema[-2]
        curr_close = close[-1]

        # Crossover detection (confirmed on closed bar)
        cross_up = (prev_fast <= prev_slow) and (curr_fast > curr_slow)
        cross_dn = (prev_fast >= prev_slow) and (curr_fast < curr_slow)

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(atr_val * self.params["tp_mult"]))

        # Long Entry: EMA Cross Up + Price above Slow EMA + Supertrend Bullish
        if cross_up and curr_close > curr_slow and st_dir == 1:
            self.position = "long"
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short Entry: EMA Cross Down + Price below Slow EMA + Supertrend Bearish
        if cross_dn and curr_close < curr_slow and st_dir == -1:
            self.position = "short"
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        # Exit Logic: Close position if the fast EMA crosses back against the position
        if self.position == "long" and cross_dn:
            self.position = None
            return Signal.close()

        if self.position == "short" and cross_up:
            self.position = None
            return Signal.close()

        return None