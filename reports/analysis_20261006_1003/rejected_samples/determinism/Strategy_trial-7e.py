import numpy as np
import pandas as pd
from core.indicators import ema, atr, supertrend

PARAMS = {
    "fast_len": {"default": 12, "min": 5, "max": 30},
    "slow_len": {"default": 26, "min": 15, "max": 60},
    "atr_period": {"default": 14, "min": 7, "max": 30},
    "sl_mult": {"default": 1.5, "min": 0.5, "max": 4.0},
    "tp_mult": {"default": 2.5, "min": 1.5, "max": 6.0},
    "st_mult": {"default": 3.0, "min": 1.0, "max": 5.0}
}

class Strategy:
    def __init__(self, params):
        self.params = params
        self.position = 0

    def on_bar(self, bars: pd.DataFrame):
        if len(bars) < 60:
            return None

        close = bars["close"].values.astype(float)
        
        fast_len = int(self.params["fast_len"])
        slow_len = int(self.params["slow_len"])
        atr_period = int(self.params["atr_period"])
        st_mult = float(self.params["st_mult"])

        fast_ema = ema(close, fast_len)
        slow_ema = ema(close, slow_len)
        atr_val = atr(bars, atr_period)[-1]
        st_dir, _, _ = supertrend(bars, period=10, multiplier=st_mult)

        if np.isnan(atr_val) or atr_val <= 0:
            return None
        if np.isnan(fast_ema[-1]) or np.isnan(slow_ema[-1]):
            return None
        if np.isnan(fast_ema[-2]) or np.isnan(slow_ema[-2]):
            return None

        curr_fast = fast_ema[-1]
        prev_fast = fast_ema[-2]
        curr_slow = slow_ema[-1]
        prev_slow = slow_ema[-2]
        curr_st_dir = st_dir[-1]

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(atr_val * self.params["tp_mult"]))

        # Long entry: EMA bullish crossover + Supertrend bullish
        if prev_fast <= prev_slow and curr_fast > curr_slow and curr_st_dir == 1:
            if self.position != 1:
                self.position = 1
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short entry: EMA bearish crossover + Supertrend bearish
        if prev_fast >= prev_slow and curr_fast < curr_slow and curr_st_dir == -1:
            if self.position != -1:
                self.position = -1
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        # Exit conditions: EMA crosses back against the position
        if self.position == 1 and curr_fast < curr_slow:
            self.position = 0
            return Signal.close()

        if self.position == -1 and curr_fast > curr_slow:
            self.position = 0
            return Signal.close()

        return None