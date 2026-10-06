import numpy as np
import pandas as pd
from core.indicators import ema, atr, supertrend

class Strategy:
    PARAMS = {
        "ema_fast": {"default": 21, "min": 5, "max": 50},
        "ema_slow": {"default": 55, "min": 20, "max": 200},
        "atr_period": {"default": 14, "min": 7, "max": 30},
        "sl_mult": {"default": 1.5, "min": 0.5, "max": 4.0},
        "tp_mult": {"default": 2.5, "min": 1.0, "max": 6.0},
        "st_period": {"default": 10, "min": 5, "max": 20}
    }

    def __init__(self, params):
        self.params = params
        self.position = 0

    def on_bar(self, bars: pd.DataFrame):
        if len(bars) < 60:
            return None

        close = bars['close'].values
        high = bars['high'].values
        low = bars['low'].values

        fast_len = int(self.params["ema_fast"])
        slow_len = int(self.params["ema_slow"])
        atr_len = int(self.params["atr_period"])
        st_len = int(self.params["st_period"])

        ema_fast = ema(close, fast_len)
        ema_slow = ema(close, slow_len)
        atr_val = atr(high, low, close, atr_len)[-1]
        
        # Supertrend returns (direction_array, band_array)
        st_dir, st_band = supertrend(high, low, close, st_len, 3.0)
        st_direction = st_dir[-1]

        if np.isnan(atr_val) or atr_val <= 0:
            return None
        if np.isnan(ema_fast[-1]) or np.isnan(ema_slow[-1]):
            return None

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(atr_val * self.params["tp_mult"]))

        curr_price = close[-1]
        prev_ema_f = ema_fast[-2]
        prev_ema_s = ema_slow[-2]
        curr_ema_f = ema_fast[-1]
        curr_ema_s = ema_slow[-1]

        # Trend alignment: EMA cross and slope confirmation + Supertrend filter
        long_trend = (curr_ema_f > curr_ema_s) and (curr_ema_f > prev_ema_f) and (st_direction == 1)
        short_trend = (curr_ema_f < curr_ema_s) and (curr_ema_f < prev_ema_f) and (st_direction == -1)

        # Microstructure pullback entry: price pulls back to the fast EMA zone
        pullback_long = (low[-1] <= curr_ema_f) and (curr_price > curr_ema_f)
        pullback_short = (high[-1] >= curr_ema_f) and (curr_price < curr_ema_f)

        if long_trend and pullback_long and self.position <= 0:
            self.position = 1
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        if short_trend and pullback_short and self.position >= 0:
            self.position = -1
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        # Exit logic: trend invalidation via EMA crossover against position
        if self.position == 1 and curr_ema_f < curr_ema_s:
            self.position = 0
            return Signal.close()

        if self.position == -1 and curr_ema_f > curr_ema_s:
            self.position = 0
            return Signal.close()

        return None