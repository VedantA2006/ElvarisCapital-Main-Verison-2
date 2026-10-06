import numpy as np
import pandas as pd
from core.indicators import ema, atr, supertrend

class Strategy:
    PARAMS = {
        "ema_fast": {"default": 21, "min": 5, "max": 50},
        "ema_slow": {"default": 55, "min": 20, "max": 200},
        "atr_period": {"default": 14, "min": 7, "max": 30},
        "sl_mult": {"default": 2.0, "min": 1.0, "max": 4.0},
        "tp_mult": {"default": 3.0, "min": 1.5, "max": 6.0},
        "st_period": {"default": 10, "min": 5, "max": 20}
    }

    def __init__(self, params):
        self.params = params
        self.position = 0

    def on_bar(self, bars: pd.DataFrame):
        if len(bars) < 60:
            return None

        close = bars["close"].values.astype(float)
        high = bars["high"].values.astype(float)
        low = bars["low"].values.astype(float)

        fast_len = int(self.params["ema_fast"])
        slow_len = int(self.params["ema_slow"])
        atr_len = int(self.params["atr_period"])
        st_len = int(self.params["st_period"])

        ema_fast = ema(close, fast_len)
        ema_slow = ema(close, slow_len)
        atr_val = atr(high, low, close, atr_len)[-1]
        
        st_dir, _ = supertrend(high, low, close, st_len, 3.0)

        if np.isnan(ema_fast[-1]) or np.isnan(ema_slow[-1]) or np.isnan(atr_val) or atr_val <= 0:
            return None

        # Current and previous bar values for confirmed crossover (no lookahead)
        ef_curr = ema_fast[-1]
        ef_prev = ema_fast[-2]
        es_curr = ema_slow[-1]
        es_prev = ema_slow[-2]

        # Supertrend direction for current bar
        st_curr = st_dir[-1]

        bull_cross = (ef_prev <= es_prev) and (ef_curr > es_curr)
        bear_cross = (ef_prev >= es_prev) and (ef_curr < es_curr)

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(atr_val * self.params["tp_mult"]))

        # Entry Logic
        if self.position == 0:
            if bull_cross and st_curr == 1:
                self.position = 1
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)
            
            if bear_cross and st_curr == -1:
                self.position = -1
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        # Exit Logic
        elif self.position == 1:
            # Close long if EMA crosses down or Supertrend flips bearish
            if bear_cross or st_curr == -1:
                self.position = 0
                return Signal.close()

        elif self.position == -1:
            # Close short if EMA crosses up or Supertrend flips bullish
            if bull_cross or st_curr == 1:
                self.position = 0
                return Signal.close()

        return None