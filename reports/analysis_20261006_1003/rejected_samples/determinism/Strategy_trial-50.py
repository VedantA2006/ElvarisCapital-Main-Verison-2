import numpy as np
import pandas as pd
from core.indicators import ema, atr, supertrend

class Strategy:
    PARAMS = {
        "ema_fast": {"default": 21, "min": 5, "max": 50},
        "ema_slow": {"default": 55, "min": 20, "max": 200},
        "atr_period": {"default": 14, "min": 7, "max": 30},
        "sl_mult": {"default": 1.5, "min": 0.5, "max": 4.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 6.0},
        "st_period": {"default": 10, "min": 5, "max": 20}
    }

    def __init__(self, params):
        self.params = params
        self.position = None

    def on_bar(self, bars: pd.DataFrame):
        if len(bars) < 60:
            return None

        close = bars["close"].values
        high = bars["high"].values
        low = bars["low"].values

        fast_len = int(self.params["ema_fast"])
        slow_len = int(self.params["ema_slow"])
        atr_len = int(self.params["atr_period"])
        st_len = int(self.params["st_period"])

        ema_f = ema(close, fast_len)
        ema_s = ema(close, slow_len)
        atr_val = atr(high, low, close, atr_len)[-1]
        
        st_dir, _, _ = supertrend(high, low, close, st_len, 3.0)
        st_current = st_dir[-1]
        st_prev = st_dir[-2]

        if np.isnan(ema_f[-1]) or np.isnan(ema_s[-1]) or np.isnan(atr_val) or atr_val <= 0:
            return None

        ef_0 = float(ema_f[-1])
        ef_1 = float(ema_f[-2])
        es_0 = float(ema_s[-1])
        es_1 = float(ema_s[-2])

        # Trend alignment
        bull_trend = ef_0 > es_0
        bear_trend = ef_0 < es_0

        # EMA bullish crossover (fast crosses above slow)
        cross_up = ef_1 <= es_1 and ef_0 > es_0
        # EMA bearish crossover (fast crosses below slow)
        cross_dn = ef_1 >= es_1 and ef_0 < es_0

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(atr_val * self.params["tp_mult"]))

        # Entry logic
        if self.position is None:
            # Long entry: Bullish EMA cross OR existing uptrend with Supertrend flipping bullish
            if cross_up or (bull_trend and st_prev == -1 and st_current == 1):
                self.position = "long"
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

            # Short entry: Bearish EMA cross OR existing downtrend with Supertrend flipping bearish
            if cross_dn or (bear_trend and st_prev == 1 and st_current == -1):
                self.position = "short"
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        # Exit logic
        elif self.position == "long":
            # Close long if trend breaks down or supertrend flips bearish
            if ef_0 < es_0 or st_current == -1:
                self.position = None
                return Signal.close()

        elif self.position == "short":
            # Close short if trend breaks up or supertrend flips bullish
            if ef_0 > es_0 or st_current == 1:
                self.position = None
                return Signal.close()

        return None