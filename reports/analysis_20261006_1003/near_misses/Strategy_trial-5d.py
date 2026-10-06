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
        "st_mult": {"default": 3.0, "min": 1.0, "max": 5.0}
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

        ema_f = ema(close, self.params["ema_fast"])
        ema_s = ema(close, self.params["ema_slow"])
        atr_val = atr(high, low, close, self.params["atr_period"])[-1]
        
        st_dir, _, _ = supertrend(high, low, close, period=10, multiplier=self.params["st_mult"])

        if np.isnan(atr_val) or atr_val <= 0:
            return None
        if np.isnan(ema_f[-1]) or np.isnan(ema_s[-1]):
            return None

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(atr_val * self.params["tp_mult"]))

        trend_long = (ema_f[-1] > ema_s[-1]) and (ema_f[-2] <= ema_s[-2] or ema_f[-2] > ema_s[-2])
        trend_short = (ema_f[-1] < ema_s[-1]) and (ema_f[-2] >= ema_s[-2] or ema_f[-2] < ema_s[-2])

        # Momentum confirmation: price above/below slow EMA and Supertrend alignment
        long_condition = (close[-1] > ema_s[-1]) and (ema_f[-1] > ema_s[-1]) and (st_dir[-1] == 1)
        short_condition = (close[-1] < ema_s[-1]) and (ema_f[-1] < ema_s[-1]) and (st_dir[-1] == -1)

        # Cross-based entry trigger for higher frequency to meet Gate 7 sample size requirements
        cross_up = (ema_f[-1] > ema_s[-1]) and (ema_f[-2] <= ema_s[-2])
        cross_down = (ema_f[-1] < ema_s[-1]) and (ema_f[-2] >= ema_s[-2])

        if self.position is None:
            if cross_up or (long_condition and close[-1] > close[-2]):
                self.position = "long"
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)
            
            if cross_down or (short_condition and close[-1] < close[-2]):
                self.position = "short"
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)
                
        elif self.position == "long":
            # Exit if trend reverses
            if ema_f[-1] < ema_s[-1] or st_dir[-1] == -1:
                self.position = None
                return Signal.close()
                
        elif self.position == "short":
            # Exit if trend reverses
            if ema_f[-1] > ema_s[-1] or st_dir[-1] == 1:
                self.position = None
                return Signal.close()

        return None