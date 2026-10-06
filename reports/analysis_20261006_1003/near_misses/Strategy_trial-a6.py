import numpy as np
import pandas as pd
from core.indicators import ema, atr, supertrend

PARAMS = {
    "ema_fast": {"default": 21, "min": 5, "max": 50},
    "ema_slow": {"default": 55, "min": 20, "max": 200},
    "atr_period": {"default": 14, "min": 7, "max": 30},
    "sl_mult": {"default": 1.5, "min": 0.5, "max": 4.0},
    "tp_mult": {"default": 2.5, "min": 1.0, "max": 6.0},
    "st_period": {"default": 10, "min": 5, "max": 20}
}

class Strategy:
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

        ema_f = ema(close, fast_len)
        ema_s = ema(close, slow_len)
        atr_val = atr(high, low, close, atr_len)[-1]
        
        # Supertrend returns (direction_array, st_line_array) where direction is 1 for up, -1 for down
        st_dir, st_line = supertrend(high, low, close, st_len, 3.0)
        st_d = st_dir[-1]

        if np.isnan(ema_f[-1]) or np.isnan(ema_s[-1]) or np.isnan(atr_val) or atr_val <= 0:
            return None

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(atr_val * self.params["tp_mult"]))

        # Trend alignment: Fast EMA > Slow EMA and price above Supertrend
        trend_up = (ema_f[-1] > ema_s[-1]) and (st_d == 1)
        # Trend alignment: Fast EMA < Slow EMA and price below Supertrend
        trend_down = (ema_f[-1] < ema_s[-1]) and (st_d == -1)

        # Momentum pullback: previous bar closed lower than its open (bearish candle in an uptrend)
        pullback_long = close[-2] < bars['open'].values[-2]
        # Momentum pullback: previous bar closed higher than its open (bullish candle in a downtrend)
        pullback_short = close[-2] > bars['open'].values[-2]

        # Entry conditions
        enter_long = trend_up and pullback_long and (close[-1] > ema_f[-1])
        enter_short = trend_down and pullback_short and (close[-1] < ema_f[-1])

        if self.position == 0:
            if enter_long:
                self.position = 1
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)
            elif enter_short:
                self.position = -1
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)
                
        elif self.position == 1:
            # Exit long if trend breaks down
            if trend_down:
                self.position = 0
                return Signal.close()
                
        elif self.position == -1:
            # Exit short if trend breaks up
            if trend_up:
                self.position = 0
                return Signal.close()

        return None