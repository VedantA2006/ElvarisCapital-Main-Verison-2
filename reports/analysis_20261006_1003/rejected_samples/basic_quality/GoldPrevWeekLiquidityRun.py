import numpy as np
import pandas as pd
from core.indicators import previous_week_hl, atr, is_session

class Strategy:
    PARAMS = {
        "atr_period": {"default": 14, "min": 7, "max": 28},
        "sl_mult": {"default": 1.2, "min": 0.5, "max": 2.5},
        "rr_ratio": {"default": 2.0, "min": 1.2, "max": 3.5}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Session filter: Only trade during London or NY
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Volatility / SL sizing
        atr_val = float(atr(bars, period=self.params["atr_period"]).iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        # Previous Week High/Low
        prev_high, prev_low = previous_week_hl(bars)
        ph = float(prev_high.iloc[-1])
        pl = float(prev_low.iloc[-1])

        if np.isnan(ph) or np.isnan(pl):
            return None

        # Current bar OHLC
        c_high = float(bars["high"].iloc[-1])
        c_low = float(bars["low"].iloc[-1])
        c_close = float(bars["close"].iloc[-1])

        # Sweep logic: Price wicks beyond the level but closes back inside
        sweep_high = (c_high > ph) and (c_close < ph)
        sweep_low = (c_low < pl) and (c_close > pl)

        # Stop loss distance based on the sweep wick + ATR buffer
        sl_dist = max(5.0, abs(c_high - ph) + atr_val * self.params["sl_mult"])
        tp_dist = max(7.5, sl_dist * self.params["rr_ratio"])

        if sweep_high:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        if sweep_low:
            sl_dist_long = max(5.0, abs(pl - c_low) + atr_val * self.params["sl_mult"])
            tp_dist_long = max(7.5, sl_dist_long * self.params["rr_ratio"])
            return Signal.enter_long(sl_distance=sl_dist_long, tp_distance=tp_dist_long)

        return None