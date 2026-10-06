import numpy as np
import pandas as pd
from core.indicators import supertrend, fvg, atr, is_session

class Strategy:
    PARAMS = {
        "st_period": {"default": 10, "min": 5, "max": 20},
        "st_mult": {"default": 3.0, "min": 1.5, "max": 5.0},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0},
        "fvg_min_gap": {"default": 1.0, "min": 0.0, "max": 5.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        # Warmup safety
        if len(bars) < 60:
            return None

        # Session filter: Only trade during London or NY sessions
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Calculate indicators
        st_res = supertrend(bars, period=self.params["st_period"], multiplier=self.params["st_mult"])
        bull_fvg, bear_fvg = fvg(bars, min_gap_usd=self.params["fvg_min_gap"])
        atr_series = atr(bars, period=self.params["atr_period"])

        # Extract current values safely
        atr_val = float(atr_series.iloc[-1])
        st_dir = int(st_res.direction.iloc[-1])
        has_bull_fvg = bool(bull_fvg.iloc[-1])
        has_bear_fvg = bool(bear_fvg.iloc[-1])

        # NaN and positive safety checks
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        # Calculate distances ensuring they are strictly positive and meet minimums
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(7.5, float(atr_val * self.params["tp_mult"]))

        # Entry Logic: Trend alignment via Supertrend + Institutional footprint via FVG
        if st_dir == 1 and has_bull_fvg:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        if st_dir == -1 and has_bear_fvg:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None