import numpy as np
import pandas as pd
from core.indicators import liquidity_sweep, choch, atr, is_session, premium_discount

class Strategy:
    PARAMS = {
        "swing_n": {"default": 4, "min": 2, "max": 8},
        "lookback": {"default": 6, "min": 3, "max": 12},
        "atr_period": {"default": 14, "min": 7, "max": 30},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "rr_ratio": {"default": 2.0, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars: pd.DataFrame):
        # Warmup safety
        if len(bars) < 60:
            return None

        # Session filter: Only trade during London or NY
        in_london = is_session(bars, "london").iloc[-1]
        in_ny = is_session(bars, "ny").iloc[-1]
        if not (in_london or in_ny):
            return None

        swing_n = int(self.params["swing_n"])
        lookback = int(self.params["lookback"])
        atr_period = int(self.params["atr_period"])

        # Market structure calculations
        high_sweep, low_sweep = liquidity_sweep(bars, swing_n=swing_n)
        bull_choch, bear_choch = choch(bars, swing_n=swing_n)
        pd_zone = premium_discount(bars, swing_n=swing_n)

        # Lookback windows for recent events (strictly closed bars, no lookahead)
        recent_low_sweep = low_sweep.iloc[-lookback:].any()
        recent_high_sweep = high_sweep.iloc[-lookback:].any()
        recent_bull_choch = bull_choch.iloc[-lookback:].any()
        recent_bear_choch = bear_choch.iloc[-lookback:].any()

        current_pd = float(pd_zone.iloc[-1])

        # ATR calculation for dynamic risk management
        atr_series = atr(bars, atr_period)
        atr_val = float(atr_series.iloc[-1])
        
        # NaN and zero safety check
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        # Calculate SL and TP distances ensuring strict positivity constraints
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(sl_dist * self.params["rr_ratio"]))

        # Long Entry Logic: Low swept + Bullish CHoCH + Discount Zone
        if recent_low_sweep and recent_bull_choch and current_pd < 0.5:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short Entry Logic: High swept + Bearish CHoCH + Premium Zone
        if recent_high_sweep and recent_bear_choch and current_pd > 0.5:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None