import numpy as np
import pandas as pd

class Strategy:
    PARAMS = {
        "swing_n": {"default": 3, "min": 2, "max": 6},
        "zscore_period": {"default": 20, "min": 10, "max": 50},
        "zscore_thresh": {"default": 1.8, "min": 1.2, "max": 2.8},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0},
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        from core.indicators import zscore, atr, liquidity_sweep, choch, is_session

        # Session filter: only trade during London or NY for sufficient liquidity
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Core indicators
        zs = zscore(bars, period=self.params["zscore_period"])
        atr_series = atr(bars, period=self.params["atr_period"])
        
        atr_val = float(atr_series.iloc[-1])
        zs_val = float(zs.iloc[-1])

        # Safety checks for NaN and positive values
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        if np.isnan(zs_val):
            return None

        # Market structure signals
        high_sweep, low_sweep = liquidity_sweep(bars, swing_n=int(self.params["swing_n"]))
        bull_choch, bear_choch = choch(bars, swing_n=int(self.params["swing_n"]))

        hs = bool(high_sweep.iloc[-1])
        ls = bool(low_sweep.iloc[-1])
        b_choch = bool(bull_choch.iloc[-1])
        s_choch = bool(bear_choch.iloc[-1])

        thresh = self.params["zscore_thresh"]

        # Calculate safe distances
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.5, float(atr_val * self.params["tp_mult"]))

        # Long Entry: Liquidity sweep of lows + Oversold Z-score + Bullish CHoCH
        if ls and zs_val < -thresh and b_choch:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short Entry: Liquidity sweep of highs + Overbought Z-score + Bearish CHoCH
        if hs and zs_val > thresh and s_choch:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None
