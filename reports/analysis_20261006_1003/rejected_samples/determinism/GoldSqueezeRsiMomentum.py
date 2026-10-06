import numpy as np
import pandas as pd
from core.indicators import squeeze, rsi, adx, atr

class Strategy:
    PARAMS = {
        "rsi_n": {"default": 14, "min": 5, "max": 30},
        "adx_n": {"default": 14, "min": 7, "max": 30},
        "adx_thresh": {"default": 20.0, "min": 10.0, "max": 40.0},
        "sl_mult": {"default": 2.0, "min": 1.0, "max": 4.0},
        "rr_ratio": {"default": 2.0, "min": 1.5, "max": 4.0},
        "cooldown_bars": {"default": 3, "min": 1, "max": 8}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999

    def on_bar(self, bars: pd.DataFrame):
        # Warmup safety
        if len(bars) < 60:
            return None

        # Cooldown enforcement to prevent overtrading the same expansion leg
        if (len(bars) - 1) - self.last_trade_bar < self.params["cooldown_bars"]:
            return None

        # Calculate indicators using strictly closed bars
        sqz = squeeze(bars, bb_n=20, bb_std=2.0, kc_n=20, kc_mult=1.5)
        rsi_vals = rsi(bars, self.params["rsi_n"])
        adx_vals = adx(bars, self.params["adx_n"])
        atr_vals = atr(bars, 14)

        # Extract latest values safely
        sqz_val = float(sqz.iloc[-1])
        rsi_val = float(rsi_vals.iloc[-1])
        adx_val = float(adx_vals.iloc[-1])
        atr_val = float(atr_vals.iloc[-1])

        # NaN and validity checks
        if np.isnan(sqz_val) or np.isnan(rsi_val) or np.isnan(adx_val) or np.isnan(atr_val):
            return None
        if atr_val <= 0:
            return None

        # Squeeze logic: 
        # Typically returns 1 (Bullish Expansion/Release), -1 (Bearish Expansion/Release), 0 (Squeeze On)
        # We look for non-zero values indicating the squeeze has fired off.
        is_squeeze_releasing = abs(sqz_val) > 0.0

        # Trend/Momentum filters
        is_trending = adx_val > self.params["adx_thresh"]
        is_bullish_mom = rsi_val > 50.0
        is_bearish_mom = rsi_val < 50.0

        # Calculate safe distances
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.5, float(sl_dist * self.params["rr_ratio"]))

        # Entry Logic
        if is_squeeze_releasing and is_trending:
            if sqz_val > 0 and is_bullish_mom:
                self.last_trade_bar = len(bars) - 1
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)
            elif sqz_val < 0 and is_bearish_mom:
                self.last_trade_bar = len(bars) - 1
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None