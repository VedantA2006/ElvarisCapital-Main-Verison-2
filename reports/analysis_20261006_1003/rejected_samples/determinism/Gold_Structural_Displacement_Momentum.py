import numpy as np
import pandas as pd
from core.indicators import bos, fvg, atr, adx, is_session

class Strategy:
    PARAMS = {
        "swing_n": {"default": 5, "min": 3, "max": 10},
        "adx_n": {"default": 14, "min": 7, "max": 30},
        "adx_thresh": {"default": 20.0, "min": 10.0, "max": 40.0},
        "atr_n": {"default": 14, "min": 7, "max": 30},
        "sl_mult": {"default": 1.5, "min": 0.5, "max": 3.0},
        "tp_rr": {"default": 2.0, "min": 1.0, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars: pd.DataFrame):
        # Warmup safety
        if len(bars) < 60:
            return None

        # Extract parameters
        swing_n = int(self.params["swing_n"])
        adx_n = int(self.params["adx_n"])
        adx_thresh = float(self.params["adx_thresh"])
        atr_n = int(self.params["atr_n"])
        sl_mult = float(self.params["sl_mult"])
        tp_rr = float(self.params["tp_rr"])

        # Calculate indicators
        atr_series = atr(bars, atr_n)
        adx_series = adx(bars, adx_n)
        bull_bos, bear_bos = bos(bars, swing_n)
        
        # FVG with minimum gap to ensure genuine displacement
        bull_fvg, bear_fvg = fvg(bars, min_gap_usd=1.0)
        
        # Session filters
        is_london = is_session(bars, "london")
        is_ny = is_session(bars, "ny")

        # Get latest values safely
        atr_val = float(atr_series.iloc[-1])
        adx_val = float(adx_series.iloc[-1])
        bull_bos_val = bool(bull_bos.iloc[-1])
        bear_bos_val = bool(bear_bos.iloc[-1])
        bull_fvg_val = bool(bull_fvg.iloc[-1])
        bear_fvg_val = bool(bear_fvg.iloc[-1])
        is_london_val = bool(is_london.iloc[-1])
        is_ny_val = bool(is_ny.iloc[-1])

        # NaN and validity checks for Stop Loss safety
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        if np.isnan(adx_val):
            return None

        # Active session check (London or NY)
        active_session = is_london_val or is_ny_val

        # Trend regime check
        trending = adx_val > adx_thresh

        # Calculate distances ensuring strict positivity and minimums
        sl_dist = max(5.0, float(atr_val * sl_mult))
        tp_dist = max(1.50, float(sl_dist * tp_rr))

        # Entry Logic
        if active_session and trending:
            # Long: Bullish Break of Structure coinciding with Bullish Fair Value Gap
            if bull_bos_val and bull_fvg_val:
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)
            
            # Short: Bearish Break of Structure coinciding with Bearish Fair Value Gap
            if bear_bos_val and bear_fvg_val:
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None