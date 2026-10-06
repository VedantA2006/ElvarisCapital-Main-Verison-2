import numpy as np
import pandas as pd
from core.indicators import session_range, liquidity_sweep, fvg, atr, is_session

class Strategy:
    PARAMS = {
        "swing_n": {"default": 2, "min": 1, "max": 5},
        "atr_period": {"default": 14, "min": 7, "max": 30},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 5.0},
        "fvg_min_gap": {"default": 1.0, "min": 0.0, "max": 3.0}
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

        # Calculate indicators
        swing_n = int(self.params["swing_n"])
        atr_period = int(self.params["atr_period"])
        fvg_gap = float(self.params["fvg_min_gap"])

        atr_series = atr(bars, atr_period)
        atr_val = float(atr_series.iloc[-1])

        # ATR safety check
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        # Get market structure signals
        high_sweep, low_sweep = liquidity_sweep(bars, swing_n=swing_n)
        bull_fvg, bear_fvg = fvg(bars, min_gap_usd=fvg_gap)

        # Extract current bar values safely
        curr_high_sweep = bool(high_sweep.iloc[-1]) if not pd.isna(high_sweep.iloc[-1]) else False
        curr_low_sweep = bool(low_sweep.iloc[-1]) if not pd.isna(low_sweep.iloc[-1]) else False
        curr_bull_fvg = bool(bull_fvg.iloc[-1]) if not pd.isna(bull_fvg.iloc[-1]) else False
        curr_bear_fvg = bool(bear_fvg.iloc[-1]) if not pd.isna(bear_fvg.iloc[-1]) else False

        # Calculate distances
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(atr_val * self.params["tp_mult"]))

        # Entry Logic
        # Long: Sweep of Asian/session lows + Bullish FVG (displacement up)
        if curr_low_sweep and curr_bull_fvg:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short: Sweep of Asian/session highs + Bearish FVG (displacement down)
        if curr_high_sweep and curr_bear_fvg:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None