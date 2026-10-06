import numpy as np
import pandas as pd
from core.indicators import opening_range, liquidity_sweep, adx, atr, is_session

class Strategy:
    PARAMS = {
        "asia_bars": {"default": 8, "min": 4, "max": 12},
        "swing_n": {"default": 3, "min": 2, "max": 6},
        "adx_thresh": {"default": 15.0, "min": 10.0, "max": 30.0},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "rr_ratio": {"default": 2.0, "min": 1.5, "max": 3.5}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars: pd.DataFrame):
        # Warmup safety
        if len(bars) < 60:
            return None

        # Only trade during London or NY sessions
        in_london = is_session(bars, "london").iloc[-1]
        in_ny = is_session(bars, "ny").iloc[-1]
        if not (in_london or in_ny):
            return None

        # Get Asian Opening Range (UTC day start, first N hours)
        asia_bars = int(self.params["asia_bars"])
        or_df = opening_range(bars, duration_bars=asia_bars)
        if or_df is None or len(or_df) == 0:
            return None

        asia_high = float(or_df["high"].iloc[-1])
        asia_low = float(or_df["low"].iloc[-1])

        # Ensure valid range
        if np.isnan(asia_high) or np.isnan(asia_low) or asia_high <= asia_low:
            return None

        # Trend/Momentum Filter: ADX
        adx_val = float(adx(bars, 14).iloc[-1])
        if np.isnan(adx_val) or adx_val < self.params["adx_thresh"]:
            return None

        # Volatility for SL/TP
        atr_val = float(atr(bars, 14).iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        # Liquidity Sweeps of the Asian Range
        swing_n = int(self.params["swing_n"])
        high_sweep, low_sweep = liquidity_sweep(bars, swing_n=swing_n)

        swept_high = bool(high_sweep.iloc[-1])
        swept_low = bool(low_sweep.iloc[-1])

        # Calculate safe distances
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(10.0, float(sl_dist * self.params["rr_ratio"]))

        close_price = float(bars["close"].iloc[-1])

        # SHORT LOGIC: Sweep above Asian High traps breakout buyers -> Reversal down
        if swept_high and close_price > asia_high:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        # LONG LOGIC: Sweep below Asian Low traps breakout sellers -> Reversal up
        if swept_low and close_price < asia_low:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        return None