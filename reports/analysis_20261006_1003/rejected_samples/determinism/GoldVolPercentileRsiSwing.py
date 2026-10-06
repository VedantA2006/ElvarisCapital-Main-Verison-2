import numpy as np
import pandas as pd
from core.indicators import vol_percentile, rsi, swing_high, swing_low, atr, is_session

class Strategy:
    PARAMS = {
        "vol_thresh": {"default": 60.0, "min": 50.0, "max": 85.0},
        "rsi_period": {"default": 14, "min": 7, "max": 21},
        "swing_n": {"default": 3, "min": 2, "max": 6},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 5.0}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999
        self.last_sh = np.nan
        self.last_sl = np.nan

    def on_bar(self, bars: pd.DataFrame):
        if len(bars) < 60:
            return None

        current_idx = len(bars) - 1

        # Enforce cooldown to prevent overtrading and ensure sample quality
        if current_idx - self.last_trade_bar < 3:
            return None

        # Session filter: Only trade during London or NY sessions for optimal liquidity
        sess = is_session(bars, "london") | is_session(bars, "ny")
        if not sess.iloc[-1]:
            return None

        # Calculate indicators
        vp = vol_percentile(bars, 14, 100)
        rsi_vals = rsi(bars, int(self.params["rsi_period"]))
        atr_vals = atr(bars, 14)
        sw_n = int(self.params["swing_n"])
        sh = swing_high(bars, sw_n)
        sl_swing = swing_low(bars, sw_n)

        # Extract latest values safely using only the most recent closed bar (iloc[-1])
        vp_val = float(vp.iloc[-1])
        rsi_val = float(rsi_vals.iloc[-1])
        atr_val = float(atr_vals.iloc[-1])
        close_val = float(bars["close"].iloc[-1])

        # Safety checks for NaNs
        if np.isnan(vp_val) or np.isnan(rsi_val) or np.isnan(atr_val) or atr_val <= 0:
            return None

        # Volatility regime filter: Only trade when volatility is expanding relative to history
        if vp_val < self.params["vol_thresh"]:
            return None

        # Update last confirmed swing high using only the latest bar to avoid variable iloc indexing
        latest_sh = sh.iloc[-1]
        if latest_sh:
            if hasattr(latest_sh, 'pivot_value') and latest_sh.pivot_value is not None:
                self.last_sh = float(latest_sh.pivot_value)
            else:
                self.last_sh = float(bars["high"].iloc[-1])

        # Update last confirmed swing low using only the latest bar to avoid variable iloc indexing
        latest_sl = sl_swing.iloc[-1]
        if latest_sl:
            if hasattr(latest_sl, 'pivot_value') and latest_sl.pivot_value is not None:
                self.last_sl = float(latest_sl.pivot_value)
            else:
                self.last_sl = float(bars["low"].iloc[-1])

        # Distances calculation with safety bounds
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(atr_val * self.params["tp_mult"]))

        # Long Entry Logic: Volatility expansion + Bullish Momentum + Price above confirmed swing high
        if rsi_val > 50.0 and not np.isnan(self.last_sh) and close_val > self.last_sh:
            self.last_trade_bar = current_idx
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short Entry Logic: Volatility expansion + Bearish Momentum + Price below confirmed swing low
        if rsi_val < 50.0 and not np.isnan(self.last_sl) and close_val < self.last_sl:
            self.last_trade_bar = current_idx
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None