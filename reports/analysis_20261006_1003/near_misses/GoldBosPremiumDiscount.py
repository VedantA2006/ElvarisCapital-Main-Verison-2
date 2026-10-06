import numpy as np
import pandas as pd
from core.indicators import bos, premium_discount, atr, ema

class Strategy:
    PARAMS = {
        "swing_n": {"default": 5, "min": 3, "max": 10},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 5.0}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999

    def on_bar(self, bars: pd.DataFrame):
        if len(bars) < 60:
            return None

        # Cooldown to prevent overtrading and ensure sample distribution
        if (len(bars) - 1) - self.last_trade_bar < 6:
            return None

        swing_n = int(self.params["swing_n"])
        atr_period = int(self.params["atr_period"])

        # Market Structure & Location
        bull_bos, bear_bos = bos(bars, swing_n)
        pd_zone = premium_discount(bars, swing_n)
        
        # Trend Filter
        ema50 = ema(bars, 50)
        
        # Volatility / Risk
        atr_val_series = atr(bars, atr_period)

        # Extract current bar values safely
        curr_bull_bos = bool(bull_bos.iloc[-1])
        curr_bear_bos = bool(bear_bos.iloc[-1])
        curr_pd = float(pd_zone.iloc[-1])
        curr_close = float(bars["close"].iloc[-1])
        curr_ema = float(ema50.iloc[-1])
        curr_atr = float(atr_val_series.iloc[-1])

        # Check recent BOS (within last 3 bars) to catch the immediate pullback into premium/discount
        recent_bull_bos = bool(bull_bos.iloc[-3:].any())
        recent_bear_bos = bool(bear_bos.iloc[-3:].any())

        # Safety checks
        if np.isnan(curr_atr) or curr_atr <= 0:
            return None
        if np.isnan(curr_ema) or np.isnan(curr_pd):
            return None

        sl_dist = max(5.0, float(curr_atr * self.params["sl_mult"]))
        tp_dist = max(1.50, float(curr_atr * self.params["tp_mult"]))

        # Long Entry: Bullish BOS recently occurred, price pulled back to Discount, above EMA50
        if recent_bull_bos and curr_pd < 0.5 and curr_close > curr_ema:
            self.last_trade_bar = len(bars) - 1
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short Entry: Bearish BOS recently occurred, price pulled back to Premium, below EMA50
        if recent_bear_bos and curr_pd > 0.5 and curr_close < curr_ema:
            self.last_trade_bar = len(bars) - 1
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None