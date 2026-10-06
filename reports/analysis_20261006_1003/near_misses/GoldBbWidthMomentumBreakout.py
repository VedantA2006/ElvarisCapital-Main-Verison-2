import numpy as np
import pandas as pd

class Strategy:
    PARAMS = {
        "bb_period": {"default": 20, "min": 10, "max": 40},
        "bb_std": {"default": 2.0, "min": 1.5, "max": 3.0},
        "macd_fast": {"default": 12, "min": 8, "max": 20},
        "macd_slow": {"default": 26, "min": 20, "max": 40},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 3.0, "min": 2.0, "max": 5.0}
    }

    def __init__(self, params):
        self.params = params
        self.cooldown = 0

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        from core.indicators import bollinger, macd, atr, is_session

        # Session filter: only trade during London or NY sessions for institutional volume
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        bb_period = int(self.params["bb_period"])
        bb_std = float(self.params["bb_std"])
        bb = bollinger(bars, period=bb_period, num_std=bb_std)
        
        macd_fast = int(self.params["macd_fast"])
        macd_slow = int(self.params["macd_slow"])
        m = macd(bars, fast_period=macd_fast, slow_period=macd_slow, signal_period=9)
        
        atr_series = atr(bars, period=14)
        atr_val = float(atr_series.iloc[-1])

        if np.isnan(atr_val) or atr_val <= 0:
            return None

        close = float(bars["close"].iloc[-1])
        upper_bb = float(bb.upper.iloc[-1])
        lower_bb = float(bb.lower.iloc[-1])
        bw_curr = float(bb.bandwidth.iloc[-1])
        bw_prev = float(bb.bandwidth.iloc[-2])

        if np.isnan(bw_curr) or np.isnan(bw_prev) or np.isnan(upper_bb) or np.isnan(lower_bb):
            return None

        # Calculate median bandwidth over the last 100 bars to define 'compression'
        bw_history = bb.bandwidth.iloc[-100:].dropna()
        if len(bw_history) < 50:
            return None
        median_bw = float(np.median(bw_history.values))

        # MACD Histogram values
        hist_curr = float(m.hist.iloc[-1])
        hist_prev = float(m.hist.iloc[-2])

        if np.isnan(hist_curr) or np.isnan(hist_prev):
            return None

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(7.5, float(atr_val * self.params["tp_mult"]))

        # LONG ENTRY LOGIC
        # 1. Volatility Expansion: Current bandwidth > Previous bandwidth
        # 2. Prior Compression: Previous bandwidth was below the rolling median
        # 3. Price Breakout: Close is above the Upper Bollinger Band
        # 4. Momentum Confirmation: MACD Histogram is positive and increasing
        vol_expanding = bw_curr > bw_prev
        prior_compression = bw_prev < median_bw
        price_breakout_long = close > upper_bb
        mom_bullish = hist_curr > 0 and hist_curr > hist_prev

        if vol_expanding and prior_compression and price_breakout_long and mom_bullish:
            self.cooldown = 3
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # SHORT ENTRY LOGIC
        # 1. Volatility Expansion: Current bandwidth > Previous bandwidth
        # 2. Prior Compression: Previous bandwidth was below the rolling median
        # 3. Price Breakdown: Close is below the Lower Bollinger Band
        # 4. Momentum Confirmation: MACD Histogram is negative and decreasing
        price_breakout_short = close < lower_bb
        mom_bearish = hist_curr < 0 and hist_curr < hist_prev

        if vol_expanding and prior_compression and price_breakout_short and mom_bearish:
            self.cooldown = 3
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None