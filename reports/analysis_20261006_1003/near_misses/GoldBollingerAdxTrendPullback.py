import numpy as np
import pandas as pd

class Strategy:
    PARAMS = {
        "bb_period": {"default": 20, "min": 10, "max": 30},
        "adx_period": {"default": 14, "min": 7, "max": 21},
        "adx_thresh": {"default": 25.0, "min": 15.0, "max": 40.0},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Cooldown to prevent overtrading and ensure sample quality
        if len(bars) - self.last_trade_bar < 3:
            return None

        from core.indicators import bollinger, adx, atr, rsi, is_session

        # Session filter: only trade during London or NY sessions for optimal liquidity
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        bb_period = int(self.params["bb_period"])
        adx_period = int(self.params["adx_period"])
        atr_period = int(self.params["atr_period"])
        adx_thresh = float(self.params["adx_thresh"])

        bb = bollinger(bars, period=bb_period, num_std=2.0)
        adx_res = adx(bars, period=adx_period)
        atr_series = atr(bars, period=atr_period)
        rsi_series = rsi(bars, period=14)

        atr_val = float(atr_series.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        # Extract current and previous values safely
        adx_val = float(adx_res.adx_line.iloc[-1])
        plus_di = float(adx_res.plus_di.iloc[-1])
        minus_di = float(adx_res.minus_di.iloc[-1])
        rsi_val = float(rsi_series.iloc[-1])

        close_curr = float(bars["close"].iloc[-1])
        close_prev = float(bars["close"].iloc[-2])

        bb_upper_curr = float(bb.upper.iloc[-1])
        bb_lower_curr = float(bb.lower.iloc[-1])
        bb_mid_curr = float(bb.middle.iloc[-1])
        bb_mid_prev = float(bb.middle.iloc[-2])

        bb_upper_prev = float(bb.upper.iloc[-2])
        bb_lower_prev = float(bb.lower.iloc[-2])

        # Check for NaNs in indicators
        if any(np.isnan(x) for x in [adx_val, plus_di, minus_di, rsi_val, bb_upper_curr, bb_mid_curr]):
            return None

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(7.5, float(atr_val * self.params["tp_mult"]))

        # Did price touch/exceed the bands in the last 3 bars? (Ensures it was extended)
        touched_upper_recently = any(float(bars["high"].iloc[-i]) >= float(bb.upper.iloc[-i]) for i in range(1, 4))
        touched_lower_recently = any(float(bars["low"].iloc[-i]) <= float(bb.lower.iloc[-i]) for i in range(1, 4))

        # LONG ENTRY LOGIC
        # 1. Strong Uptrend: ADX > thresh and +DI > -DI
        # 2. Extended recently: Touched upper band in last 3 bars
        # 3. Pullback to Value: Current close crosses below middle band from above
        # 4. Momentum intact: RSI > 40
        if adx_val > adx_thresh and plus_di > minus_di:
            if touched_upper_recently:
                if close_prev >= bb_mid_prev and close_curr < bb_mid_curr:
                    if rsi_val > 40.0:
                        self.last_trade_bar = len(bars)
                        return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # SHORT ENTRY LOGIC
        # 1. Strong Downtrend: ADX > thresh and -DI > +DI
        # 2. Extended recently: Touched lower band in last 3 bars
        # 3. Pullback to Value: Current close crosses above middle band from below
        # 4. Momentum intact: RSI < 60
        if adx_val > adx_thresh and minus_di > plus_di:
            if touched_lower_recently:
                if close_prev <= bb_mid_prev and close_curr > bb_mid_curr:
                    if rsi_val < 60.0:
                        self.last_trade_bar = len(bars)
                        return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None