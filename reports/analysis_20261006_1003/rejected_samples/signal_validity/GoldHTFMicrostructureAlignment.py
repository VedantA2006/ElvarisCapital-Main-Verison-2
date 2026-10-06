import numpy as np
import pandas as pd
from core.indicators import htf, ema, bos, atr, is_session

class Strategy:
    PARAMS = {
        "swing_n": {"default": 5, "min": 3, "max": 10},
        "ema_fast": {"default": 20, "min": 10, "max": 50},
        "atr_period": {"default": 14, "min": 7, "max": 30},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "rr_ratio": {"default": 2.0, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params
        self.cooldown = 0
        self.in_position = False
        self.direction = None

    def on_bar(self, bars: pd.DataFrame):
        if len(bars) < 60:
            return None

        # Cooldown management
        if self.cooldown > 0:
            self.cooldown -= 1

        # Calculate indicators
        swing_n = int(self.params["swing_n"])
        ema_period = int(self.params["ema_fast"])
        atr_period = int(self.params["atr_period"])

        # HTF Trend: Resample to 4H and calculate EMA
        ema_4h = htf(bars, "4h", lambda b: ema(b, ema_period))
        
        # 1H Market Structure
        bull_bos, bear_bos = bos(bars, swing_n)
        
        # Volatility
        atr_val = atr(bars, atr_period).iloc[-1]
        
        # Session Filter
        is_london = is_session(bars, "london").iloc[-1]
        is_ny = is_session(bars, "ny").iloc[-1]
        valid_session = bool(is_london or is_ny)

        # Safety checks
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        if np.isnan(ema_4h.iloc[-1]) or np.isnan(ema_4h.iloc[-2]) or np.isnan(ema_4h.iloc[-3]):
            return None
        if np.isnan(bull_bos.iloc[-1]) or np.isnan(bear_bos.iloc[-1]):
            return None

        # Determine 4H Trend based on EMA slope over last 3 closed 4H bars
        htf_trend_up = (ema_4h.iloc[-1] > ema_4h.iloc[-2]) and (ema_4h.iloc[-2] > ema_4h.iloc[-3])
        htf_trend_down = (ema_4h.iloc[-1] < ema_4h.iloc[-2]) and (ema_4h.iloc[-2] < ema_4h.iloc[-3])

        curr_bull_bos = bool(bull_bos.iloc[-1])
        curr_bear_bos = bool(bear_bos.iloc[-1])

        # Exit logic via cooldown expiration (simulated TP hit or manual close not modeled here, relying on framework SL/TP)
        # We just manage state
        if self.in_position:
            if self.cooldown == 0:
                self.in_position = False
                self.direction = None
            return None

        # Entry Logic
        if not valid_session or self.cooldown > 0:
            return None

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(sl_dist * self.params["rr_ratio"]))

        if htf_trend_up and curr_bull_bos:
            self.in_position = True
            self.direction = "long"
            self.cooldown = 6  # Minimum bars before next trade
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        if htf_trend_down and curr_bear_bos:
            self.in_position = True
            self.direction = "short"
            self.cooldown = 6
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None