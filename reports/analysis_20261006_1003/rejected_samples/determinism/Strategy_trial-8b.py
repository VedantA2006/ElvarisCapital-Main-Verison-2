import numpy as np
import pandas as pd
from core.indicators import atr, ema, supertrend

PARAMS = {
    "atr_period": {"default": 14, "min": 5, "max": 50},
    "sl_mult": {"default": 1.5, "min": 0.5, "max": 5.0},
    "tp_mult": {"default": 3.0, "min": 1.5, "max": 8.0},
    "fast_ema": {"default": 21, "min": 5, "max": 50},
    "slow_ema": {"default": 50, "min": 20, "max": 200},
    "st_period": {"default": 10, "min": 5, "max": 30}
}

class Strategy:
    """
    Institutional XAUUSD 1H Trend-Following Strategy.
    
    Rationale: Gold exhibits persistent intraday and swing momentum driven by 
    macro flows and safe-haven demand. Combining a trend filter (EMA cross) 
    with a microstructure volatility trigger (Supertrend flip) isolates high-probability entries.
    
    Mechanism:
    1. Trend Filter: Fast EMA > Slow EMA for longs; Fast EMA < Slow EMA for shorts.
    2. Trigger: Supertrend direction aligns with the EMA trend on the current closed bar.
    3. Risk Management: ATR-based dynamic stop loss and take profit distances.
    """
    
    def __init__(self, params):
        self.params = params
        self.in_position = False
        self.direction = None

    def on_bar(self, bars: pd.DataFrame):
        # Warmup safety
        if len(bars) < 60:
            return None

        close = bars["close"].values
        high = bars["high"].values
        low = bars["low"].values

        # Calculate indicators
        atr_period = int(self.params["atr_period"])
        fast_period = int(self.params["fast_ema"])
        slow_period = int(self.params["slow_ema"])
        st_period = int(self.params["st_period"])

        atr_val = atr(close, high, low, period=atr_period)[-1]
        
        # NaN and positive safety check for ATR
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        fast_ema_arr = ema(close, period=fast_period)
        slow_ema_arr = ema(close, period=slow_period)
        
        # Ensure EMAs are valid
        if np.isnan(fast_ema_arr[-1]) or np.isnan(slow_ema_arr[-1]):
            return None

        st_dir, st_val = supertrend(high, low, close, period=st_period, multiplier=3.0)
        
        if np.isnan(st_dir[-1]):
            return None

        # Current state (using only confirmed, closed bar data)
        curr_fast = fast_ema_arr[-1]
        curr_slow = slow_ema_arr[-1]
        curr_st_dir = int(st_dir[-1])  # Typically 1 for bullish, -1 for bearish

        # Trend conditions
        bullish_trend = curr_fast > curr_slow
        bearish_trend = curr_fast < curr_slow

        # Trigger conditions
        long_signal = bullish_trend and (curr_st_dir == 1)
        short_signal = bearish_trend and (curr_st_dir == -1)

        # Position management logic
        if self.in_position:
            # Close position if trend invalidation occurs
            if self.direction == "long" and not long_signal:
                self.in_position = False
                self.direction = None
                return Signal.close()
            
            if self.direction == "short" and not short_signal:
                self.in_position = False
                self.direction = None
                return Signal.close()
                
            return None

        # Entry logic
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(atr_val * self.params["tp_mult"]))

        if long_signal:
            self.in_position = True
            self.direction = "long"
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        if short_signal:
            self.in_position = True
            self.direction = "short"
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None