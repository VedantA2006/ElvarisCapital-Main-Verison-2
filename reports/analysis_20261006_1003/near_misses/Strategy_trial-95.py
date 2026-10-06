import numpy as np
import pandas as pd
from core.indicators import ema, atr

class Strategy:
    """
    XAUUSD 4H Institutional Trend-Following Strategy
    
    Rationale: Gold exhibits persistent intraday and swing trends driven by macro flows 
    and institutional hedging. A dual-EMA trend filter combined with an ATR-scaled 
    pullback entry captures the statistical edge of buying dips in established uptrends 
    and selling rallies in downtrends.
    
    Mechanism:
    - Fast EMA (20) and Slow EMA (50) define the baseline trend.
    - Entries trigger on mean-reverting pullbacks to the fast EMA zone while the broader 
      trend remains intact.
    - Longs require close > slow_ema, fast_ema > slow_ema, and price pulling back near 
      the fast EMA (within 0.3 * ATR).
    - Shorts require the inverse.
    - Exits are dynamic ATR multiples for both SL and TP, ensuring positive expectancy 
      via a favorable risk-to-reward ratio.
    """
    
    PARAMS = {
        "fast_len": {"default": 20, "min": 8, "max": 35},
        "slow_len": {"default": 50, "min": 30, "max": 100},
        "atr_len": {"default": 14, "min": 7, "max": 25},
        "pullback_mult": {"default": 0.3, "min": 0.1, "max": 0.8},
        "sl_mult": {"default": 1.5, "min": 0.8, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 5.0}
    }

    def __init__(self, params):
        self.params = params
        self.position = 0

    def on_bar(self, bars: pd.DataFrame):
        # Warmup safety check
        if len(bars) < 60:
            return None

        close = bars['close'].values
        high = bars['high'].values
        low = bars['low'].values

        # Compute indicators using leak-safe core functions
        fast_ema = ema(close, self.params["fast_len"])
        slow_ema = ema(close, self.params["slow_len"])
        atr_val = atr(high, low, close, self.params["atr_len"])[-1]

        # NaN and zero safety checks
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        
        if np.isnan(fast_ema[-1]) or np.isnan(slow_ema[-1]):
            return None

        c = close[-1]
        f_ema = fast_ema[-1]
        s_ema = slow_ema[-1]

        # Dynamic distance calculations strictly bounded to prevent invalid signals
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(atr_val * self.params["tp_mult"]))

        # Trend definitions
        uptrend = (f_ema > s_ema) and (c > s_ema)
        downtrend = (f_ema < s_ema) and (c < s_ema)

        # Pullback thresholds scaled by volatility
        pb_thresh = atr_val * self.params["pullback_mult"]

        # Entry logic: Pullback into the fast EMA during an active trend
        long_pullback = uptrend and (c >= f_ema - pb_thresh) and (c <= f_ema + pb_thresh)
        short_pullback = downtrend and (c >= f_ema - pb_thresh) and (c <= f_ema + pb_thresh)

        # Position management and signal generation
        if self.position == 0:
            if long_pullback:
                self.position = 1
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)
            elif short_pullback:
                self.position = -1
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)
                
        elif self.position == 1:
            # Close long if trend structure breaks
            if c < s_ema:
                self.position = 0
                return Signal.close()
                
        elif self.position == -1:
            # Close short if trend structure breaks
            if c > s_ema:
                self.position = 0
                return Signal.close()

        return None