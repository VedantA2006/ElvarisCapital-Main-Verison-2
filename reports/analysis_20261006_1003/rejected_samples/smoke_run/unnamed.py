import pandas as pd
from core.indicators import atr, ema, session_range

class Strategy:
    """
    XAUUSD 1H Institutional Microstructure Strategy
    
    Rationale: Gold exhibits strong intraday momentum driven by London and New York 
    session liquidity injections. This strategy captures the expansion phase following 
    the Asian session range breakout, filtered by higher-timeframe trend alignment.
    
    Mechanism:
    1. Compute the Asian session range (prior day's low-volume consolidation).
    2. Determine directional bias using a fast/slow EMA crossover on the 1H chart.
    3. Enter long when price breaks above the Asian high with bullish EMA alignment.
    4. Enter short when price breaks below the Asian low with bearish EMA alignment.
    5. Risk management uses ATR-based stop loss and take profit distances to adapt 
       to current volatility regimes.
    """
    
    PARAMS = {
        "fast_ema": {"default": 9, "min": 5, "max": 21},
        "slow_ema": {"default": 21, "min": 13, "max": 50},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_atr_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_atr_mult": {"default": 2.5, "min": 1.5, "max": 5.0},
        "cooldown_bars": {"default": 6, "min": 1, "max": 12}
    }

    def __init__(self, params):
        self.params = params
        self.position = 0  # 1 for long, -1 for short, 0 for flat
        self.bars_since_trade = 999

    def on_bar(self, bars: pd.DataFrame):
        if len(bars) < max(self.params["slow_ema"], self.params["atr_period"]) + 2:
            return None

        self.bars_since_trade += 1

        close = bars["close"]
        
        # Trend filters
        fast_ema = ema(close, self.params["fast_ema"])
        slow_ema = ema(close, self.params["slow_ema"])
        
        # Volatility measurement
        atr_val = atr(bars, self.params["atr_period"])
        
        # Current bar values (no lookahead)
        curr_close = close.iloc[-1]
        prev_fast = fast_ema.iloc[-2]
        prev_slow = slow_ema.iloc[-2]
        curr_fast = fast_ema.iloc[-1]
        curr_slow = slow_ema.iloc[-1]
        curr_atr = atr_val.iloc[-1]

        # Extract Asian session range from the core helper
        sr = session_range(bars)
        asian_high = sr["high"].iloc[-1]
        asian_low = sr["low"].iloc[-1]

        # Dynamic distances in USD points
        sl_dist = curr_atr * self.params["sl_atr_mult"]
        tp_dist = curr_atr * self.params["tp_atr_mult"]

        # --- Position Management ---
        if self.position != 0:
            # Exit on EMA trend reversal
            if self.position == 1 and curr_fast < curr_slow:
                self.position = 0
                self.bars_since_trade = 0
                return Signal.close()
            
            if self.position == -1 and curr_fast > curr_slow:
                self.position = 0
                self.bars_since_trade = 0
                return Signal.close()
                
            return None

        # --- Entry Logic ---
        if self.bars_since_trade < self.params["cooldown_bars"]:
            return None

        # Long: Bullish EMA cross/alignment + Breakout above Asian High
        if curr_fast > curr_slow and prev_fast <= prev_slow:
            if curr_close > asian_high:
                self.position = 1
                self.bars_since_trade = 0
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short: Bearish EMA cross/alignment + Breakout below Asian Low
        if curr_fast < curr_slow and prev_fast >= prev_slow:
            if curr_close < asian_low:
                self.position = -1
                self.bars_since_trade = 0
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        # Continuation entries if already aligned but missed the initial cross
        if curr_fast > curr_slow and curr_close > asian_high and self.bars_since_trade >= self.params["cooldown_bars"]:
            if close.iloc[-2] <= asian_high:  # Just broke out on this bar
                self.position = 1
                self.bars_since_trade = 0
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        if curr_fast < curr_slow and curr_close < asian_low and self.bars_since_trade >= self.params["cooldown_bars"]:
            if close.iloc[-2] >= asian_low:  # Just broke down on this bar
                self.position = -1
                self.bars_since_trade = 0
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None