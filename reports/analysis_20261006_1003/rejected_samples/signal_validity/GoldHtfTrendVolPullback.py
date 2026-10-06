import numpy as np
import pandas as pd
from core.indicators import htf, rsi, atr, ema, bos

class Strategy:
    PARAMS = {
        "adx_thresh": {"default": 25.0, "min": 15.0, "max": 40.0},
        "rsi_pullback": {"default": 45.0, "min": 25.0, "max": 60.0},
        "swing_n": {"default": 3, "min": 2, "max": 8},
        "sl_mult": {"default": 1.5, "min": 0.8, "max": 3.0},
        "tp_mult": {"default": 3.0, "min": 1.5, "max": 6.0},
        "cooldown": {"default": 3, "min": 1, "max": 8}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999

    def on_bar(self, bars: pd.DataFrame):
        # Warmup safety
        if len(bars) < 60:
            return None

        current_idx = len(bars) - 1
        if current_idx - self.last_trade_bar < int(self.params["cooldown"]):
            return None

        # --- Higher Timeframe (4H) Context using strictly closed candles ---
        try:
            htf_adx = htf(bars, "4h", lambda df: df["adx"])
            htf_close = htf(bars, "4h", lambda df: df["close"])
            htf_ema = htf(bars, "4h", lambda df: ema(df, 20))
        except Exception:
            return None

        if htf_adx is None or htf_close is None or htf_ema is None:
            return None
        if len(htf_adx) < 2 or len(htf_close) < 2 or len(htf_ema) < 2:
            return None

        # Use strictly the last fully closed 4H bar [-2] to prevent lookahead
        prev_htf_adx = float(htf_adx.iloc[-2])
        prev_htf_close = float(htf_close.iloc[-2])
        prev_htf_ema = float(htf_ema.iloc[-2])

        if np.isnan(prev_htf_adx) or np.isnan(prev_htf_close) or np.isnan(prev_htf_ema):
            return None

        htf_trending_up = (prev_htf_adx > self.params["adx_thresh"]) and (prev_htf_close > prev_htf_ema)
        htf_trending_down = (prev_htf_adx > self.params["adx_thresh"]) and (prev_htf_close < prev_htf_ema)

        if not htf_trending_up and not htf_trending_down:
            return None

        # --- 1H Indicators ---
        rsi_val = rsi(bars, 14)
        atr_val = atr(bars, 14)

        curr_rsi = float(rsi_val.iloc[-1])
        curr_atr = float(atr_val.iloc[-1])

        if np.isnan(curr_rsi) or np.isnan(curr_atr) or curr_atr <= 0:
            return None

        # --- Market Structure (BOS) ---
        swing_n = int(self.params["swing_n"])
        bull_bos, bear_bos = bos(bars, swing_n)

        curr_bull_bos = bool(bull_bos.iloc[-1])
        curr_bear_bos = bool(bear_bos.iloc[-1])

        rsi_thresh = float(self.params["rsi_pullback"])

        # --- Entry Logic ---
        # Long: 4H Uptrend + 1H RSI pulled back + 1H Bullish Break of Structure
        if htf_trending_up and (curr_rsi < rsi_thresh) and curr_bull_bos:
            sl_dist = max(5.0, float(curr_atr * self.params["sl_mult"]))
            tp_dist = max(1.50, float(curr_atr * self.params["tp_mult"]))
            self.last_trade_bar = current_idx
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short: 4H Downtrend + 1H RSI pulled back (overbought) + 1H Bearish Break of Structure
        if htf_trending_down and (curr_rsi > (100.0 - rsi_thresh)) and curr_bear_bos:
            sl_dist = max(5.0, float(curr_atr * self.params["sl_mult"]))
            tp_dist = max(1.50, float(curr_atr * self.params["tp_mult"]))
            self.last_trade_bar = current_idx
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None