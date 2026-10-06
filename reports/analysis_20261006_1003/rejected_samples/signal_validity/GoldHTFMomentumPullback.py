import numpy as np
import pandas as pd
from core.indicators import htf, adx, roc, rsi, ema, atr, is_session

class Strategy:
    PARAMS = {
        "adx_thresh": {"default": 25.0, "min": 15.0, "max": 40.0},
        "rsi_long_thresh": {"default": 45.0, "min": 30.0, "max": 50.0},
        "rsi_short_thresh": {"default": 55.0, "min": 50.0, "max": 70.0},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 5.0},
        "cooldown_bars": {"default": 3, "min": 1, "max": 8}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999

    def on_bar(self, bars: pd.DataFrame):
        if len(bars) < 60:
            return None

        idx = len(bars) - 1
        
        # Cooldown check to prevent overtrading and ensure sample quality
        if (idx - self.last_trade_bar) < self.params["cooldown_bars"]:
            return None

        # Only trade during London or NY sessions for optimal liquidity and follow-through
        in_london = is_session(bars, "london").iloc[-1]
        in_ny = is_session(bars, "ny").iloc[-1]
        if not (in_london or in_ny):
            return None

        # --- Higher Timeframe (4H) Institutional Flow ---
        htf_adx = htf(bars, "4h", lambda b: adx(b, 14))
        htf_roc = htf(bars, "4h", lambda b: roc(b, 10))
        
        if htf_adx is None or htf_roc is None or len(htf_adx) < 2 or len(htf_roc) < 2:
            return None
            
        current_htf_adx = float(htf_adx.iloc[-1])
        current_htf_roc = float(htf_roc.iloc[-1])
        
        if np.isnan(current_htf_adx) or np.isnan(current_htf_roc):
            return None

        # --- Lower Timeframe (1H) Indicators ---
        rsi_1h = rsi(bars, 14)
        ema_1h = ema(bars, 50)
        atr_1h = atr(bars, 14)
        
        current_rsi = float(rsi_1h.iloc[-1])
        prev_rsi = float(rsi_1h.iloc[-2])
        current_close = float(bars['close'].iloc[-1])
        current_ema = float(ema_1h.iloc[-1])
        current_atr = float(atr_1h.iloc[-1])
        
        # Warmup & Stop Loss Safety
        if np.isnan(current_rsi) or np.isnan(prev_rsi) or np.isnan(current_ema) or np.isnan(current_atr) or current_atr <= 0:
            return None

        # Calculate strictly positive distances
        sl_dist = max(5.0, float(current_atr * self.params["sl_mult"]))
        tp_dist = max(1.50, float(current_atr * self.params["tp_mult"]))

        # --- Entry Logic ---
        # Long: 4H Trend is strong and bullish, 1H price is above baseline, 1H RSI just bounced out of pullback
        if current_htf_adx > self.params["adx_thresh"] and current_htf_roc > 0:
            if current_close > current_ema:
                if prev_rsi <= self.params["rsi_long_thresh"] and current_rsi > self.params["rsi_long_thresh"]:
                    self.last_trade_bar = idx
                    return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short: 4H Trend is strong and bearish, 1H price is below baseline, 1H RSI just rejected from pullback
        if current_htf_adx > self.params["adx_thresh"] and current_htf_roc < 0:
            if current_close < current_ema:
                if prev_rsi >= self.params["rsi_short_thresh"] and current_rsi < self.params["rsi_short_thresh"]:
                    self.last_trade_bar = idx
                    return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None