import numpy as np
import pandas as pd

class Strategy:
    PARAMS = {
        "adx_period": {"default": 14, "min": 7, "max": 21},
        "adx_thresh": {"default": 22.0, "min": 15.0, "max": 35.0},
        "stoch_k": {"default": 14, "min": 5, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 3.0, "min": 2.0, "max": 5.0},
        "cooldown": {"default": 3, "min": 1, "max": 6}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Cooldown check to prevent overtrading and ensure sample quality
        current_idx = len(bars) - 1
        if (current_idx - self.last_trade_bar) < int(self.params["cooldown"]):
            return None

        from core.indicators import adx, ema, stoch, atr, is_session

        # Session filter: Only trade during London or NY sessions for optimal liquidity
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Calculate indicators
        adx_res = adx(bars, period=int(self.params["adx_period"]))
        ema_fast = ema(bars, period=20)
        ema_slow = ema(bars, period=50)
        stoch_res = stoch(bars, k_period=int(self.params["stoch_k"]), d_period=3, slowing=3)
        atr_series = atr(bars, period=14)

        # Extract current and previous values safely
        adx_val = float(adx_res.adx_line.iloc[-1])
        plus_di = float(adx_res.plus_di.iloc[-1])
        minus_di = float(adx_res.minus_di.iloc[-1])
        
        ef_1 = float(ema_fast.iloc[-1])
        es_1 = float(ema_slow.iloc[-1])
        
        sk_1 = float(stoch_res.slow_k.iloc[-1])
        sd_1 = float(stoch_res.slow_d.iloc[-1])
        sk_2 = float(stoch_res.slow_k.iloc[-2])
        sd_2 = float(stoch_res.slow_d.iloc[-2])
        
        atr_val = float(atr_series.iloc[-1])

        # NaN and safety checks
        if np.isnan(adx_val) or np.isnan(atr_val) or atr_val <= 0:
            return None
        if np.isnan(sk_1) or np.isnan(sd_1) or np.isnan(sk_2) or np.isnan(sd_2):
            return None
        if np.isnan(ef_1) or np.isnan(es_1):
            return None

        # Trend Strength Filter
        if adx_val < self.params["adx_thresh"]:
            return None

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(7.5, float(atr_val * self.params["tp_mult"]))

        # Long Entry Logic:
        # 1. Bullish trend (+DI > -DI, Fast EMA > Slow EMA)
        # 2. Stochastic pullback exhausted (Slow K was < 30 on previous bar)
        # 3. Stochastic bullish crossover on current bar (K crosses above D)
        if plus_di > minus_di and ef_1 > es_1:
            if sk_2 < 30.0 and sk_2 <= sd_2 and sk_1 > sd_1:
                self.last_trade_bar = current_idx
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short Entry Logic:
        # 1. Bearish trend (-DI > +DI, Fast EMA < Slow EMA)
        # 2. Stochastic pullback exhausted (Slow K was > 70 on previous bar)
        # 3. Stochastic bearish crossover on current bar (K crosses below D)
        if minus_di > plus_di and ef_1 < es_1:
            if sk_2 > 70.0 and sk_2 >= sd_2 and sk_1 < sd_1:
                self.last_trade_bar = current_idx
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None