class Strategy:
    PARAMS = {
        "ema_fast": {"default": 50, "min": 20, "max": 80},
        "ema_slow": {"default": 200, "min": 100, "max": 300},
        "rsi_period": {"default": 7, "min": 4, "max": 14},
        "adx_period": {"default": 14, "min": 10, "max": 25},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
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

        from core.indicators import ema, rsi, adx, atr, is_session

        # Session filter: Only trade during London or NY for liquidity
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Indicators
        fast_ema = ema(bars, period=self.params["ema_fast"])
        slow_ema = ema(bars, period=self.params["ema_slow"])
        rsi_series = rsi(bars, period=self.params["rsi_period"])
        adx_res = adx(bars, period=self.params["adx_period"])
        atr_series = atr(bars, period=14)

        # Extract current and previous values safely
        atr_val = float(atr_series.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        fast_curr = float(fast_ema.iloc[-1])
        fast_prev = float(fast_ema.iloc[-2])
        slow_curr = float(slow_ema.iloc[-1])
        slow_prev = float(slow_ema.iloc[-2])

        if np.isnan(fast_curr) or np.isnan(slow_curr) or np.isnan(fast_prev) or np.isnan(slow_prev):
            return None

        rsi_curr = float(rsi_series.iloc[-1])
        rsi_prev = float(rsi_series.iloc[-2])
        adx_curr = float(adx_res.adx_line.iloc[-1])
        plus_di = float(adx_res.plus_di.iloc[-1])
        minus_di = float(adx_res.minus_di.iloc[-1])

        if np.isnan(rsi_curr) or np.isnan(rsi_prev) or np.isnan(adx_curr):
            return None

        # Calculate SL and TP distances
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(7.5, float(atr_val * self.params["tp_mult"]))

        # Trend conditions
        bull_trend = (fast_curr > slow_curr) and (plus_di > minus_di) and (adx_curr > 20.0)
        bear_trend = (fast_curr < slow_curr) and (minus_di > plus_di) and (adx_curr > 20.0)

        # Entry Logic: RSI Pullback Exhaustion snapping back into trend
        # Long: RSI crosses above 35 (recovering from oversold pullback in uptrend)
        if bull_trend and rsi_prev <= 35.0 and rsi_curr > 35.0:
            self.cooldown = 3
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short: RSI crosses below 65 (recovering from overbought pullback in downtrend)
        if bear_trend and rsi_prev >= 65.0 and rsi_curr < 65.0:
            self.cooldown = 3
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None