class Strategy:
    PARAMS = {
        "keltner_ema_period": {"default": 20, "min": 10, "max": 40},
        "keltner_atr_period": {"default": 14, "min": 7, "max": 21},
        "keltner_mult": {"default": 2.0, "min": 1.0, "max": 3.0},
        "adx_period": {"default": 14, "min": 7, "max": 28},
        "adx_threshold": {"default": 22, "min": 15, "max": 35},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        from core.indicators import keltner, adx, atr, is_session

        # Compute indicators
        kc = keltner(
            bars,
            ema_period=self.params["keltner_ema_period"],
            atr_period=self.params["keltner_atr_period"],
            multiplier=self.params["keltner_mult"]
        )
        adx_res = adx(bars, period=self.params["adx_period"])
        atr_series = atr(bars, period=self.params["keltner_atr_period"])

        # Extract current values safely
        atr_val = float(atr_series.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        adx_val = float(adx_res.adx_line.iloc[-1])
        plus_di = float(adx_res.plus_di.iloc[-1])
        minus_di = float(adx_res.minus_di.iloc[-1])
        close = float(bars["close"].iloc[-1])
        kc_upper = float(kc.upper.iloc[-1])
        kc_lower = float(kc.lower.iloc[-1])

        # Validate no NaN in critical signals
        if any(np.isnan(v) for v in [adx_val, plus_di, minus_di, kc_upper, kc_lower]):
            return None

        # Session filter: only trade during London or NY
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # ADX trend strength filter
        if adx_val < self.params["adx_threshold"]:
            return None

        # Risk parameters
        sl_dist = max(5.0, atr_val * self.params["sl_mult"])
        tp_dist = max(7.5, atr_val * 2.5)  # Fixed RR ~1.67 minimum

        # Long entry: close above upper Keltner + bullish DI dominance
        if close > kc_upper and plus_di > minus_di:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short entry: close below lower Keltner + bearish DI dominance
        if close < kc_lower and minus_di > plus_di:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None