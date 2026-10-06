class Strategy:
    PARAMS = {
        "adx_period": {"default": 14, "min": 10, "max": 20},
        "kc_ema_period": {"default": 20, "min": 15, "max": 30},
        "kc_atr_period": {"default": 10, "min": 7, "max": 14},
        "kc_multiplier": {"default": 2.0, "min": 1.5, "max": 3.0},
        "sl_mult": {"default": 1.8, "min": 1.2, "max": 2.5},
        "tp_mult": {"default": 2.8, "min": 2.0, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        from core.indicators import adx, keltner, roc, atr, is_session
        import numpy as np

        # Compute indicators
        adx_res = adx(bars, period=self.params["adx_period"])
        kc = keltner(
            bars,
            ema_period=self.params["kc_ema_period"],
            atr_period=self.params["kc_atr_period"],
            multiplier=self.params["kc_multiplier"]
        )
        roc_series = roc(bars, period=10)
        atr_series = atr(bars, period=14)
        london = is_session(bars, "london")
        ny = is_session(bars, "ny")

        # Extract latest values safely
        adx_val = float(adx_res.adx_line.iloc[-1])
        roc_val = float(roc_series.iloc[-1])
        atr_val = float(atr_series.iloc[-1])
        close_curr = float(bars["close"].iloc[-1])
        close_prev = float(bars["close"].iloc[-2])
        kc_upper_curr = float(kc.upper.iloc[-1])
        kc_lower_curr = float(kc.lower.iloc[-1])
        kc_upper_prev = float(kc.upper.iloc[-2])
        kc_lower_prev = float(kc.lower.iloc[-2])
        in_session = bool(london.iloc[-1] or ny.iloc[-1])

        # Safety checks
        if np.isnan(adx_val) or np.isnan(roc_val) or np.isnan(atr_val) or atr_val <= 0:
            return None

        if not in_session:
            return None

        # ADX trend strength filter
        if adx_val < 20.0:
            return None

        # Risk distances
        sl_dist = max(5.0, atr_val * self.params["sl_mult"])
        tp_dist = max(7.5, atr_val * self.params["tp_mult"])

        # Long entry: fresh breakout above upper band with positive momentum
        long_breakout = (close_curr > kc_upper_curr) and (close_prev <= kc_upper_prev)
        long_momentum = roc_val > 0.0
        if long_breakout and long_momentum:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short entry: fresh breakout below lower band with negative momentum
        short_breakout = (close_curr < kc_lower_curr) and (close_prev >= kc_lower_prev)
        short_momentum = roc_val < 0.0
        if short_breakout and short_momentum:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None