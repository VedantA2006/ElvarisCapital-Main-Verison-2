class Strategy:
    PARAMS = {
        "kc_ema_period": {"default": 20, "min": 10, "max": 40},
        "kc_atr_period": {"default": 14, "min": 7, "max": 28},
        "kc_multiplier": {"default": 2.0, "min": 1.0, "max": 3.5},
        "rsi_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0},
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        from core.indicators import keltner, rsi, atr, is_session
        import numpy as np

        # Session filter: only trade during London or NY
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Compute indicators
        kc = keltner(
            bars,
            ema_period=self.params["kc_ema_period"],
            atr_period=self.params["kc_atr_period"],
            multiplier=self.params["kc_multiplier"]
        )
        rsi_series = rsi(bars, period=self.params["rsi_period"])
        atr_series = atr(bars, period=self.params["kc_atr_period"])

        # Extract current and previous values safely
        close_curr = float(bars["close"].iloc[-1])
        close_prev = float(bars["close"].iloc[-2])
        upper_curr = float(kc.upper.iloc[-1])
        upper_prev = float(kc.upper.iloc[-2])
        lower_curr = float(kc.lower.iloc[-1])
        lower_prev = float(kc.lower.iloc[-2])
        rsi_val = float(rsi_series.iloc[-1])
        atr_val = float(atr_series.iloc[-1])

        # Safety checks for NaN / invalid ATR
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        if any(np.isnan(v) for v in [upper_curr, upper_prev, lower_curr, lower_prev, rsi_val]):
            return None

        # Adaptive SL/TP distances in USD points
        sl_dist = max(5.0, atr_val * self.params["sl_mult"])
        tp_dist = max(7.5, atr_val * self.params["tp_mult"])

        # Long entry: fresh breakout above upper Keltner + RSI momentum (not exhausted)
        long_breakout = (close_curr > upper_curr) and (close_prev <= upper_prev)
        long_rsi_ok = (rsi_val > 50.0) and (rsi_val < 75.0)
        if long_breakout and long_rsi_ok:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short entry: fresh breakout below lower Keltner + RSI momentum (not exhausted)
        short_breakout = (close_curr < lower_curr) and (close_prev >= lower_prev)
        short_rsi_ok = (rsi_val < 50.0) and (rsi_val > 25.0)
        if short_breakout and short_rsi_ok:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None