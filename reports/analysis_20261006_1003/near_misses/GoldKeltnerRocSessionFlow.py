class Strategy:
    PARAMS = {
        "kc_ema_period": {"default": 20, "min": 10, "max": 30},
        "kc_atr_period": {"default": 10, "min": 7, "max": 14},
        "kc_multiplier": {"default": 2.0, "min": 1.5, "max": 3.0},
        "roc_period": {"default": 10, "min": 5, "max": 20},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        from core.indicators import keltner, roc, atr, is_session
        import numpy as np

        # Compute indicators
        kc = keltner(
            bars,
            ema_period=self.params["kc_ema_period"],
            atr_period=self.params["kc_atr_period"],
            multiplier=self.params["kc_multiplier"]
        )
        roc_series = roc(bars, period=self.params["roc_period"])
        atr_series = atr(bars, period=self.params["kc_atr_period"])

        # Extract latest values safely
        close_curr = float(bars["close"].iloc[-1])
        close_prev = float(bars["close"].iloc[-2])
        upper_curr = float(kc.upper.iloc[-1])
        lower_curr = float(kc.lower.iloc[-1])
        upper_prev = float(kc.upper.iloc[-2])
        lower_prev = float(kc.lower.iloc[-2])
        roc_val = float(roc_series.iloc[-1])
        atr_val = float(atr_series.iloc[-1])

        # Safety checks
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        if np.isnan(roc_val):
            return None

        # Session filter
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Compute SL/TP distances
        sl_dist = max(5.0, atr_val * self.params["sl_mult"])
        tp_dist = max(7.5, atr_val * self.params["tp_mult"])

        # Long entry: current close above upper band, previous close was inside or below upper band, ROC > 0
        long_breakout = (close_curr > upper_curr) and (close_prev <= upper_prev) and (roc_val > 0)
        # Short entry: current close below lower band, previous close was inside or above lower band, ROC < 0
        short_breakout = (close_curr < lower_curr) and (close_prev >= lower_prev) and (roc_val < 0)

        if long_breakout:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)
        if short_breakout:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None