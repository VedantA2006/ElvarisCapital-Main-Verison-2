class Strategy:
    PARAMS = {
        "trend_period": {"default": 21, "min": 10, "max": 50},
        "mom_period": {"default": 10, "min": 5, "max": 20},
        "roc_threshold": {"default": 0.3, "min": 0.1, "max": 1.0},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0},
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        from core.indicators import ema, roc, atr, is_session
        import numpy as np

        # Compute indicators
        trend_ema = ema(bars, period=self.params["trend_period"])
        mom_roc = roc(bars, period=self.params["mom_period"])
        atr_series = atr(bars, period=self.params["atr_period"])

        # Safety checks
        atr_val = float(atr_series.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        # Session filter
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Current and previous values
        close_curr = float(bars["close"].iloc[-1])
        ema_curr = float(trend_ema.iloc[-1])
        roc_curr = float(mom_roc.iloc[-1])
        roc_prev = float(mom_roc.iloc[-2])
        thresh = self.params["roc_threshold"]

        # NaN safety for ROC/EMA
        if np.isnan(ema_curr) or np.isnan(roc_curr) or np.isnan(roc_prev):
            return None

        # Distance calculations
        sl_dist = max(5.0, atr_val * self.params["sl_mult"])
        tp_dist = max(7.5, atr_val * self.params["tp_mult"])

        # Long entry: price above EMA + fresh positive ROC acceleration
        if close_curr > ema_curr and roc_curr > thresh and roc_prev <= thresh:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short entry: price below EMA + fresh negative ROC acceleration
        if close_curr < ema_curr and roc_curr < -thresh and roc_prev >= -thresh:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None