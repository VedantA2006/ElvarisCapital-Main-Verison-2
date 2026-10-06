class Strategy:
    PARAMS = {
        "fast_ema": {"default": 9, "min": 5, "max": 20},
        "slow_ema": {"default": 21, "min": 15, "max": 50},
        "roc_period": {"default": 10, "min": 5, "max": 20},
        "roc_thresh": {"default": 0.8, "min": 0.3, "max": 2.0},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Compute indicators
        fast_ema = ema(bars, self.params["fast_ema"])
        slow_ema = ema(bars, self.params["slow_ema"])
        roc_vals = roc(bars, self.params["roc_period"])
        atr_vals = atr(bars, 14)
        in_session = is_session(bars, "london_ny")

        # Get latest values
        fast_val = fast_ema.iloc[-1]
        slow_val = slow_ema.iloc[-1]
        roc_val = roc_vals.iloc[-1]
        atr_val = atr_vals.iloc[-1]
        session_active = in_session.iloc[-1]

        # Safety checks
        if np.isnan(fast_val) or np.isnan(slow_val) or np.isnan(roc_val) or np.isnan(atr_val):
            return None
        if atr_val <= 0:
            return None
        if not session_active:
            return None

        close_price = bars["close"].iloc[-1]
        roc_thresh = self.params["roc_thresh"]

        # Entry logic
        if fast_val > slow_val and roc_val > roc_thresh and close_price > fast_val:
            sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
            tp_dist = max(1.5, float(atr_val * self.params["tp_mult"]))
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        elif fast_val < slow_val and roc_val < -roc_thresh and close_price < fast_val:
            sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
            tp_dist = max(1.5, float(atr_val * self.params["tp_mult"]))
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None