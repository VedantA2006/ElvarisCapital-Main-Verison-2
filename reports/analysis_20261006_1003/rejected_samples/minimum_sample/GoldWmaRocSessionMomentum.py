class Strategy:
    PARAMS = {
        "wma_fast": {"default": 20, "min": 10, "max": 30},
        "wma_slow": {"default": 50, "min": 30, "max": 80},
        "roc_period": {"default": 10, "min": 5, "max": 20},
        "roc_thresh": {"default": 0.3, "min": 0.1, "max": 0.8},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Cooldown check
        if len(bars) - self.last_trade_bar < 3:
            return None

        # Compute indicators
        wma_fast = wma(bars, self.params["wma_fast"])
        wma_slow = wma(bars, self.params["wma_slow"])
        roc_val = roc(bars, self.params["roc_period"])
        atr_val = atr(bars, 14)
        in_session = is_session(bars, "london") | is_session(bars, "ny")

        # Safety checks
        if np.isnan(atr_val.iloc[-1]) or atr_val.iloc[-1] <= 0:
            return None
        if np.isnan(roc_val.iloc[-1]):
            return None
        if not in_session.iloc[-1]:
            return None
        if atr_val.iloc[-1] < 3.0:
            return None

        # Entry logic
        roc_thresh = self.params["roc_thresh"]
        sl_dist = max(4.0, float(atr_val.iloc[-1] * self.params["sl_mult"]))
        tp_dist = max(6.0, float(atr_val.iloc[-1] * self.params["tp_mult"]))

        # Long condition
        if (wma_fast.iloc[-1] > wma_slow.iloc[-1]) and (roc_val.iloc[-1] > roc_thresh):
            self.last_trade_bar = len(bars)
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short condition
        if (wma_fast.iloc[-1] < wma_slow.iloc[-1]) and (roc_val.iloc[-1] < -roc_thresh):
            self.last_trade_bar = len(bars)
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None