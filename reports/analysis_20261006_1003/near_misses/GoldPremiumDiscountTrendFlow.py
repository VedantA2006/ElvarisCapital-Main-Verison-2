class Strategy:
    PARAMS = {
        "swing_n": {"default": 5, "min": 3, "max": 10},
        "roc_n": {"default": 10, "min": 5, "max": 20},
        "discount_thresh": {"default": 0.45, "min": 0.3, "max": 0.5},
        "premium_thresh": {"default": 0.55, "min": 0.5, "max": 0.7},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params
        self.cooldown = 0

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Decrement cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        # Compute indicators
        atr_val = atr(bars, 14).iloc[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        # Session filter
        in_session = is_session(bars, "london") | is_session(bars, "ny")
        if not in_session.iloc[-1]:
            return None

        # Market structure and location
        bull_bos, bear_bos = bos(bars, self.params["swing_n"])
        pd_zone = premium_discount(bars, swing_n=self.params["swing_n"])
        roc_val = roc(bars, self.params["roc_n"]).iloc[-1]

        if np.isnan(roc_val):
            return None

        current_pd = pd_zone.iloc[-1]
        if np.isnan(current_pd):
            return None

        # Entry logic
        if bull_bos.iloc[-1] and current_pd < self.params["discount_thresh"] and roc_val > 0:
            sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
            tp_dist = max(1.5, float(atr_val * self.params["tp_mult"]))
            self.cooldown = 3
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        if bear_bos.iloc[-1] and current_pd > self.params["premium_thresh"] and roc_val < 0:
            sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
            tp_dist = max(1.5, float(atr_val * self.params["tp_mult"]))
            self.cooldown = 3
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None