class Strategy:
    PARAMS = {
        "kc_ema_n": {"default": 20, "min": 10, "max": 30},
        "kc_atr_n": {"default": 14, "min": 8, "max": 20},
        "kc_mult": {"default": 2.0, "min": 1.5, "max": 3.0},
        "htf_ema_n": {"default": 20, "min": 10, "max": 30},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Compute indicators
        kc = keltner(bars, ema_n=self.params["kc_ema_n"], atr_n=self.params["kc_atr_n"], mult=self.params["kc_mult"])
        atr_val = atr(bars, n=self.params["kc_atr_n"])
        htf_ema = htf(bars, "4h", lambda df: ema(df, n=self.params["htf_ema_n"]))
        session_mask = is_session(bars, "london") | is_session(bars, "ny")

        # Safety checks
        if np.isnan(atr_val.iloc[-1]) or atr_val.iloc[-1] <= 0:
            return None
        if np.isnan(kc["upper"].iloc[-1]) or np.isnan(kc["lower"].iloc[-1]):
            return None
        if np.isnan(htf_ema.iloc[-1]) or np.isnan(htf_ema.iloc[-2]):
            return None

        # Current and previous bar values
        close_curr = bars["close"].iloc[-1]
        close_prev = bars["close"].iloc[-2]
        upper_curr = kc["upper"].iloc[-1]
        lower_curr = kc["lower"].iloc[-1]
        upper_prev = kc["upper"].iloc[-2]
        lower_prev = kc["lower"].iloc[-2]
        htf_slope = htf_ema.iloc[-1] - htf_ema.iloc[-2]

        # Session filter
        if not session_mask.iloc[-1]:
            return None

        # Long entry: breakout above upper band + HTF uptrend + prior bar inside
        if (close_curr > upper_curr and
            close_prev <= upper_prev and
            htf_slope > 0):
            sl_dist = max(5.0, float(atr_val.iloc[-1] * self.params["sl_mult"]))
            tp_dist = max(1.5, float(atr_val.iloc[-1] * self.params["tp_mult"]))
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short entry: breakout below lower band + HTF downtrend + prior bar inside
        if (close_curr < lower_curr and
            close_prev >= lower_prev and
            htf_slope < 0):
            sl_dist = max(5.0, float(atr_val.iloc[-1] * self.params["sl_mult"]))
            tp_dist = max(1.5, float(atr_val.iloc[-1] * self.params["tp_mult"]))
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None