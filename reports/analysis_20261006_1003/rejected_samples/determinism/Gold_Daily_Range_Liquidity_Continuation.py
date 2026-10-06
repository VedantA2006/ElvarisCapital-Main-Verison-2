class Strategy:
    PARAMS = {
        "swing_n": {"default": 3, "min": 2, "max": 6},
        "adx_n": {"default": 14, "min": 8, "max": 30},
        "adx_thresh": {"default": 22.0, "min": 15.0, "max": 35.0},
        "atr_n": {"default": 14, "min": 8, "max": 30},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Compute indicators
        adx_val = adx(bars, self.params["adx_n"])
        atr_val = atr(bars, self.params["atr_n"])
        prev_day = previous_day_hl(bars)
        sweep = liquidity_sweep(bars, swing_n=self.params["swing_n"])
        sess = is_session(bars, "london") | is_session(bars, "ny")

        # Safety checks
        if np.isnan(adx_val.iloc[-1]) or np.isnan(atr_val.iloc[-1]):
            return None
        if atr_val.iloc[-1] <= 0:
            return None

        # Current bar values
        curr_adx = float(adx_val.iloc[-1])
        curr_atr = float(atr_val.iloc[-1])
        in_session = bool(sess.iloc[-1])
        high_sweep = bool(sweep.high_sweep.iloc[-1])
        low_sweep = bool(sweep.low_sweep.iloc[-1])
        prev_high = float(prev_day.high.iloc[-1])
        prev_low = float(prev_day.low.iloc[-1])
        close_price = float(bars.close.iloc[-1])

        # Filter: session and ADX regime
        if not in_session or curr_adx < self.params["adx_thresh"]:
            return None

        # Entry logic
        sl_dist = max(5.0, curr_atr * self.params["sl_mult"])
        tp_dist = max(1.5, curr_atr * self.params["tp_mult"])

        if high_sweep and close_price > prev_high:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)
        elif low_sweep and close_price < prev_low:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None