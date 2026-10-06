class Strategy:
    PARAMS = {
        "ema_fast": {"default": 21, "min": 10, "max": 50},
        "ema_slow": {"default": 55, "min": 30, "max": 100},
        "vol_lookback": {"default": 60, "min": 20, "max": 120},
        "min_vol_pct": {"default": 40, "min": 20, "max": 70},
        "sl_mult": {"default": 2.0, "min": 1.0, "max": 4.0},
        "tp_mult": {"default": 3.0, "min": 1.5, "max": 6.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Compute indicators
        ema_f = ema(bars, self.params["ema_fast"])
        ema_s = ema(bars, self.params["ema_slow"])
        macd_line, signal_line, hist = macd(bars, 12, 26, 9)
        vol_pct = vol_percentile(bars, atr_n=14, lookback=self.params["vol_lookback"])
        atr_val = atr(bars, 14)
        in_session = is_session(bars, "london_ny")

        # Safety checks for NaN/invalid values
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        if np.isnan(vol_pct) or np.isnan(hist) or np.isnan(ema_f.iloc[-1]) or np.isnan(ema_s.iloc[-1]):
            return None

        # Current bar values
        curr_ema_f = float(ema_f.iloc[-1])
        curr_ema_s = float(ema_s.iloc[-1])
        curr_hist = float(hist.iloc[-1])
        curr_vol_pct = float(vol_pct.iloc[-1])
        curr_in_session = bool(in_session.iloc[-1])

        # Parameter extraction
        min_vol = self.params["min_vol_pct"] / 100.0
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(atr_val * self.params["tp_mult"]))

        # Entry logic
        long_cond = (curr_ema_f > curr_ema_s) and (curr_hist > 0) and (curr_vol_pct > min_vol) and curr_in_session
        short_cond = (curr_ema_f < curr_ema_s) and (curr_hist < 0) and (curr_vol_pct > min_vol) and curr_in_session

        if long_cond:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)
        elif short_cond:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None