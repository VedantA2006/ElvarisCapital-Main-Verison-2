class Strategy:
    PARAMS = {
        "swing_n": {"default": 3, "min": 2, "max": 6},
        "ema_period": {"default": 20, "min": 10, "max": 50},
        "atr_period": {"default": 14, "min": 7, "max": 30},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Compute indicators
        high_sweep, low_sweep = liquidity_sweep(bars, swing_n=int(self.params["swing_n"]))
        ema_val = ema(bars, n=int(self.params["ema_period"])).iloc[-1]
        atr_series = atr(bars, n=int(self.params["atr_period"]))
        atr_val = atr_series.iloc[-1]
        session_active = is_session(bars, "london") | is_session(bars, "ny")
        current_session = session_active.iloc[-1]

        # Safety checks
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        if np.isnan(ema_val):
            return None

        # Ensure minimum trade frequency by avoiding overly strict filters
        # Only require session + sweep + EMA alignment
        last_close = bars["close"].iloc[-1]

        # Long setup
        if low_sweep.iloc[-1] and last_close > ema_val and current_session:
            sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
            tp_dist = max(1.5, float(atr_val * self.params["tp_mult"]))
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short setup
        if high_sweep.iloc[-1] and last_close < ema_val and current_session:
            sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
            tp_dist = max(1.5, float(atr_val * self.params["tp_mult"]))
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None