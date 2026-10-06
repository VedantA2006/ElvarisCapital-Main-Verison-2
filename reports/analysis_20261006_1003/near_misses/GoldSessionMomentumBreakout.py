class Strategy:
    PARAMS = {
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0},
        "session_window_minutes": {"default": 120, "min": 60, "max": 180},
        "min_atr_usd": {"default": 3.0, "min": 1.5, "max": 6.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        from core.indicators import atr, is_session, minutes_since_session_open, session_range
        import numpy as np

        # Check session filter
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Only trade within configured window after session open
        mins_since_open = float(minutes_since_session_open(bars).iloc[-1])
        if mins_since_open > self.params["session_window_minutes"]:
            return None

        # Get current session range (uses active session automatically)
        sess = session_range(bars, "london" if in_london else "ny")
        sess_high = float(sess.high.iloc[-1])
        sess_low = float(sess.low.iloc[-1])
        close = float(bars["close"].iloc[-1])
        prev_close = float(bars["close"].iloc[-2])

        # Volatility filter
        atr_series = atr(bars, period=self.params["atr_period"])
        atr_val = float(atr_series.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        if atr_val < self.params["min_atr_usd"]:
            return None

        sl_dist = max(5.0, atr_val * self.params["sl_mult"])
        tp_dist = max(7.5, atr_val * self.params["tp_mult"])

        # Breakout logic: current bar closes beyond session extreme, previous bar did not
        if close > sess_high and prev_close <= sess_high:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)
        if close < sess_low and prev_close >= sess_low:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None