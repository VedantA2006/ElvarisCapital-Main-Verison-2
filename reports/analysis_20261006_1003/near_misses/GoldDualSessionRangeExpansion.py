class Strategy:
    PARAMS = {
        "atr_period": {"default": 14, "min": 8, "max": 30},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0},
        "range_lookback": {"default": 6, "min": 3, "max": 12}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Compute ATR and validate
        atr_val = atr(bars, self.params["atr_period"]).iloc[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        # Check we are in NY session for entry eligibility
        in_ny = is_session(bars, "ny").iloc[-1]
        if not in_ny:
            return None

        # Get London session range (strictly from closed bars)
        london_range = session_range(bars, "london")
        if london_range is None or len(london_range) == 0:
            return None

        # Ensure London range is from a completed session (not current)
        # session_range returns dict with 'high' and 'low' Series aligned to bars
        london_high = london_range["high"].iloc[-1]
        london_low = london_range["low"].iloc[-1]
        if np.isnan(london_high) or np.isnan(london_low):
            return None

        # Current closed bar close
        close = bars["close"].iloc[-1]

        # Entry logic: breakout of London range during NY
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(8.0, float(atr_val * self.params["tp_mult"]))

        if close > london_high:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)
        elif close < london_low:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None