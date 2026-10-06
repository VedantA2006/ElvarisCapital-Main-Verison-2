class Strategy:
    PARAMS = {
        "swing_n": {"default": 3, "min": 2, "max": 8},
        "atr_period": {"default": 14, "min": 7, "max": 30},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "rr_ratio": {"default": 2.0, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        # Warmup safety
        if len(bars) < 60:
            return None

        # Only trade during London or NY sessions
        in_london = is_session(bars, "london").iloc[-1]
        in_ny = is_session(bars, "ny").iloc[-1]
        if not (in_london or in_ny):
            return None

        # Get Asian session range
        asia_range = session_range(bars, "asia")
        if asia_range is None:
            return None
        
        asia_high = asia_range.high.iloc[-1]
        asia_low = asia_range.low.iloc[-1]

        if np.isnan(asia_high) or np.isnan(asia_low):
            return None

        swing_n = int(self.params["swing_n"])

        # Liquidity sweeps of the Asian range
        high_sweep, low_sweep = liquidity_sweep(bars, swing_n=swing_n)
        
        # Structural change of character
        bull_choch, bear_choch = choch(bars, swing_n=swing_n)

        # Volatility / ATR for risk management
        atr_series = atr(bars, int(self.params["atr_period"]))
        atr_val = float(atr_series.iloc[-1])

        if np.isnan(atr_val) or atr_val <= 0:
            return None

        # Calculate distances ensuring they are positive and meet minimums
        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(sl_dist * self.params["rr_ratio"]))

        # SHORT LOGIC: Swept Asian High + Bearish CHoCH
        if high_sweep.iloc[-1] and bear_choch.iloc[-1]:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        # LONG LOGIC: Swept Asian Low + Bullish CHoCH
        if low_sweep.iloc[-1] and bull_choch.iloc[-1]:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        return None