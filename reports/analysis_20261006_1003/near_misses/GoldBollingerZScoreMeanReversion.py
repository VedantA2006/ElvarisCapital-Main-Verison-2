class Strategy:
    PARAMS = {
        "bb_period": {"default": 20, "min": 10, "max": 40},
        "bb_std": {"default": 2.0, "min": 1.5, "max": 3.0},
        "z_period": {"default": 20, "min": 10, "max": 40},
        "z_thresh": {"default": 1.8, "min": 1.2, "max": 2.5},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0}
    }

    def __init__(self, params):
        self.params = params
        self.cooldown = 0

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        from core.indicators import bollinger, zscore, atr, is_session, session_vwap

        # Check active session
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        bb_period = int(self.params["bb_period"])
        bb_std = float(self.params["bb_std"])
        z_period = int(self.params["z_period"])
        z_thresh = float(self.params["z_thresh"])
        atr_period = int(self.params["atr_period"])
        sl_mult = float(self.params["sl_mult"])

        # Calculate indicators
        bb = bollinger(bars, period=bb_period, num_std=bb_std)
        z = zscore(bars, period=z_period)
        atr_series = atr(bars, period=atr_period)
        vwap_series = session_vwap(bars)

        # Extract current values safely
        atr_val = float(atr_series.iloc[-1])
        z_val = float(z.iloc[-1])
        close_val = float(bars["close"].iloc[-1])
        bb_upper = float(bb.upper.iloc[-1])
        bb_lower = float(bb.lower.iloc[-1])
        vwap_val = float(vwap_series.iloc[-1])

        # NaN / Zero safety checks
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        if np.isnan(z_val) or np.isnan(vwap_val) or np.isnan(bb_upper) or np.isnan(bb_lower):
            return None

        # Distances
        sl_dist = max(5.0, atr_val * sl_mult)

        # Entry Logic: Mean Reversion
        # Long: Price pierced lower BB and Z-score is deeply negative
        if close_val < bb_lower and z_val <= -z_thresh:
            # TP targets partial reversion towards VWAP
            tp_dist_raw = abs(vwap_val - close_val) * 0.6
            tp_dist = max(7.5, tp_dist_raw)
            self.cooldown = 3
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short: Price pierced upper BB and Z-score is deeply positive
        if close_val > bb_upper and z_val >= z_thresh:
            # TP targets partial reversion towards VWAP
            tp_dist_raw = abs(close_val - vwap_val) * 0.6
            tp_dist = max(7.5, tp_dist_raw)
            self.cooldown = 3
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None