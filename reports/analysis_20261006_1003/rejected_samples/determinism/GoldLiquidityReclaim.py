class Strategy:
    PARAMS = {
        "swing_lookback": {"default": 5, "min": 3, "max": 10},
        "atr_period": {"default": 14, "min": 10, "max": 30},
        "sl_atr_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_usd": {"default": 12.0, "min": 6.0, "max": 25.0},
        "vol_percentile_min": {"default": 30, "min": 10, "max": 50}
    }

    def __init__(self, params):
        self.params = params
        self.last_signal_bar = -1

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Avoid duplicate signals on same bar
        current_bar_idx = len(bars) - 1
        if current_bar_idx == self.last_signal_bar:
            return None

        # Session filter
        if not is_session(bars, "london_ny").iloc[-1]:
            return None

        # Volatility filter
        vol_pct = vol_percentile(
            bars["high"], bars["low"], bars["close"],
            atr_n=self.params["atr_period"],
            lookback=100
        ).iloc[-1]
        if np.isnan(vol_pct) or vol_pct < self.params["vol_percentile_min"]:
            return None

        # ATR for stop distance
        atr_val = atr(
            bars["high"], bars["low"], bars["close"],
            n=self.params["atr_period"]
        ).iloc[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        swing_n = int(self.params["swing_lookback"])

        # Detect liquidity sweeps
        sweep_df = liquidity_sweep(bars, swing_n=swing_n)
        last_sweep = sweep_df.iloc[-1]

        # Check if sweep occurred in last 2 bars (current or previous closed bar)
        sweep_recency = current_bar_idx - last_sweep.name
        if sweep_recency > 1:
            return None

        close_price = bars["close"].iloc[-1]
        sl_mult = float(self.params["sl_atr_mult"])
        tp_dist = float(self.params["tp_usd"])

        # Long setup: sweep of swing low + reclaim
        if last_sweep["type"] == "low":
            swing_level = last_sweep["level"]
            # Reclaim: current close > swept swing low
            if close_price > swing_level:
                sl_dist = max(5.0, float(atr_val * sl_mult))
                self.last_signal_bar = current_bar_idx
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short setup: sweep of swing high + reclaim
        elif last_sweep["type"] == "high":
            swing_level = last_sweep["level"]
            # Reclaim: current close < swept swing high
            if close_price < swing_level:
                sl_dist = max(5.0, float(atr_val * sl_mult))
                self.last_signal_bar = current_bar_idx
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None