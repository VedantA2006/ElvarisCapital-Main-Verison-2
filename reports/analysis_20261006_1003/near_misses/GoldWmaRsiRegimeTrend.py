class Strategy:
    PARAMS = {
        "wma_period": {"default": 21, "min": 10, "max": 50},
        "rsi_period": {"default": 14, "min": 7, "max": 21},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params
        self.last_signal_bar = -10

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Cooldown to prevent overtrading and ensure sample quality
        if len(bars) - self.last_signal_bar < 3:
            return None

        from core.indicators import wma, rsi, atr, trend_range_regime, is_session

        wma_s = wma(bars, period=self.params["wma_period"])
        rsi_s = rsi(bars, period=self.params["rsi_period"])
        atr_s = atr(bars, period=self.params["atr_period"])

        regime_s = trend_range_regime(
            high=bars["high"],
            low=bars["low"],
            close=bars["close"],
            adx_period=14,
            er_period=10
        )

        sess_london = is_session(bars, "london")
        sess_ny = is_session(bars, "ny")

        curr_close = float(bars["close"].iloc[-1])
        prev_close = float(bars["close"].iloc[-2])
        curr_wma = float(wma_s.iloc[-1])
        prev_wma = float(wma_s.iloc[-2])
        curr_rsi = float(rsi_s.iloc[-1])
        curr_atr = float(atr_s.iloc[-1])
        curr_regime = str(regime_s.iloc[-1])
        in_session = bool(sess_london.iloc[-1] or sess_ny.iloc[-1])

        if np.isnan(curr_atr) or curr_atr <= 0:
            return None
        if np.isnan(curr_wma) or np.isnan(curr_rsi):
            return None

        # Only trade when the broader structure is trending
        if curr_regime != "trend":
            return None

        # Only trade during active institutional hours
        if not in_session:
            return None

        sl_dist = max(5.0, float(curr_atr * self.params["sl_mult"]))
        tp_dist = max(7.5, float(curr_atr * self.params["tp_mult"]))

        # Long: Pullback below WMA followed by resumption above WMA, with healthy momentum
        if prev_close < prev_wma and curr_close > curr_wma and 40.0 < curr_rsi < 75.0:
            self.last_signal_bar = len(bars)
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short: Pullback above WMA followed by resumption below WMA, with healthy momentum
        if prev_close > prev_wma and curr_close < curr_wma and 25.0 < curr_rsi < 60.0:
            self.last_signal_bar = len(bars)
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None