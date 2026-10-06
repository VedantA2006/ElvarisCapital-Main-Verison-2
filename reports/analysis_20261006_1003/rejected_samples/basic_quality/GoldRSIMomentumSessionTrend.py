class Strategy:
    PARAMS = {
        "rsi_period": {"default": 14, "min": 8, "max": 21},
        "ema_period": {"default": 21, "min": 10, "max": 50},
        "atr_period": {"default": 14, "min": 10, "max": 30},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        rsi_series = rsi(bars, self.params["rsi_period"])
        ema_series = ema(bars, self.params["ema_period"])
        atr_series = atr(bars, self.params["atr_period"])
        in_session = is_session(bars, "london") | is_session(bars, "ny")

        curr_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2]
        curr_ema = ema_series.iloc[-1]
        curr_atr = atr_series.iloc[-1]
        curr_in_session = in_session.iloc[-1]

        if np.isnan(curr_rsi) or np.isnan(prev_rsi) or np.isnan(curr_ema) or np.isnan(curr_atr) or curr_atr <= 0:
            return None

        if not curr_in_session:
            return None

        sl_dist = max(5.0, float(curr_atr * self.params["sl_mult"]))
        tp_dist = max(1.5, float(curr_atr * self.params["tp_mult"]))

        # Long: RSI crosses above 50 and price above EMA
        if prev_rsi <= 50.0 and curr_rsi > 50.0 and bars["close"].iloc[-1] > curr_ema:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short: RSI crosses below 50 and price below EMA
        if prev_rsi >= 50.0 and curr_rsi < 50.0 and bars["close"].iloc[-1] < curr_ema:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None