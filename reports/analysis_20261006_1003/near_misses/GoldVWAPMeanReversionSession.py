class Strategy:
    PARAMS = {
        "entry_threshold": {"default": 1.8, "min": 1.0, "max": 3.0},
        "rsi_period": {"default": 14, "min": 7, "max": 21},
        "rsi_oversold": {"default": 35, "min": 20, "max": 45},
        "rsi_overbought": {"default": 65, "min": 55, "max": 80},
        "sl_mult": {"default": 2.0, "min": 1.0, "max": 3.5},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        from core.indicators import session_vwap, atr, rsi, is_session
        import numpy as np

        # Get current values
        vwap_series = session_vwap(bars)
        atr_series = atr(bars, period=14)
        rsi_series = rsi(bars, period=self.params["rsi_period"])
        
        vwap_val = float(vwap_series.iloc[-1])
        atr_val = float(atr_series.iloc[-1])
        rsi_val = float(rsi_series.iloc[-1])
        close_val = float(bars["close"].iloc[-1])

        # Safety checks
        if np.isnan(vwap_val) or np.isnan(atr_val) or np.isnan(rsi_val):
            return None
        if atr_val <= 0:
            return None

        # Session filter: only trade during London or NY
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Calculate distances
        deviation = close_val - vwap_val
        threshold_dist = self.params["entry_threshold"] * atr_val

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(atr_val * self.params["tp_mult"]))

        # Long entry: price significantly below VWAP + RSI oversold
        if deviation < -threshold_dist and rsi_val < self.params["rsi_oversold"]:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short entry: price significantly above VWAP + RSI overbought
        if deviation > threshold_dist and rsi_val > self.params["rsi_overbought"]:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None