class Strategy:
    PARAMS = {
        "macd_fast": {"default": 12, "min": 8, "max": 20},
        "macd_slow": {"default": 26, "min": 20, "max": 40},
        "macd_signal": {"default": 9, "min": 5, "max": 15},
        "htf_ema_period": {"default": 50, "min": 20, "max": 100},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        from core.indicators import macd, ema, htf, atr, is_session
        import numpy as np

        # Session filter: London or NY only
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Higher Timeframe Trend Filter (4H EMA)
        # Using closed-bar HTF to avoid lookahead
        htf_ema_val = htf(
            bars,
            target_tf="4h",
            func=lambda df: ema(df, period=self.params["htf_ema_period"])
        )
        
        # Need at least 2 valid HTF points for slope
        if htf_ema_val.iloc[-1] is None or htf_ema_val.iloc[-2] is None:
            return None
            
        htf_curr = float(htf_ema_val.iloc[-1])
        htf_prev = float(htf_ema_val.iloc[-2])
        
        if np.isnan(htf_curr) or np.isnan(htf_prev):
            return None

        htf_trend_up = htf_curr > htf_prev
        htf_trend_down = htf_curr < htf_prev

        # 1H Indicators
        macd_res = macd(
            bars,
            fast_period=self.params["macd_fast"],
            slow_period=self.params["macd_slow"],
            signal_period=self.params["macd_signal"]
        )
        hist = macd_res.hist
        
        ema_200 = ema(bars, period=200)
        atr_series = atr(bars, period=14)

        # Safety checks for NaN/Invalid values
        curr_hist = float(hist.iloc[-1])
        prev_hist = float(hist.iloc[-2])
        curr_close = float(bars["close"].iloc[-1])
        curr_ema200 = float(ema_200.iloc[-1])
        atr_val = float(atr_series.iloc[-1])

        if np.isnan(curr_hist) or np.isnan(prev_hist) or np.isnan(curr_ema200) or np.isnan(atr_val) or atr_val <= 0:
            return None

        # Risk Management Distances
        sl_dist = max(5.0, atr_val * self.params["sl_mult"])
        tp_dist = max(7.5, atr_val * self.params["tp_mult"])

        # Long Entry: HTF Up + MACD Hist Zero Cross Up + Price > EMA200
        if htf_trend_up and curr_close > curr_ema200:
            if prev_hist <= 0 and curr_hist > 0:
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short Entry: HTF Down + MACD Hist Zero Cross Down + Price < EMA200
        if htf_trend_down and curr_close < curr_ema200:
            if prev_hist >= 0 and curr_hist < 0:
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None