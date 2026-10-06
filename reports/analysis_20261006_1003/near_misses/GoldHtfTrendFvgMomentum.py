class Strategy:
    PARAMS = {
        "htf_ema_period": {"default": 50, "min": 20, "max": 100},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0},
        "fvg_min_gap": {"default": 1.0, "min": 0.0, "max": 5.0}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        from core.indicators import htf, ema, fvg, atr, is_session
        import numpy as np

        # Higher timeframe trend filter (4H EMA)
        htf_ema_series = htf(
            bars,
            target_tf="4h",
            func=lambda df: ema(df, period=self.params["htf_ema_period"])
        )
        if htf_ema_series is None or len(htf_ema_series) < 2:
            return None

        htf_ema_curr = float(htf_ema_series.iloc[-1])
        htf_ema_prev = float(htf_ema_series.iloc[-2])
        if np.isnan(htf_ema_curr) or np.isnan(htf_ema_prev):
            return None

        # Current price vs HTF EMA
        curr_close = float(bars["close"].iloc[-1])

        # FVG detection on 1H
        min_gap = float(self.params["fvg_min_gap"])
        bull_fvg, bear_fvg = fvg(bars, min_gap_usd=min_gap)
        has_bull_fvg = bool(bull_fvg.iloc[-1])
        has_bear_fvg = bool(bear_fvg.iloc[-1])

        # Session filter
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # ATR for risk management
        atr_series = atr(bars, period=self.params["atr_period"])
        atr_val = float(atr_series.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        sl_dist = max(5.0, atr_val * self.params["sl_mult"])
        tp_dist = max(7.5, atr_val * self.params["tp_mult"])

        # Trend alignment: EMA slope + price position
        htf_rising = htf_ema_curr > htf_ema_prev
        htf_falling = htf_ema_curr < htf_ema_prev
        price_above_htf = curr_close > htf_ema_curr
        price_below_htf = curr_close < htf_ema_curr

        # Long: HTF uptrend + bullish FVG + session
        if htf_rising and price_above_htf and has_bull_fvg:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short: HTF downtrend + bearish FVG + session
        if htf_falling and price_below_htf and has_bear_fvg:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None