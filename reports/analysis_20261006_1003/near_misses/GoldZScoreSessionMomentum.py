class Strategy:
    PARAMS = {
        "zscore_period": {"default": 20, "min": 10, "max": 40},
        "zscore_threshold": {"default": 1.0, "min": 0.5, "max": 2.0},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.8, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.8, "min": 1.5, "max": 4.0},
        "vol_lookback": {"default": 100, "min": 50, "max": 200}
    }

    def __init__(self, params):
        self.params = params

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        from core.indicators import session_vwap, zscore, atr, is_session, vol_percentile
        import numpy as np

        # Compute indicators
        vwap = session_vwap(bars)
        zs = zscore(bars, period=self.params["zscore_period"])
        atr_series = atr(bars, period=self.params["atr_period"])
        vol_pct = vol_percentile(bars, lookback=self.params["vol_lookback"])

        # Safety checks
        atr_val = float(atr_series.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        zs_curr = float(zs.iloc[-1])
        zs_prev = float(zs.iloc[-2])
        vol_curr = float(vol_pct.iloc[-1])

        if np.isnan(zs_curr) or np.isnan(zs_prev) or np.isnan(vol_curr):
            return None

        # Session filter: only trade in London or NY
        in_london = bool(is_session(bars, "london").iloc[-1])
        in_ny = bool(is_session(bars, "ny").iloc[-1])
        if not (in_london or in_ny):
            return None

        # Volatility filter
        if vol_curr < 30.0:
            return None

        # Risk distances
        sl_dist = max(5.0, atr_val * self.params["sl_mult"])
        tp_dist = max(7.5, atr_val * self.params["tp_mult"])

        threshold = self.params["zscore_threshold"]

        # Long entry: Z-score crosses above +threshold
        if zs_prev <= threshold and zs_curr > threshold:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short entry: Z-score crosses below -threshold
        if zs_prev >= -threshold and zs_curr < -threshold:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None