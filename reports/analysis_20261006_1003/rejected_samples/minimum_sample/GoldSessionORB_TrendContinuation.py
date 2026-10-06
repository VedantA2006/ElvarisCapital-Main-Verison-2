import numpy as np
import pandas as pd
from core.indicators import htf, ema, bos, atr, is_session, swing_high, swing_low

class Strategy:
    PARAMS = {
        "swing_n": {"default": 5, "min": 3, "max": 10},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_rr": {"default": 2.0, "min": 1.5, "max": 4.0},
        "cooldown": {"default": 4, "min": 2, "max": 8}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999

    def on_bar(self, bars: pd.DataFrame):
        # Warmup safety
        if len(bars) < 60:
            return None

        curr_idx = len(bars) - 1

        # Cooldown enforcement
        if (curr_idx - self.last_trade_bar) < self.params["cooldown"]:
            return None

        # Session filter: London or New York
        in_london = is_session(bars, "london").iloc[-1]
        in_ny = is_session(bars, "ny").iloc[-1]
        if not (in_london or in_ny):
            return None

        # 4H Trend Bias via HTF EMA slope
        try:
            htf_ema = htf(bars, "4h", lambda x: ema(x, 20))
            if htf_ema is None or len(htf_ema) < 3:
                return None
            # Ensure no NaNs in recent values
            if np.isnan(htf_ema.iloc[-1]) or np.isnan(htf_ema.iloc[-2]):
                return None
            htf_slope = htf_ema.iloc[-1] - htf_ema.iloc[-2]
        except Exception:
            return None

        # 1H Break of Structure
        swing_n = int(self.params["swing_n"])
        bull_bos, bear_bos = bos(bars, swing_n)

        if bull_bos is None or bear_bos is None:
            return None
        if len(bull_bos) == 0 or len(bear_bos) == 0:
            return None

        last_bull = bool(bull_bos.iloc[-1])
        last_bear = bool(bear_bos.iloc[-1])

        # ATR for dynamic stop loss
        atr_series = atr(bars, 14)
        if atr_series is None or len(atr_series) == 0:
            return None
        atr_val = float(atr_series.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(sl_dist * self.params["tp_rr"]))

        # Entry Logic
        if htf_slope > 0 and last_bull:
            self.last_trade_bar = curr_idx
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        elif htf_slope < 0 and last_bear:
            self.last_trade_bar = curr_idx
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None