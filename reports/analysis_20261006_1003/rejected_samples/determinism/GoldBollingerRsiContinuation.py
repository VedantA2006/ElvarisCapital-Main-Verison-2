import numpy as np
import pandas as pd
from core.indicators import bollinger, rsi, atr

class Strategy:
    PARAMS = {
        "bb_period": {"default": 20, "min": 10, "max": 50},
        "bb_std": {"default": 2.0, "min": 1.5, "max": 3.0},
        "rsi_period": {"default": 14, "min": 5, "max": 30},
        "atr_period": {"default": 14, "min": 7, "max": 30},
        "sl_mult": {"default": 1.5, "min": 0.5, "max": 3.0},
        "tp_mult": {"default": 2.0, "min": 1.0, "max": 4.0}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999

    def on_bar(self, bars: pd.DataFrame):
        if len(bars) < 60:
            return None

        i = len(bars) - 1
        cooldown = 3
        if (i - self.last_trade_bar) < cooldown:
            return None

        bb_period = int(self.params["bb_period"])
        bb_std = float(self.params["bb_std"])
        rsi_period = int(self.params["rsi_period"])
        atr_period = int(self.params["atr_period"])

        bb = bollinger(bars, bb_period, bb_std)
        r = rsi(bars, rsi_period)
        a = atr(bars, atr_period)

        close_val = float(bars["close"].iloc[-1])
        upper_val = float(bb["upper"].iloc[-1])
        lower_val = float(bb["lower"].iloc[-1])
        rsi_val = float(r.iloc[-1])
        atr_val = float(a.iloc[-1])

        if np.isnan(upper_val) or np.isnan(lower_val) or np.isnan(rsi_val) or np.isnan(atr_val) or atr_val <= 0:
            return None

        sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
        tp_dist = max(1.50, float(sl_dist * self.params["tp_mult"]))

        # Long Entry: Close breaks above Upper BB + RSI confirms bullish momentum
        if close_val > upper_val and rsi_val > 50.0:
            self.last_trade_bar = i
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short Entry: Close breaks below Lower BB + RSI confirms bearish momentum
        if close_val < lower_val and rsi_val < 50.0:
            self.last_trade_bar = i
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        return None