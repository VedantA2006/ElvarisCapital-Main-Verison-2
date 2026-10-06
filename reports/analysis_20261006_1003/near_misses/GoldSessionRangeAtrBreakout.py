class Strategy:
    PARAMS = {
        "atr_period": {"default": 14, "min": 8, "max": 30},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0},
        "cooldown_bars": {"default": 3, "min": 1, "max": 6}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999
        self.in_position = False

    def on_bar(self, bars):
        if len(bars) < 60:
            return None

        # Cooldown check
        current_bar_idx = len(bars) - 1
        if current_bar_idx - self.last_trade_bar < self.params["cooldown_bars"]:
            return None

        # Get indicators
        atr_series = atr(bars, self.params["atr_period"])
        atr_val = float(atr_series.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        # Session filter: only trade in London or NY
        london_mask = is_session(bars, "london")
        ny_mask = is_session(bars, "ny")
        active_session = bool(london_mask.iloc[-1] or ny_mask.iloc[-1])
        if not active_session:
            return None

        # Get Asian session range (strictly closed data)
        asian_range = session_range(bars, "asia")
        asian_high = float(asian_range["high"].iloc[-1])
        asian_low = float(asian_range["low"].iloc[-1])
        if np.isnan(asian_high) or np.isnan(asian_low) or asian_high <= asian_low:
            return None

        close_price = float(bars["close"].iloc[-1])

        # Entry logic
        if not self.in_position:
            # Long breakout
            if close_price > asian_high:
                sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
                tp_dist = max(8.0, float(atr_val * self.params["tp_mult"]))
                self.in_position = True
                self.last_trade_bar = current_bar_idx
                return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

            # Short breakout
            elif close_price < asian_low:
                sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))
                tp_dist = max(8.0, float(atr_val * self.params["tp_mult"]))
                self.in_position = True
                self.last_trade_bar = current_bar_idx
                return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        else:
            # Position management handled by platform via SL/TP
            # No manual close signals needed unless adding trailing logic
            pass

        return None