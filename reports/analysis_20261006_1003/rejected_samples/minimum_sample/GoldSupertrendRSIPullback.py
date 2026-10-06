class Strategy:
    PARAMS = {
        "st_period": {"default": 10, "min": 5, "max": 20},
        "st_mult": {"default": 3.0, "min": 1.5, "max": 5.0},
        "rsi_period": {"default": 10, "min": 5, "max": 21},
        "rsi_pullback": {"default": 45, "min": 30, "max": 50},
        "sl_mult": {"default": 2.0, "min": 1.0, "max": 4.0},
        "tp_mult": {"default": 3.0, "min": 1.5, "max": 6.0}
    }

    def __init__(self, params):
        self.params = params
        self.last_trade_bar = -999

    def on_bar(self, bars):
        # Warmup safety
        if len(bars) < 60:
            return None

        # Session Filter: Only trade during London or NY sessions
        if not (is_session(bars, "london").iloc[-1] or is_session(bars, "ny").iloc[-1]):
            return None

        # Cooldown to prevent overtrading and ensure minimum sample size distribution
        if len(bars) - self.last_trade_bar < 3:
            return None

        # Calculate indicators using strictly closed bars
        st_dir, st_val = supertrend(bars, int(self.params["st_period"]), float(self.params["st_mult"]))
        rsi_val = rsi(bars, int(self.params["rsi_period"]))
        atr_val = atr(bars, 14)

        # Extract current values safely
        curr_st_dir = float(st_dir.iloc[-1])
        curr_rsi = float(rsi_val.iloc[-1])
        curr_atr = float(atr_val.iloc[-1])

        # NaN and zero safety checks
        if np.isnan(curr_st_dir) or np.isnan(curr_rsi) or np.isnan(curr_atr) or curr_atr <= 0:
            return None

        # Dynamic SL/TP distances ensuring positive USD points
        sl_dist = max(5.0, float(curr_atr * self.params["sl_mult"]))
        tp_dist = max(1.50, float(curr_atr * self.params["tp_mult"]))

        # Entry Logic
        rsi_thresh = float(self.params["rsi_pullback"])

        # Long Setup: Trend is up, RSI pulled back to discount zone
        if curr_st_dir == 1.0 and curr_rsi < rsi_thresh:
            self.last_trade_bar = len(bars)
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)

        # Short Setup: Trend is down, RSI pulled back to premium zone
        if curr_st_dir == -1.0 and curr_rsi > (100.0 - rsi_thresh):
            self.last_trade_bar = len(bars)
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)

        # Exit Logic: Close position if Supertrend flips against us
        # Assuming framework tracks state; if flat, this returns None implicitly
        if hasattr(self, '_position_direction'):
            if self._position_direction == 1 and curr_st_dir == -1.0:
                return Signal.close()
            if self._position_direction == -1 and curr_st_dir == 1.0:
                return Signal.close()

        return None