{
  "spec": {
    "name": "GoldVwapRegimeTrend",
    "timeframe": "1h",
    "hypothesis": "Institutional gold traders use the daily session VWAP as a benchmark for fair value. When price crosses and holds above/below VWAP during an established trending regime, it signals that institutional flow is actively defending the new direction. Retail traders who fade these moves or attempt mean-reversion against the trend get stopped out, providing the liquidity needed to fuel continuation. By filtering entries through a trend/range regime detector, we avoid false breakouts in choppy conditions where VWAP acts as a magnet rather than a support/resistance level.",
    "concept_family": "trend",
    "indicators_used": [
      "session_vwap",
      "ema",
      "atr",
      "trend_range_regime",
      "is_session"
    ],
    "entry_logic": "Long: Trend regime is 'trend', close > session VWAP, previous close <= previous VWAP (fresh cross), and close > EMA(20). Short: Trend regime is 'trend', close < session VWAP, previous close >= previous VWAP, and close < EMA(20). Cooldown of 3 bars enforced between trades.",
    "exit_logic": "Stop loss distance is ATR(14) * sl_mult (minimum 5.0 USD points). Take profit distance is ATR(14) * tp_mult (minimum 7.5 USD points).",
    "filters": [
      "trend_range_regime == 'trend'",
      "London or NY session active",
      "3-bar cooldown"
    ],
    "session_filter": "london_ny",
    "direction": "both",
    "parameters": {
      "ema_period": {
        "default": 20,
        "min": 10,
        "max": 50
      },
      "atr_period": {
        "default": 14,
        "min": 7,
        "max": 21
      },
      "sl_mult": {
        "default": 1.5,
        "min": 1.0,
        "max": 3.0
      },
      "tp_mult": {
        "default": 2.5,
        "min": 1.5,
        "max": 4.0
      },
      "adx_period": {
        "default": 14,
        "min": 7,
        "max": 21
      }
    },
    "expected_trades_per_year": 65,
    "expected_failure_conditions": "Protracted low-volatility summer chop where regime detection lags and VWAP crosses whipsaw frequently"
  },
  "code": "import numpy as np\nimport pandas as pd\nfrom core.indicators import session_vwap, ema, atr, trend_range_regime, is_session\n\n\nclass Strategy:\n    PARAMS = {\n        \"ema_period\": {\"default\": 20, \"min\": 10, \"max\": 50},\n        \"atr_period\": {\"default\": 14, \"min\": 7, \"max\": 21},\n        \"sl_mult\": {\"default\": 1.5, \"min\": 1.0, \"max\": 3.0},\n        \"tp_mult\": {\"default\": 2.5, \"min\": 1.5, \"max\": 4.0},\n        \"adx_period\": {\"default\": 14, \"min\": 7, \"max\": 21}\n    }\n\n    def __init__(self, params):\n        self.params = params\n        self.last_trade_bar = -999\n\n    def on_bar(self, bars):\n        if len(bars) < 60:\n            return None\n\n        current_idx = len(bars) - 1\n        if current_idx - self.last_trade_bar < 3:\n            return None\n\n        in_london = bool(is_session(bars, \"london\").iloc[-1])\n        in_ny = bool(is_session(bars, \"ny\").iloc[-1])\n        if not (in_london or in_ny):\n            return None\n\n        regime = trend_range_regime(\n            bars[\"high\"],\n            bars[\"low\"],\n            bars[\"close\"],\n            adx_period=self.params[\"adx_period\"],\n            er_period=10\n        )\n        if str(regime.iloc[-1]) != \"trend\":\n            return None\n\n        vwap = session_vwap(bars)\n        ema_val = ema(bars, period=self.params[\"ema_period\"])\n        atr_series = atr(bars, period=self.params[\"atr_period\"])\n\n        close_curr = float(bars[\"close\"].iloc[-1])\n        close_prev = float(bars[\"close\"].iloc[-2])\n        vwap_curr = float(vwap.iloc[-1])\n        vwap_prev = float(vwap.iloc[-2])\n        ema_curr = float(ema_val.iloc[-1])\n        atr_curr = float(atr_series.iloc[-1])\n\n        if np.isnan(vwap_curr) or np.isnan(vwap_prev) or np.isnan(ema_curr) or np.isnan(atr_curr) or atr_curr <= 0:\n            return None\n\n        sl_dist = max(5.0, float(atr_curr * self.params[\"sl_mult\"]))\n        tp_dist = max(7.5, float(atr_curr * self.params[\"tp_mult\"]))\n\n        if close_curr > vwap_curr and close_prev <= vwap_prev and close_curr > ema_curr:\n            self.last_trade_bar = current_idx\n            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)\n\n        if close_curr < vwap_curr and close_prev >= vwap_prev and close_curr < ema_curr:\n            self.last_trade_bar = current_idx\n            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)\n\n        return None"
}