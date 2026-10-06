{
  "spec": {
    "name": "GoldHtfEmaStochSessionMomentum",
    "timeframe": "1h",
    "hypothesis": "XAUUSD exhibits strong intraday momentum persistence when aligned with the higher timeframe (4H) trend, particularly during London/NY overlap. Institutional flow dominates gold pricing; retail traders often fade 1H moves against the 4H trend, providing liquidity for continuation entries. This strategy exploits that misalignment by entering only on stochastic pullbacks in the direction of the confirmed 4H EMA trend during high-liquidity sessions, capturing the resumption of institutional order flow before retail re-engages.",
    "concept_family": "trend",
    "indicators_used": [
      "htf",
      "ema",
      "stoch",
      "atr",
      "is_session"
    ],
    "entry_logic": "Long: 4H close > 4H EMA(50), 1H Stochastic Slow K crosses above Slow D from below 30, and current bar is in London or NY session. Short: 4H close < 4H EMA(50), 1H Stochastic Slow K crosses below Slow D from above 70, and current bar is in London or NY session.",
    "exit_logic": "Stop loss set to max(5.0, ATR(14) * sl_mult). Take profit set to max(7.5, ATR(14) * tp_mult). Both distances are positive USD points.",
    "filters": [
      "4H trend alignment via HTF EMA",
      "Stochastic crossover confirmation",
      "London or NY session filter"
    ],
    "session_filter": "london_ny",
    "direction": "both",
    "parameters": {
      "htf_ema_period": {
        "default": 50,
        "min": 30,
        "max": 80
      },
      "stoch_k": {
        "default": 14,
        "min": 8,
        "max": 21
      },
      "stoch_d": {
        "default": 3,
        "min": 2,
        "max": 5
      },
      "atr_period": {
        "default": 14,
        "min": 10,
        "max": 20
      },
      "sl_mult": {
        "default": 1.5,
        "min": 1.0,
        "max": 2.5
      },
      "tp_mult": {
        "default": 2.5,
        "min": 1.8,
        "max": 4.0
      }
    },
    "expected_trades_per_year": 65,
    "expected_failure_conditions": "Extended 4H ranging markets where HTF EMA flattens and generates false trend signals; low-volatility Asian-session-dominated weeks with insufficient session overlap triggers."
  },
  "code": "import numpy as np\nimport pandas as pd\nfrom core.indicators import htf, ema, stoch, atr, is_session\n\nclass Strategy:\n    PARAMS = {\n        \"htf_ema_period\": {\"default\": 50, \"min\": 30, \"max\": 80},\n        \"stoch_k\": {\"default\": 14, \"min\": 8, \"max\": 21},\n        \"stoch_d\": {\"default\": 3, \"min\": 2, \"max\": 5},\n        \"atr_period\": {\"default\": 14, \"min\": 10, \"max\": 20},\n        \"sl_mult\": {\"default\": 1.5, \"min\": 1.0, \"max\": 2.5},\n        \"tp_mult\": {\"default\": 2.5, \"min\": 1.8, \"max\": 4.0}\n    }\n\n    def __init__(self, params):\n        self.params = params\n\n    def on_bar(self, bars: pd.DataFrame):\n        if len(bars) < 60:\n            return None\n\n        # Higher timeframe trend filter: 4H close and 4H EMA\n        htf_close = htf(bars, target_tf=\"4h\", func=lambda df: df[\"close\"])\n        htf_ema_val = htf(bars, target_tf=\"4h\", func=lambda df: ema(df, period=self.params[\"htf_ema_period\"]))\n\n        htf_c_curr = float(htf_close.iloc[-1])\n        htf_e_curr = float(htf_ema_val.iloc[-1])\n\n        if np.isnan(htf_c_curr) or np.isnan(htf_e_curr):\n            return None\n\n        htf_trend_up = htf_c_curr > htf_e_curr\n        htf_trend_down = htf_c_curr < htf_e_curr\n\n        # Stochastic oscillator on 1H\n        stoch_result = stoch(bars, k_period=self.params[\"stoch_k\"], d_period=self.params[\"stoch_d\"], slowing=3)\n        slow_k = stoch_result.slow_k\n        slow_d = stoch_result.slow_d\n\n        # ATR for risk sizing\n        atr_series = atr(bars, period=self.params[\"atr_period\"])\n        atr_val = float(atr_series.iloc[-1])\n        if np.isnan(atr_val) or atr_val <= 0:\n            return None\n\n        # Session filter\n        in_london = bool(is_session(bars, \"london\").iloc[-1])\n        in_ny = bool(is_session(bars, \"ny\").iloc[-1])\n        if not (in_london or in_ny):\n            return None\n\n        # Risk distances\n        sl_dist = max(5.0, float(atr_val * self.params[\"sl_mult\"]))\n        tp_dist = max(7.5, float(atr_val * self.params[\"tp_mult\"]))\n\n        # Entry conditions\n        k_curr = float(slow_k.iloc[-1])\n        k_prev = float(slow_k.iloc[-2])\n        d_curr = float(slow_d.iloc[-1])\n        d_prev = float(slow_d.iloc[-2])\n\n        if np.isnan(k_curr) or np.isnan(k_prev) or np.isnan(d_curr) or np.isnan(d_prev):\n            return None\n\n        # Long: HTF uptrend + stochastic bullish cross from oversold\n        if htf_trend_up and k_prev <= d_prev and k_curr > d_curr and k_prev < 30:\n            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)\n\n        # Short: HTF downtrend + stochastic bearish cross from overbought\n        if htf_trend_down and k_prev >= d_prev and k_curr < d_curr and k_prev > 70:\n            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)\n\n        return None"
}