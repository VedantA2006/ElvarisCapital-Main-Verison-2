{
  "spec": {
    "name": "GoldVolRegimeWmaOrderBlock",
    "timeframe": "4h",
    "hypothesis": "Gold is heavily driven by institutional macro flows that persist through trending regimes but reverse violently when liquidity is exhausted. By filtering trades strictly to high-volatility trending regimes, we avoid the mean-reverting chop that destroys trend-following systems. Within these active regimes, entering on confirmed Order Blocks (institutional footprints) aligned with a fast WMA slope ensures we are trading in the direction of smart money accumulation/distribution. The losing side consists of retail traders attempting to fade strong momentum moves during high-impact news cycles or low-liquidity traps.",
    "concept_family": "trend",
    "indicators_used": [
      "wma",
      "atr",
      "order_block",
      "vol_regime",
      "trend_range_regime"
    ],
    "entry_logic": "Long: vol_regime is 'high_vol', trend_range_regime is 'trend', WMA(20) is rising (current > previous), and a bullish order block is confirmed on the current bar. Short: vol_regime is 'high_vol', trend_range_regime is 'trend', WMA(20) is falling (current < previous), and a bearish order block is confirmed on the current bar.",
    "exit_logic": "Stop loss is set at 1.5x ATR(14) with a minimum distance of 5.0 USD points. Take profit is set at 2.5x ATR(14) with a minimum distance of 7.5 USD points.",
    "filters": [
      "vol_regime == 'high_vol'",
      "trend_range_regime == 'trend'"
    ],
    "session_filter": "none",
    "direction": "both",
    "parameters": {
      "wma_period": {
        "default": 20,
        "min": 10,
        "max": 40
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
      }
    },
    "expected_trades_per_year": 65,
    "expected_failure_conditions": "Extended periods of low volatility and range-bound price action where regime filters remain inactive, leading to zero trades."
  },
  "code": "import numpy as np\nimport pandas as pd\n\nclass Strategy:\n    PARAMS = {\n        \"wma_period\": {\"default\": 20, \"min\": 10, \"max\": 40},\n        \"atr_period\": {\"default\": 14, \"min\": 7, \"max\": 21},\n        \"sl_mult\": {\"default\": 1.5, \"min\": 1.0, \"max\": 3.0},\n        \"tp_mult\": {\"default\": 2.5, \"min\": 1.5, \"max\": 4.0}\n    }\n\n    def __init__(self, params):\n        self.params = params\n\n    def on_bar(self, bars: pd.DataFrame):\n        if len(bars) < 60:\n            return None\n\n        from core.indicators import wma, atr, order_block, vol_regime, trend_range_regime\n\n        wma_period = int(self.params[\"wma_period\"])\n        atr_period = int(self.params[\"atr_period\"])\n\n        # Compute indicators using exact API signatures\n        wma_series = wma(bars, period=wma_period)\n        atr_series = atr(bars, period=atr_period)\n        bull_ob, bear_ob = order_block(bars, swing_n=2)\n        v_regime = vol_regime(bars[\"high\"], bars[\"low\"], bars[\"close\"], atr_period=atr_period, lookback=100)\n        t_regime = trend_range_regime(bars[\"high\"], bars[\"low\"], bars[\"close\"], adx_period=14, er_period=10)\n\n        # Extract latest values safely\n        atr_val = float(atr_series.iloc[-1])\n        if np.isnan(atr_val) or atr_val <= 0:\n            return None\n\n        # Regime Filters - use string comparison safely\n        v_state = str(v_regime.iloc[-1]).strip().lower()\n        t_state = str(t_regime.iloc[-1]).strip().lower()\n        if v_state != \"high_vol\":\n            return None\n        if t_state != \"trend\":\n            return None\n\n        # WMA Slope\n        wma_curr = float(wma_series.iloc[-1])\n        wma_prev = float(wma_series.iloc[-2])\n        if np.isnan(wma_curr) or np.isnan(wma_prev):\n            return None\n\n        sl_dist = max(5.0, float(atr_val * self.params[\"sl_mult\"]))\n        tp_dist = max(7.5, float(atr_val * self.params[\"tp_mult\"]))\n\n        # Safely extract boolean order block states\n        bull_ob_val = bool(bull_ob.iloc[-1]) if not pd.isna(bull_ob.iloc[-1]) else False\n        bear_ob_val = bool(bear_ob.iloc[-1]) if not pd.isna(bear_ob.iloc[-1]) else False\n\n        # Long Entry\n        if wma_curr > wma_prev and bull_ob_val:\n            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)\n\n        # Short Entry\n        if wma_curr < wma_prev and bear_ob_val:\n            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)\n\n        return None"
}