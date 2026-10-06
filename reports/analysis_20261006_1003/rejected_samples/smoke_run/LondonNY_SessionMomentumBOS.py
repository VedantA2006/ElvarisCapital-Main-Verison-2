{
  "spec": {
    "name": "LondonNY_SessionMomentumBOS",
    "timeframe": "1h",
    "hypothesis": "Gold's intraday liquidity is heavily concentrated during the London and New York sessions. Institutional order flow often establishes a directional bias early in the session, creating Break of Structure (BOS) events as stop-losses resting above/below recent swing points are triggered. Retail traders attempting to fade these initial momentum thrusts become trapped, providing fuel for continuation. By aligning ADX trend strength with confirmed market structure breaks (BOS) during active hours, we capture the institutional sweep while avoiding low-liquidity Asian chop.",
    "concept_family": "trend",
    "indicators_used": [
      "bos",
      "adx",
      "atr",
      "ema",
      "is_session"
    ],
    "entry_logic": "Enter long if bullish BOS is confirmed, ADX > threshold, +DI > -DI, and price is above the 21 EMA. Enter short if bearish BOS is confirmed, ADX > threshold, -DI > +DI, and price is below the 21 EMA.",
    "exit_logic": "Stop loss is placed at ATR * sl_mult distance. Take profit is placed at ATR * tp_mult distance to ensure a positive risk-reward ratio capturing the momentum extension.",
    "filters": [
      "Active only during London or NY sessions",
      "ADX must be above minimum threshold",
      "Price must be on the correct side of the 21 EMA"
    ],
    "session_filter": "london_ny",
    "direction": "both",
    "parameters": {
      "adx_period": {
        "default": 14,
        "min": 7,
        "max": 21
      },
      "adx_thresh": {
        "default": 20.0,
        "min": 15.0,
        "max": 35.0
      },
      "ema_period": {
        "default": 21,
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
      }
    },
    "expected_trades_per_year": 65,
    "expected_failure_conditions": "Protracted low-volatility ranging environments where ADX remains depressed and structural breaks are immediately reversed (fakeouts)."
  },
  "code": "import numpy as np\nimport pandas as pd\nfrom core.indicators import bos, adx, atr, ema, is_session\n\nclass Strategy:\n    PARAMS = {\n        \"adx_period\": {\"default\": 14, \"min\": 7, \"max\": 21},\n        \"adx_thresh\": {\"default\": 20.0, \"min\": 15.0, \"max\": 35.0},\n        \"ema_period\": {\"default\": 21, \"min\": 10, \"max\": 50},\n        \"atr_period\": {\"default\": 14, \"min\": 7, \"max\": 21},\n        \"sl_mult\": {\"default\": 1.5, \"min\": 1.0, \"max\": 3.0},\n        \"tp_mult\": {\"default\": 2.5, \"min\": 1.5, \"max\": 4.0}\n    }\n\n    def __init__(self, params):\n        self.params = params\n        self.last_trade_bar = -999\n\n    def on_bar(self, bars: pd.DataFrame):\n        if len(bars) < 60:\n            return None\n\n        if len(bars) - self.last_trade_bar < 4:\n            return None\n\n        sess_l = bool(is_session(bars, \"london\").iloc[-1])\n        sess_n = bool(is_session(bars, \"ny\").iloc[-1])\n        if not (sess_l or sess_n):\n            return None\n\n        bull_bos, bear_bos = bos(bars, swing_n=2)\n        adx_res = adx(bars, period=self.params[\"adx_period\"])\n        atr_s = atr(bars, period=self.params[\"atr_period\"])\n        ema_s = ema(bars, period=self.params[\"ema_period\"])\n\n        av = float(atr_s.iloc[-1])\n        if np.isnan(av) or av <= 0:\n            return None\n\n        ca = float(adx_res.adx_line.iloc[-1])\n        if np.isnan(ca):\n            return None\n\n        cp = float(adx_res.plus_di.iloc[-1])\n        cm = float(adx_res.minus_di.iloc[-1])\n        cb = bool(bull_bos.iloc[-1])\n        cbr = bool(bear_bos.iloc[-1])\n        cc = float(bars[\"close\"].iloc[-1])\n        ce = float(ema_s.iloc[-1])\n\n        sd = max(5.0, av * self.params[\"sl_mult\"])\n        td = max(7.5, av * self.params[\"tp_mult\"])\n        at = self.params[\"adx_thresh\"]\n\n        if cb and ca > at and cp > cm and cc > ce:\n            self.last_trade_bar = len(bars)\n            return Signal.enter_long(sl_distance=sd, tp_distance=td)\n\n        if cbr and ca > at and cm > cp and cc < ce:\n            self.last_trade_bar = len(bars)\n            return Signal.enter_short(sl_distance=sd, tp_distance=td)\n\n        return None"
}