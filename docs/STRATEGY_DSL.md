# ELVARIS CAPITAL — STRATEGY DOMAIN-SPECIFIC LANGUAGE (DSL) SPECIFICATION (V3)

---

## 1. Motivation & Design Goals

The Strategy DSL (`strategy/dsl/`) provides a structured, declarative representation for XAUUSD systematic strategies. It serves as the formal intermediary between LLM creative hypothesis formulation and deterministic backtest execution.

### Core Goals:
1. **Zero Natural Language Ambiguity**: Every condition is an exact mathematical expression.
2. **Prevent Lookahead By Construction**: Offset syntax strictly enforces non-negative indices ($t, t-1, \dots$).
3. **Multi-Timeframe Integration**: Native support for context indicators aligned strictly at closed bar boundaries.
4. **Complexity Caps**: Maximum 6 tunable parameters and bounded condition trees prevent curve-fitting.

---

## 2. DSL YAML / JSON Schema (`strategy/dsl/schema.py`)

A strategy specification consists of:
```yaml
name: Gold_LondonSweep_Displacement_15m
market: XAUUSD
family: LIQUIDITY
hypothesis: Exploits London session liquidity sweeps of Asian session highs followed by displacement back into fair value.

timeframes:
  primary: 15m
  context: 1h

indicators:
  - id: ema_macro
    type: ema
    timeframe: 1h
    params: { period: 50 }

  - id: atr_vol
    type: atr
    params: { period: 14 }

  - id: london_sweep
    type: liquidity_sweep
    params: { lookback: 20 }

entry_rules:
  - direction: LONG
    session_filter: london_ny
    cooldown_bars: 3
    all_conditions:
      - left: close
        operator: ">"
        right: ema_macro
        offset: 0
      - left: london_sweep
        operator: "=="
        right: 1
        offset: 0

exit:
  stop_loss_type: atr
  stop_loss_multiplier: 1.8
  take_profit_type: rr
  take_profit_ratio: 2.4

risk:
  risk_per_trade: 0.01
  max_spread_usd: 0.60

parameters:
  sl_mult:
    default: 1.8
    min: 1.0
    max: 3.5
  tp_ratio:
    default: 2.4
    min: 1.5
    max: 4.5
```

---

## 3. Approved Indicator & Price Action Primitives

| Family | Supported Primitives |
|:---|:---|
| **Trend** | `ema`, `sma`, `wma`, `supertrend` |
| **Momentum** | `rsi`, `roc`, `macd`, `stoch`, `zscore` |
| **Volatility & Bands** | `atr`, `bollinger`, `keltner`, `donchian`, `realized_vol`, `vol_percentile`, `squeeze` |
| **Market Structure (SMC)** | `swing_high`, `swing_low`, `bos` (Break of Structure), `choch` (Change of Character), `fvg` (Fair Value Gap) |
| **Liquidity** | `liquidity_sweep`, `previous_day_hl`, `previous_week_hl`, `opening_range` |
| **Sessions** | `is_session` (`london`, `ny`, `asia`, `london_ny`), `session_vwap`, `minutes_since_session_open` |
| **Multi-Timeframe** | `htf(target_tf, func)` strictly aligned to closed higher-timeframe candles |

---

## 4. Compilation Pipeline (`strategy/dsl/compiler.py`)

The `DSLCompiler` translates the declarative schema into a hardened Python Strategy class:
1. **Indicator Vectorization**: Precomputes indicator vectors over `bars` using leak-free primitives from `core.indicators`.
2. **Session Conditioning**: Masks signals outside the designated session window.
3. **Execution Sizing**: Computes exact stop loss ($SL$) and take profit ($TP$) distances in USD per oz.
4. **Coordinated Signal Emission**: Emits `Signal.enter_long(...)` or `Signal.enter_short(...)` obeying cooldown bars.
5. **AST Safety Guarantee**: The output contains zero dynamic `getattr`, `eval`, or file access, ensuring $100\%$ pass rates through the Gate 2 Security Scanner.
