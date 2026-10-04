# QuantForge Indicator & Market Structure Library (`core/indicators.py`)

This library provides tested, vectorised, and strictly leak-safe indicators and market structure helpers for strategy research and LLM generation.

---

## Guarantees & Safety Architecture

1. **Past-Only Evaluation**: Every helper computes values using strictly past-and-current bars ($0..t$).
2. **Truncation Property Proven**: Every function is mathematically verified by property tests:
   $$\text{indicator}(df[0..t+1])[-1] == \text{indicator}(df[0..N])[t]$$
3. **Closed-Bar HTF**: `htf()` exposes higher-timeframe metrics only *after* the HTF candle has completely closed, eliminating repainting.
4. **Confirmed Swings**: `swing_high()` and `swing_low()` are confirmed and exposed strictly $N$ bars after the pivot occurred.
5. **Deterministic Market Structure**: Explicit mechanical definitions for BOS, CHoCH, Order Blocks, and Fair Value Gaps.

---

## 1. Trend & Momentum

### `sma(series, period=20) -> pd.Series`
- **Description**: Simple Moving Average over rolling `period` bars.
- **Parameters**: `series` (price series, e.g. `bars['close']`), `period` (int).
- **Leak Safety**: Uses rolling window of length `period` ending at bar $t$.

### `ema(series, period=20) -> pd.Series`
- **Description**: Exponential Moving Average with smoothing multiplier $\alpha = \frac{2}{\text{period} + 1}$.
- **Parameters**: `series`, `period` (int).
- **Leak Safety**: Evaluated with `adjust=False` so weights decay strictly backwards into the past.

### `wma(series, period=20) -> pd.Series`
- **Description**: Linearly Weighted Moving Average with weights $[1, 2, \dots, \text{period}]$.
- **Parameters**: `series`, `period` (int).

### `macd(series, fast_period=12, slow_period=26, signal_period=9) -> tuple[pd.Series, pd.Series, pd.Series]`
- **Description**: Moving Average Convergence Divergence.
- **Returns**: `(macd_line, signal_line, histogram)`.

### `true_range(high, low, close) -> pd.Series`
- **Description**: True Range $\max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$.
- **Leak Safety**: Previous close is shifted by 1 bar.

### `atr(high, low, close, period=14) -> pd.Series`
- **Description**: Average True Range using Wilder's exponential smoothing ($\alpha = 1/\text{period}$).

### `adx(high, low, close, period=14) -> tuple[pd.Series, pd.Series, pd.Series]`
- **Description**: Average Directional Movement Index.
- **Returns**: `(adx, plus_di, minus_di)`.

### `supertrend(high, low, close, period=10, multiplier=3.0) -> tuple[pd.Series, pd.Series]`
- **Description**: SuperTrend volatility band and trend direction.
- **Returns**: `(supertrend_line, direction)` where direction is $+1$ (bullish) or $-1$ (bearish).

### `donchian(high, low, period=20) -> tuple[pd.Series, pd.Series, pd.Series]`
- **Description**: Donchian Channel over rolling `period` bars.
- **Returns**: `(upper_band, middle_band, lower_band)`.

### `keltner(high, low, close, ema_period=20, atr_period=10, multiplier=2.0) -> tuple[pd.Series, pd.Series, pd.Series]`
- **Description**: Keltner Channels (EMA midline $\pm$ multiplier $\times$ ATR).
- **Returns**: `(upper, middle, lower)`.

### `roc(series, period=10) -> pd.Series`
- **Description**: Rate of Change percentage: $\frac{P_t - P_{t-N}}{P_{t-N}} \times 100$.

---

## 2. Mean Reversion & Volatility

### `rsi(series, period=14) -> pd.Series`
- **Description**: Wilder's Relative Strength Index in range $[0, 100]$.

### `stoch(high, low, close, k_period=14, d_period=3, slowing=3) -> tuple[pd.Series, pd.Series]`
- **Description**: Slow Stochastic Oscillator (%K and %D).

### `bollinger(series, period=20, num_std=2.0) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]`
- **Description**: Bollinger Bands and metrics.
- **Returns**: `(upper, middle, lower, bandwidth, pct_b)`.

### `zscore(series, period=20) -> pd.Series`
- **Description**: Rolling Z-Score: $\frac{\text{series} - \mu_{20}}{\sigma_{20}}$.

### `realized_vol(series, period=20, annualize=True) -> pd.Series`
- **Description**: Realized volatility of log returns, annualized by $\sqrt{252 \times 24}$ for 1h bars.

### `vol_percentile(series, lookback=100) -> pd.Series`
- **Description**: Percentile rank (0..100) of current value within rolling `lookback` history.

### `squeeze(high, low, close, bb_period=20, bb_std=2.0, kc_period=20, kc_mult=1.5) -> pd.Series`
- **Description**: Volatility Squeeze indicator (True when Bollinger Bands are completely inside Keltner Channels).

---

## 3. Volume & Price Location

### `session_vwap(df, volume_col="volume") -> pd.Series`
- **Description**: Cumulative Volume-Weighted Average Price resetting at each session boundary.

### `anchored_vwap(df, anchor_mask, volume_col="volume") -> pd.Series`
- **Description**: VWAP anchored to bars where `anchor_mask` is True.

### `previous_day_hl(df) -> tuple[pd.Series, pd.Series]`
- **Description**: Previous completed day's High and Low.
- **Leak Safety**: Shifted by 1 calendar day; exposed on Day $D$ from Day $D-1$.

### `previous_week_hl(df) -> tuple[pd.Series, pd.Series]`
- **Description**: Previous completed week's High and Low.
- **Leak Safety**: Shifted by 1 calendar week; exposed on Week $W$ from Week $W-1$.

### `session_range(df, session_name) -> tuple[pd.Series, pd.Series]`
- **Description**: High and Low of the most recently COMPLETED session (e.g. `'asia'`, `'london'`).

### `opening_range(df, n_bars=1) -> tuple[pd.Series, pd.Series]`
- **Description**: Opening range of the day over the first $N$ bars; available strictly from bar $N+1$.

---

## 4. Time Features

### `hour_utc(df) -> pd.Series`
- **Description**: Integer hour (0..23) in UTC.

### `dow(df) -> pd.Series`
- **Description**: Day of week (0=Monday, 6=Sunday).

### `is_session(df, session_name) -> pd.Series`
- **Description**: Boolean mask for specified session (`'asia'`, `'london'`, `'new_york'`).

### `minutes_since_session_open(df) -> pd.Series`
- **Description**: Elapsed minutes since the start of the current trading session.

### `month_end_flag(df, days_before=2) -> pd.Series`
- **Description**: Boolean flag indicating if current date is within `days_before` business days of month-end.

---

## 5. Higher Timeframe (HTF) Closed-Candle Builder

### `htf(df, target_tf="4h", func=lambda d: d["close"]) -> pd.Series`
- **Description**: Aggregates lower-TF bars into higher-TF bars and applies `func`.
- **Leak Safety**: The result for HTF candle $K$ is shifted by 1 HTF bar, so it is exposed to lower-TF bars **strictly after the HTF candle has closed**.

---

## 6. Confirmed Swings & Pivots

### `swing_high(high, n=2) -> SwingResult`
- **Description**: Confirmed swing high pivot requiring $n$ lower bars before and after.
- **Returns**: `SwingResult(pivot_value, pivot_bar, confirmed_at)`.
- **Leak Safety**: A swing at bar $t$ is confirmed and exposed strictly at bar $t + n$.

### `swing_low(low, n=2) -> SwingResult`
- **Description**: Confirmed swing low pivot requiring $n$ higher bars before and after.
- **Leak Safety**: Confirmed and exposed strictly at bar $t + n$.

---

## 7. Mechanical Market Structure

### `bos(high, low, close, swing_n=2) -> tuple[pd.Series, pd.Series]`
- **Description**: Break of Structure.
  - Bullish BOS: Bar close > last confirmed swing high.
  - Bearish BOS: Bar close < last confirmed swing low.

### `choch(high, low, close, swing_n=2) -> tuple[pd.Series, pd.Series]`
- **Description**: Change of Character (trend reversal breaking opposite confirmed swing).

### `fvg(high, low) -> tuple[pd.Series, pd.Series]`
- **Description**: 3-bar Fair Value Gap imbalance.
  - Bullish FVG: `low[t] > high[t-2]`.
  - Bearish FVG: `high[t] < low[t-2]`.

### `liquidity_sweep(high, low, close, swing_n=2) -> tuple[pd.Series, pd.Series]`
- **Description**: Wick pierces confirmed swing level but bar closes back inside.

### `premium_discount(high, low, close, swing_n=5) -> pd.Series`
- **Description**: Normalized position in current swing range (0.0 = discount low, 1.0 = premium high, 0.5 = equilibrium).

### `order_block(df, swing_n=2) -> tuple[pd.Series, pd.Series]`
- **Description**: Order Block level (last opposite-color candle before a confirmed impulse BOS).

---

## 8. Market Regimes

### `trend_range_regime(high, low, close, adx_period=14, er_period=10) -> pd.Series`
- **Description**: Classifies bars into `'trending'`, `'ranging'`, or `'neutral'`.

### `vol_regime(high, low, close, atr_period=14, lookback=100) -> pd.Series`
- **Description**: Classifies bars into `'low_vol'`, `'normal_vol'`, or `'high_vol'`.

### `session_regime(df) -> pd.Series`
- **Description**: Returns active session name or `'off_hours'`.
