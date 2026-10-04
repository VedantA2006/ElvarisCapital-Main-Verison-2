"""core/indicators.py: Comprehensive, tested, leak-safe indicator and market structure library.

Phase F5 / Closes: DATA-4.

Guarantees:
1. Leak-Safe: Every helper takes only past-and-current bars (0..t). No future peeking.
2. Truncation Proof: Value at bar t computed on df[:t+1] is bitwise identical (within 1e-9)
   to the value at index t from evaluating the full dataset.
3. Closed-bar HTF: htf() exposes higher-timeframe metrics ONLY after the HTF bar closes.
4. Confirmed Swings: swing_high/low(n) become known only n bars after the pivot occurred.
5. Mechanical Structure: Explicit numeric definitions for BOS, CHoCH, Order Blocks, FVG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# 1. Trend & Momentum Helpers
# ═══════════════════════════════════════════════════════════════════════════

def sma(series: pd.Series | np.ndarray, period: int = 20) -> pd.Series:
    """Simple Moving Average over rolling `period` bars.

    Parameters:
        series: Price series (e.g. close).
        period: Number of past bars to average.
    """
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    return s.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series | np.ndarray, period: int = 20) -> pd.Series:
    """Exponential Moving Average (Wilder/Standard exponential smoothing).

    Parameters:
        series: Price series.
        period: Smoothing period (alpha = 2 / (period + 1)).
    """
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    return s.ewm(span=period, adjust=False).mean()


def wma(series: pd.Series | np.ndarray, period: int = 20) -> pd.Series:
    """Linearly Weighted Moving Average with weights [1, 2, ..., period].

    Parameters:
        series: Price series.
        period: Number of past bars.
    """
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    weights = np.arange(1, period + 1)
    denom = weights.sum()

    def _weighted_avg(window: np.ndarray) -> float:
        return float(np.dot(window, weights) / denom)

    return s.rolling(window=period, min_periods=period).apply(_weighted_avg, raw=True)


def macd(
    series: pd.Series | np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence Divergence.

    Returns:
        (macd_line, signal_line, histogram)
    """
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    fast_ema = s.ewm(span=fast_period, adjust=False).mean()
    slow_ema = s.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def true_range(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
) -> pd.Series:
    """True Range: max(high - low, |high - prev_close|, |low - prev_close|)."""
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l = pd.Series(low) if not isinstance(low, pd.Series) else low
    c = pd.Series(close) if not isinstance(close, pd.Series) else close

    c_prev = c.shift(1)
    tr1 = h - l
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = h.iloc[0] - l.iloc[0]
    return tr


def atr(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    period: int = 14,
) -> pd.Series:
    """Average True Range with Wilder exponential smoothing."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def adx(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Movement Index (ADX, +DI, -DI).

    Returns:
        (adx, plus_di, minus_di)
    """
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l = pd.Series(low) if not isinstance(low, pd.Series) else low
    c = pd.Series(close) if not isinstance(close, pd.Series) else close

    h_diff = h.diff()
    l_diff = -l.diff()

    plus_dm = np.where((h_diff > l_diff) & (h_diff > 0), h_diff, 0.0)
    minus_dm = np.where((l_diff > h_diff) & (l_diff > 0), l_diff, 0.0)

    tr = true_range(h, l, c)
    atr_smooth = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    plus_di = 100.0 * (pd.Series(plus_dm).ewm(alpha=1.0 / period, adjust=False).mean() / atr_smooth)
    minus_di = 100.0 * (pd.Series(minus_dm).ewm(alpha=1.0 / period, adjust=False).mean() / atr_smooth)

    dx_denom = plus_di + minus_di
    dx = np.where(dx_denom > 0, 100.0 * ((plus_di - minus_di).abs() / dx_denom), 0.0)
    adx_line = pd.Series(dx).ewm(alpha=1.0 / period, adjust=False).mean()

    return adx_line, plus_di, minus_di


def supertrend(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """SuperTrend indicator.

    Returns:
        (supertrend_line, direction) where direction is 1 (bullish) or -1 (bearish).
    """
    h = pd.Series(high).to_numpy(dtype=float)
    l = pd.Series(low).to_numpy(dtype=float)
    c = pd.Series(close).to_numpy(dtype=float)
    n = len(c)

    atr_vals = atr(high, low, close, period=period).to_numpy(dtype=float)
    basic_upper = (h + l) / 2.0 + multiplier * atr_vals
    basic_lower = (h + l) / 2.0 - multiplier * atr_vals

    final_upper = np.zeros(n, dtype=float)
    final_lower = np.zeros(n, dtype=float)
    st = np.zeros(n, dtype=float)
    direction = np.ones(n, dtype=int)

    for i in range(n):
        if i == 0:
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
            st[i] = final_upper[i]
            direction[i] = -1
            continue

        # Upper band
        if basic_upper[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Lower band
        if basic_lower[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction switch
        prev_dir = direction[i - 1]
        if prev_dir == 1:
            if c[i] < final_lower[i]:
                direction[i] = -1
                st[i] = final_upper[i]
            else:
                direction[i] = 1
                st[i] = final_lower[i]
        else:
            if c[i] > final_upper[i]:
                direction[i] = 1
                st[i] = final_lower[i]
            else:
                direction[i] = -1
                st[i] = final_upper[i]

    return pd.Series(st), pd.Series(direction)


def donchian(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    period: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channel over rolling `period` past bars.

    Returns:
        (upper, middle, lower)
    """
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l = pd.Series(low) if not isinstance(low, pd.Series) else low

    upper = h.rolling(window=period, min_periods=period).max()
    lower = l.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2.0
    return upper, middle, lower


def keltner(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels (EMA midline +/- multiplier * ATR).

    Returns:
        (upper, middle, lower)
    """
    c = pd.Series(close) if not isinstance(close, pd.Series) else close
    middle = c.ewm(span=ema_period, adjust=False).mean()
    atr_vals = atr(high, low, close, period=atr_period)
    upper = middle + multiplier * atr_vals
    lower = middle - multiplier * atr_vals
    return upper, middle, lower


def roc(series: pd.Series | np.ndarray, period: int = 10) -> pd.Series:
    """Rate of Change percentage over `period` bars: ((P_t - P_{t-N}) / P_{t-N}) * 100."""
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    prev = s.shift(period)
    return (s - prev) / prev * 100.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. Mean Reversion & Volatility Helpers
# ═══════════════════════════════════════════════════════════════════════════

def rsi(series: pd.Series | np.ndarray, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's RSI).

    Returns:
        Series in range [0, 100].
    """
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / np.where(avg_loss == 0, 1e-9, avg_loss)
    res = 100.0 - (100.0 / (1.0 + rs))
    return res


def stoch(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
    slowing: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator (%K, %D)."""
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l = pd.Series(low) if not isinstance(low, pd.Series) else low
    c = pd.Series(close) if not isinstance(close, pd.Series) else close

    lowest_low = l.rolling(window=k_period, min_periods=k_period).min()
    highest_high = h.rolling(window=k_period, min_periods=k_period).max()

    denom = np.where((highest_high - lowest_low) == 0, 1e-9, highest_high - lowest_low)
    fast_k = 100.0 * (c - lowest_low) / denom

    slow_k = fast_k.rolling(window=slowing, min_periods=slowing).mean()
    slow_d = slow_k.rolling(window=d_period, min_periods=d_period).mean()
    return slow_k, slow_d


def bollinger(
    series: pd.Series | np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Returns:
        (upper, middle, lower, bandwidth, pct_b)
    """
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    middle = s.rolling(window=period, min_periods=period).mean()
    std = s.rolling(window=period, min_periods=period).std()

    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / np.where(middle == 0, 1e-9, middle)
    pct_b = (s - lower) / np.where((upper - lower) == 0, 1e-9, upper - lower)
    return upper, middle, lower, bandwidth, pct_b


def zscore(series: pd.Series | np.ndarray, period: int = 20) -> pd.Series:
    """Rolling Z-Score: (series - rolling_mean) / rolling_std."""
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    m = s.rolling(window=period, min_periods=period).mean()
    std = s.rolling(window=period, min_periods=period).std()
    return (s - m) / np.where(std == 0, 1e-9, std)


def realized_vol(
    series: pd.Series | np.ndarray,
    period: int = 20,
    annualize: bool = True,
    bars_per_year: int = 252 * 24,
) -> pd.Series:
    """Rolling realized volatility of log returns."""
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    log_ret = np.log(s / s.shift(1))
    vol = log_ret.rolling(window=period, min_periods=period).std()
    if annualize:
        vol = vol * np.sqrt(bars_per_year)
    return vol


def vol_percentile(
    series: pd.Series | np.ndarray,
    lookback: int = 100,
) -> pd.Series:
    """Rolling percentile rank (0..100) of current value within past lookback bars."""
    s = pd.Series(series) if not isinstance(series, pd.Series) else series

    def _pct_rank(window: np.ndarray) -> float:
        val = window[-1]
        return float(np.mean(window <= val) * 100.0)

    return s.rolling(window=lookback, min_periods=min(20, lookback)).apply(_pct_rank, raw=True)


def squeeze(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
) -> pd.Series:
    """Bollinger Band / Keltner Channel Volatility Squeeze.

    Returns:
        Boolean Series: True if Bollinger Band is strictly inside Keltner Channel.
    """
    bb_upper, _, bb_lower, _, _ = bollinger(close, period=bb_period, num_std=bb_std)
    kc_upper, _, kc_lower = keltner(high, low, close, ema_period=kc_period, atr_period=kc_period, multiplier=kc_mult)
    return (bb_upper < kc_upper) & (bb_lower > kc_lower)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Volume & Price Location Helpers
# ═══════════════════════════════════════════════════════════════════════════

def session_vwap(df: pd.DataFrame, volume_col: str = "volume") -> pd.Series:
    """Cumulative Volume-Weighted Average Price resetting at each new session.

    If session column is absent, resets at 00:00 UTC daily.
    """
    v = df[volume_col] if volume_col in df.columns else pd.Series(1.0, index=df.index)
    p = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = p * v

    # Group by session changes or day changes
    if "session" in df.columns:
        group_id = (df["session"] != df["session"].shift(1)).cumsum()
    else:
        dates = pd.to_datetime(df["timestamp"]).dt.date if "timestamp" in df.columns else pd.Series(0, index=df.index)
        group_id = (dates != dates.shift(1)).cumsum()

    cum_pv = pv.groupby(group_id).cumsum()
    cum_v = v.groupby(group_id).cumsum()
    return cum_pv / np.where(cum_v == 0, 1e-9, cum_v)


def anchored_vwap(
    df: pd.DataFrame,
    anchor_mask: pd.Series | np.ndarray,
    volume_col: str = "volume",
) -> pd.Series:
    """VWAP anchored to instances where anchor_mask is True."""
    v = df[volume_col] if volume_col in df.columns else pd.Series(1.0, index=df.index)
    p = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = p * v

    m = pd.Series(anchor_mask, index=df.index).astype(bool)
    group_id = m.cumsum()

    cum_pv = pv.groupby(group_id).cumsum()
    cum_v = v.groupby(group_id).cumsum()
    return cum_pv / np.where(cum_v == 0, 1e-9, cum_v)


def previous_day_hl(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Previous completed day's High and Low.

    Exposes the value of Day D-1 on all bars of Day D (leak-safe).
    """
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.date_range("2022-01-01", periods=len(df), freq="h")
    dates = ts.dt.date

    # Daily aggregation
    daily_high = df["high"].groupby(dates).max()
    daily_low = df["low"].groupby(dates).min()

    # Shift by 1 day so Day D sees Day D-1
    prev_high = daily_high.shift(1)
    prev_low = daily_low.shift(1)

    # Map back to bar indices
    bar_prev_high = dates.map(prev_high)
    bar_prev_low = dates.map(prev_low)

    return pd.Series(bar_prev_high, index=df.index), pd.Series(bar_prev_low, index=df.index)


def previous_week_hl(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Previous completed week's High and Low.

    Exposes the value of Week W-1 on all bars of Week W (leak-safe).
    """
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.date_range("2022-01-01", periods=len(df), freq="h")
    # Year-week tuple
    year_weeks = ts.apply(lambda t: (t.year, t.isocalendar()[1]))

    weekly_high = df["high"].groupby(year_weeks).max()
    weekly_low = df["low"].groupby(year_weeks).min()

    prev_high = weekly_high.shift(1)
    prev_low = weekly_low.shift(1)

    bar_prev_high = year_weeks.map(prev_high)
    bar_prev_low = year_weeks.map(prev_low)

    return pd.Series(bar_prev_high, index=df.index), pd.Series(bar_prev_low, index=df.index)


def session_range(df: pd.DataFrame, session_name: str) -> tuple[pd.Series, pd.Series]:
    """High and Low of the most recently COMPLETED session named `session_name`."""
    sess = df["session"].astype(str).str.lower() if "session" in df.columns else pd.Series([""] * len(df))
    target = session_name.lower()

    n = len(df)
    res_high = np.full(n, np.nan, dtype=float)
    res_low = np.full(n, np.nan, dtype=float)

    last_high = np.nan
    last_low = np.nan
    cur_high = np.nan
    cur_low = np.nan
    in_target = False

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    for i in range(n):
        s_i = sess.iloc[i]
        if s_i == target:
            if not in_target:
                # Entering target session: reset current tracker
                in_target = True
                cur_high = highs[i]
                cur_low = lows[i]
            else:
                cur_high = max(cur_high, highs[i])
                cur_low = min(cur_low, lows[i])
        else:
            if in_target:
                # Exiting target session: target session is now COMPLETED!
                in_target = False
                last_high = cur_high
                last_low = cur_low

        # Expose only the most recently completed session's range
        res_high[i] = last_high
        res_low[i] = last_low

    return pd.Series(res_high, index=df.index), pd.Series(res_low, index=df.index)


def opening_range(df: pd.DataFrame, n_bars: int = 1) -> tuple[pd.Series, pd.Series]:
    """Opening range High and Low over the first `n_bars` of the session/day.

    Exposed strictly from bar n_bars + 1 onward.
    """
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.date_range("2022-01-01", periods=len(df), freq="h")
    dates = ts.dt.date

    n = len(df)
    or_high = np.full(n, np.nan, dtype=float)
    or_low = np.full(n, np.nan, dtype=float)

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    day_bars = 0
    cur_or_h = -np.inf
    cur_or_l = np.inf
    frozen_h = np.nan
    frozen_l = np.nan

    for i in range(n):
        if i == 0 or dates.iloc[i] != dates.iloc[i - 1]:
            day_bars = 0
            cur_or_h = -np.inf
            cur_or_l = np.inf
            frozen_h = np.nan
            frozen_l = np.nan

        day_bars += 1
        if day_bars <= n_bars:
            cur_or_h = max(cur_or_h, highs[i])
            cur_or_l = min(cur_or_l, lows[i])
            if day_bars == n_bars:
                frozen_h = cur_or_h
                frozen_l = cur_or_l

        or_high[i] = frozen_h
        or_low[i] = frozen_l

    return pd.Series(or_high, index=df.index), pd.Series(or_low, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Time Feature Helpers
# ═══════════════════════════════════════════════════════════════════════════

def hour_utc(df: pd.DataFrame) -> pd.Series:
    """Integer hour (0..23) in UTC."""
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.Series(pd.Timestamp.utcnow(), index=df.index)
    return ts.dt.hour


def dow(df: pd.DataFrame) -> pd.Series:
    """Day of week (0=Monday, 6=Sunday)."""
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.Series(pd.Timestamp.utcnow(), index=df.index)
    return ts.dt.dayofweek


def is_session(df: pd.DataFrame, session_name: str) -> pd.Series:
    """Check if current bar falls within specified session name."""
    if "session" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["session"].astype(str).str.lower() == session_name.lower()


def minutes_since_session_open(df: pd.DataFrame) -> pd.Series:
    """Minutes elapsed since the start of the current session."""
    if "session" not in df.columns or "timestamp" not in df.columns:
        return pd.Series(0, index=df.index)

    sess = df["session"].astype(str)
    ts = pd.to_datetime(df["timestamp"])
    is_new = (sess != sess.shift(1))

    sess_start_time = ts.where(is_new).ffill()
    elapsed = (ts - sess_start_time).dt.total_seconds() / 60.0
    return elapsed


def month_end_flag(df: pd.DataFrame, days_before: int = 2) -> pd.Series:
    """Flag True if date is within `days_before` business days of month end."""
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.date_range("2022-01-01", periods=len(df), freq="h")
    # Distance to next month
    next_month = ts.dt.to_period("M").dt.to_timestamp(how="end")
    days_left = (next_month.dt.day - ts.dt.day)
    return (days_left <= days_before) & (days_left >= 0)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Higher-Timeframe (HTF) Closed-Candle Builder
# ═══════════════════════════════════════════════════════════════════════════

def htf(
    df: pd.DataFrame,
    target_tf: str = "4h",
    func: Callable[[pd.DataFrame], pd.Series | np.ndarray] = lambda d: d["close"],
) -> pd.Series:
    """Build higher-timeframe features strictly from CLOSED HTF candles.

    Guarantees:
    - An HTF bar is exposed only AFTER its last constituent lower-TF bar has closed.
    - Zero repainting / zero future peeking.
    """
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.date_range("2022-01-01", periods=len(df), freq="h")

    # Determine frequency mapping
    freq = "4h" if target_tf.lower() in ("4h", "240m") else "1D"

    # Floor timestamps to HTF grid
    htf_periods = ts.dt.floor(freq)

    # Detect the last bar of each completed HTF candle
    is_last_in_htf = (htf_periods != htf_periods.shift(-1))

    # Aggregate HTF OHLCV over closed blocks
    htf_df = df.groupby(htf_periods).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum" if "volume" in df.columns else "count",
    })

    # Apply user function on closed HTF dataframe
    htf_values = func(htf_df)
    if isinstance(htf_values, (pd.Series, pd.DataFrame)):
        htf_values = htf_values.to_numpy()

    # Shift by 1 HTF bar so HTF candle K is exposed ONLY starting from the bar AFTER it closes!
    htf_val_series = pd.Series(htf_values, index=htf_df.index).shift(1)

    # Reindex back to lower-TF timestamps using forward fill
    bar_htf_values = htf_periods.map(htf_val_series)
    return pd.Series(bar_htf_values, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Confirmed Swings & Pivots
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SwingResult:
    pivot_value: pd.Series
    pivot_bar: pd.Series
    confirmed_at: pd.Series


def swing_high(high: pd.Series | np.ndarray, n: int = 2) -> SwingResult:
    """Confirmed Swing High.

    A swing high at bar t is defined as: high[t] > high[t-k] and high[t] > high[t+k] for k in 1..n.
    CRITICAL: It is confirmed and exposed ONLY at bar t + n!
    """
    h = pd.Series(high).to_numpy(dtype=float)
    size = len(h)

    pivot_val = np.full(size, np.nan, dtype=float)
    pivot_idx = np.full(size, -1, dtype=int)
    confirmed = np.full(size, -1, dtype=int)

    last_p_val = np.nan
    last_p_idx = -1
    last_conf = -1

    for i in range(size):
        # Check if bar i - n was a swing high
        cand = i - n
        if cand >= n:
            is_pivot = True
            cand_h = h[cand]
            for offset in range(1, n + 1):
                if h[cand - offset] >= cand_h or h[cand + offset] >= cand_h:
                    is_pivot = False
                    break
            if is_pivot:
                last_p_val = cand_h
                last_p_idx = cand
                last_conf = i

        pivot_val[i] = last_p_val
        pivot_idx[i] = last_p_idx
        confirmed[i] = last_conf

    return SwingResult(
        pivot_value=pd.Series(pivot_val),
        pivot_bar=pd.Series(pivot_idx),
        confirmed_at=pd.Series(confirmed),
    )


def swing_low(low: pd.Series | np.ndarray, n: int = 2) -> SwingResult:
    """Confirmed Swing Low.

    A swing low at bar t is defined as: low[t] < low[t-k] and low[t] < low[t+k] for k in 1..n.
    CRITICAL: Confirmed and exposed ONLY at bar t + n!
    """
    l = pd.Series(low).to_numpy(dtype=float)
    size = len(l)

    pivot_val = np.full(size, np.nan, dtype=float)
    pivot_idx = np.full(size, -1, dtype=int)
    confirmed = np.full(size, -1, dtype=int)

    last_p_val = np.nan
    last_p_idx = -1
    last_conf = -1

    for i in range(size):
        cand = i - n
        if cand >= n:
            is_pivot = True
            cand_l = l[cand]
            for offset in range(1, n + 1):
                if l[cand - offset] <= cand_l or l[cand + offset] <= cand_l:
                    is_pivot = False
                    break
            if is_pivot:
                last_p_val = cand_l
                last_p_idx = cand
                last_conf = i

        pivot_val[i] = last_p_val
        pivot_idx[i] = last_p_idx
        confirmed[i] = last_conf

    return SwingResult(
        pivot_value=pd.Series(pivot_val),
        pivot_bar=pd.Series(pivot_idx),
        confirmed_at=pd.Series(confirmed),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7. Mechanical Market Structure Helpers
# ═══════════════════════════════════════════════════════════════════════════

def bos(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    swing_n: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Break of Structure (BOS).

    Bullish BOS: Bar close > last confirmed swing high.
    Bearish BOS: Bar close < last confirmed swing low.
    Returns:
        (bullish_bos, bearish_bos) as boolean Series.
    """
    c = pd.Series(close) if not isinstance(close, pd.Series) else close
    sh = swing_high(high, n=swing_n)
    sl = swing_low(low, n=swing_n)

    bull_bos = (c > sh.pivot_value) & (c.shift(1) <= sh.pivot_value.shift(1))
    bear_bos = (c < sl.pivot_value) & (c.shift(1) >= sl.pivot_value.shift(1))
    return bull_bos.fillna(False), bear_bos.fillna(False)


def choch(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    swing_n: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Change of Character (CHoCH) - market structure reversal.

    Bullish CHoCH: After bearish trend, close breaks above prior confirmed swing high.
    Bearish CHoCH: After bullish trend, close breaks below prior confirmed swing low.
    """
    bull_bos, bear_bos = bos(high, low, close, swing_n=swing_n)
    c = pd.Series(close) if not isinstance(close, pd.Series) else close
    n = len(c)

    trend = np.zeros(n, dtype=int)
    bull_choch = np.zeros(n, dtype=bool)
    bear_choch = np.zeros(n, dtype=bool)

    cur_trend = 0
    for i in range(n):
        if bull_bos.iloc[i]:
            if cur_trend == -1:
                bull_choch[i] = True
            cur_trend = 1
        elif bear_bos.iloc[i]:
            if cur_trend == 1:
                bear_choch[i] = True
            cur_trend = -1
        trend[i] = cur_trend

    return pd.Series(bull_choch, index=c.index), pd.Series(bear_choch, index=c.index)


def fvg(high: pd.Series | np.ndarray, low: pd.Series | np.ndarray) -> tuple[pd.Series, pd.Series]:
    """Fair Value Gap (3-bar imbalance).

    Bullish FVG: low[t] > high[t-2] (gap between bar t-2 high and bar t low).
    Bearish FVG: high[t] < low[t-2] (gap between bar t-2 low and bar t high).
    """
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l = pd.Series(low) if not isinstance(low, pd.Series) else low

    bull_fvg = (l > h.shift(2)).fillna(False)
    bear_fvg = (h < l.shift(2)).fillna(False)
    return bull_fvg, bear_fvg


def liquidity_sweep(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    swing_n: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Liquidity Sweep / Stop Hunt.

    High sweep: high[t] > confirmed swing high, BUT close[t] <= confirmed swing high.
    Low sweep: low[t] < confirmed swing low, BUT close[t] >= confirmed swing low.
    """
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l = pd.Series(low) if not isinstance(low, pd.Series) else low
    c = pd.Series(close) if not isinstance(close, pd.Series) else close

    sh = swing_high(high, n=swing_n)
    sl = swing_low(low, n=swing_n)

    high_sweep = (h > sh.pivot_value) & (c <= sh.pivot_value)
    low_sweep = (l < sl.pivot_value) & (c >= sl.pivot_value)

    return high_sweep.fillna(False), low_sweep.fillna(False)


def premium_discount(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    swing_n: int = 5,
) -> pd.Series:
    """Relative position within current swing range (0.0 = discount low, 1.0 = premium high).

    Equilibrium = 0.50.
    """
    c = pd.Series(close) if not isinstance(close, pd.Series) else close
    sh = swing_high(high, n=swing_n)
    sl = swing_low(low, n=swing_n)

    range_span = sh.pivot_value - sl.pivot_value
    pos = (c - sl.pivot_value) / np.where(range_span <= 0, 1e-9, range_span)
    return pos.clip(lower=0.0, upper=1.0)


def order_block(
    df: pd.DataFrame,
    swing_n: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Order Block detection.

    Bullish OB: Last down-candle prior to a confirmed bullish BOS.
    Bearish OB: Last up-candle prior to a confirmed bearish BOS.
    Returns:
        (bull_ob_price, bear_ob_price)
    """
    bull_bos, bear_bos = bos(df["high"], df["low"], df["close"], swing_n=swing_n)
    n = len(df)

    c = df["close"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)

    bull_ob = np.full(n, np.nan, dtype=float)
    bear_ob = np.full(n, np.nan, dtype=float)

    last_bull_ob = np.nan
    last_bear_ob = np.nan

    for i in range(1, n):
        if bull_bos.iloc[i]:
            # Look back for last down candle (close < open)
            for j in range(i - 1, max(-1, i - 15), -1):
                if c[j] < o[j]:
                    last_bull_ob = o[j]
                    break
        elif bear_bos.iloc[i]:
            # Look back for last up candle (close > open)
            for j in range(i - 1, max(-1, i - 15), -1):
                if c[j] > o[j]:
                    last_bear_ob = o[j]
                    break

        bull_ob[i] = last_bull_ob
        bear_ob[i] = last_bear_ob

    return pd.Series(bull_ob, index=df.index), pd.Series(bear_ob, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Market Regimes
# ═══════════════════════════════════════════════════════════════════════════

def trend_range_regime(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    adx_period: int = 14,
    er_period: int = 10,
) -> pd.Series:
    """Classify market regime as 'trending', 'ranging', or 'neutral'.

    Combines ADX (> 25 = trend) and Kaufman Efficiency Ratio (|change| / path_length).
    """
    c = pd.Series(close) if not isinstance(close, pd.Series) else close
    adx_vals, _, _ = adx(high, low, close, period=adx_period)

    # Kaufman Efficiency Ratio
    direction = (c - c.shift(er_period)).abs()
    path = (c.diff().abs()).rolling(window=er_period, min_periods=er_period).sum()
    er = direction / np.where(path == 0, 1e-9, path)

    n = len(c)
    regimes = []
    for i in range(n):
        a = adx_vals.iloc[i]
        e = er.iloc[i]
        if np.isnan(a) or np.isnan(e):
            regimes.append("neutral")
        elif a > 25.0 and e > 0.40:
            regimes.append("trending")
        elif a < 20.0 and e < 0.30:
            regimes.append("ranging")
        else:
            regimes.append("neutral")

    return pd.Series(regimes, index=c.index)


def vol_regime(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    atr_period: int = 14,
    lookback: int = 100,
) -> pd.Series:
    """Classify volatility regime as 'low_vol', 'normal_vol', or 'high_vol'."""
    atr_vals = atr(high, low, close, period=atr_period)
    pcts = vol_percentile(atr_vals, lookback=lookback)

    def _classify(p: float) -> str:
        if np.isnan(p):
            return "normal_vol"
        if p <= 25.0:
            return "low_vol"
        elif p >= 75.0:
            return "high_vol"
        return "normal_vol"

    return pcts.apply(_classify)


def session_regime(df: pd.DataFrame) -> pd.Series:
    """Returns the active trading session name or 'off_hours'."""
    if "session" in df.columns:
        return df["session"].astype(str)
    return pd.Series("unknown", index=df.index)
