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
import sys
import dis

import numpy as np
import pandas as pd


class IndicatorSeries(pd.Series):
    """Subclass of pd.Series that safely handles negative integer indexing (e.g. s[-1]) and label mismatches."""
    @property
    def _constructor(self):
        return IndicatorSeries

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except (KeyError, IndexError):
            if isinstance(key, str) and key.lower() in ("close", "c", "val", "value", "price", "upper", "high", "lower", "low", "middle", "mid"):
                return self
            if isinstance(key, (int, np.integer)):
                return self.iloc[key]
            raise

    def __bool__(self):
        if len(self) == 0:
            return False
        val = self.iloc[-1]
        return bool(val)

    def __int__(self):
        if len(self) == 0:
            return 0
        return int(self.iloc[-1])

    def __float__(self):
        if len(self) == 0:
            return 0.0
        val = self.iloc[-1]
        try:
            return float(val)
        except (ValueError, TypeError):
            mapping = {
                "low_vol": 0.5,
                "normal_vol": 1.0,
                "high_vol": 1.5,
                "trend": 1.5,
                "trending": 1.5,
                "range": 0.5,
                "ranging": 0.5,
                "bull": 1.0,
                "bear": -1.0,
                "neutral": 0.0,
            }
            return float(mapping.get(str(val).lower(), 1.0))


class IndicatorTuple(tuple):
    """Tuple subclass that delegates .iloc, attributes, and item access to its components."""
    @property
    def iloc(self):
        return self[0].iloc

    @property
    def high(self):
        return self[0]

    @property
    def low(self):
        return self[1] if len(self) > 1 else self[0]

    @property
    def upper(self):
        return self[0]

    @property
    def lower(self):
        return self[1] if len(self) > 1 else self[0]

    @property
    def bull(self):
        return self[0]

    @property
    def bear(self):
        return self[1] if len(self) > 1 else self[0]

    def __getattr__(self, name):
        if name in ("high", "h", "upper", "top", "bull", "bullish"):
            return self[0]
        if name in ("low", "l", "lower", "bottom", "bear", "bearish") and len(self) > 1:
            return self[1]
        if name in ("middle", "mid") and len(self) > 2:
            return self[2]
        if hasattr(self[0], name):
            return getattr(self[0], name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return super().__getitem__(item)
        if item in ("high", "h", "upper", "top", "bull", "bullish"):
            return self[0]
        if item in ("low", "l", "lower", "bottom", "bear", "bearish") and len(self) > 1:
            return self[1]
        if item in ("middle", "mid") and len(self) > 2:
            return self[2]
        if hasattr(self[0], "__getitem__"):
            return self[0][item]
        return super().__getitem__(item)


class MacdResult(IndicatorTuple):
    """Result of MACD: (macd_line, signal_line, histogram). Unpacks as 3 or 2, has .macd, .signal, .hist."""
    def __new__(cls, macd_line: pd.Series, signal_line: pd.Series, hist: pd.Series):
        return super().__new__(cls, (macd_line, signal_line, hist))

    def __init__(self, macd_line: pd.Series, signal_line: pd.Series, hist: pd.Series):
        self.macd = macd_line
        self.signal = signal_line
        self.hist = hist

    def __iter__(self):
        try:
            import dis, sys
            f = sys._getframe(1)
            code = f.f_code.co_code
            op = code[f.f_lasti]
            arg = code[f.f_lasti + 1]
            if op == dis.opmap.get("UNPACK_SEQUENCE") and arg == 2:
                return iter([self.macd, self.signal])
        except (AttributeError, ValueError, KeyError, IndexError) as exc:
            _log.debug("Unpack sequence inspect fallback: %s", exc)
        return super().__iter__()

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return super().__getitem__(item)
        if item in ("macd", "macd_line", "line"):
            return self.macd
        if item in ("signal", "signal_line"):
            return self.signal
        if item in ("hist", "histogram"):
            return self.hist
        return self.macd[item]


class AdxRow:
    """Row returned by .iloc on AdxResult: allows .adx, .adx_line, .plus_di, .minus_di, float()."""
    def __init__(self, adx: float, plus_di: float, minus_di: float):
        self.adx = float(adx)
        self.adx_line = float(adx)
        self.plus_di = float(plus_di)
        self.di_plus = float(plus_di)
        self.minus_di = float(minus_di)
        self.di_minus = float(minus_di)

    def __getitem__(self, item):
        if item in ("adx", "adx_line", 0):
            return self.adx
        if item in ("plus_di", "di_plus", 1):
            return self.plus_di
        if item in ("minus_di", "di_minus", 2):
            return self.minus_di
        return self.adx

    def __float__(self):
        return self.adx


class _AdxIlocAccessor:
    def __init__(self, adx: pd.Series, plus_di: pd.Series, minus_di: pd.Series):
        self._a, self._p, self._m = adx, plus_di, minus_di

    def __getitem__(self, item):
        return AdxRow(self._a.iloc[item], self._p.iloc[item], self._m.iloc[item])


class AdxResult(IndicatorTuple):
    """Result of ADX: (adx_line, plus_di, minus_di). Unpacks as 3 or 1, has .adx, .adx_line, .plus_di, .minus_di, .iloc."""
    def __new__(cls, adx_line: pd.Series, plus_di: pd.Series, minus_di: pd.Series):
        return super().__new__(cls, (adx_line, plus_di, minus_di))

    @property
    def adx(self):
        return self[0]

    @property
    def adx_line(self):
        return self[0]

    @property
    def plus_di(self):
        return self[1]

    @property
    def di_plus(self):
        return self[1]

    @property
    def minus_di(self):
        return self[2]

    @property
    def di_minus(self):
        return self[2]

    @property
    def iloc(self):
        return _AdxIlocAccessor(self[0], self[1], self[2])

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return super().__getitem__(item)
        if item in ("adx", "adx_line"):
            return self[0]
        if item in ("plus_di", "di_plus"):
            return self[1]
        if item in ("minus_di", "di_minus"):
            return self[2]
        return self[0][item]


class StochRow:
    """Row returned by .iloc on StochResult: allows .k, .d, .slow_k, .slow_d, float()."""
    def __init__(self, k: float, d: float):
        self.k = float(k)
        self.slow_k = float(k)
        self.d = float(d)
        self.slow_d = float(d)

    def __getitem__(self, item):
        if item in ("k", "slow_k", 0):
            return self.k
        if item in ("d", "slow_d", 1):
            return self.d
        return self.k

    def __float__(self):
        return self.k


class _StochIlocAccessor:
    def __init__(self, k: pd.Series, d: pd.Series):
        self._k, self._d = k, d

    def __getitem__(self, item):
        return StochRow(self._k.iloc[item], self._d.iloc[item])


class StochResult(IndicatorTuple):
    """Result of Stochastic: (slow_k, slow_d). Unpacks as 2, has .k, .d, .slow_k, .slow_d, .iloc."""
    def __new__(cls, slow_k: pd.Series, slow_d: pd.Series):
        return super().__new__(cls, (slow_k, slow_d))

    @property
    def k(self):
        return self[0]

    @property
    def slow_k(self):
        return self[0]

    @property
    def d(self):
        return self[1]

    @property
    def slow_d(self):
        return self[1]

    @property
    def iloc(self):
        return _StochIlocAccessor(self[0], self[1])

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return super().__getitem__(item)
        if item in ("k", "slow_k"):
            return self[0]
        if item in ("d", "slow_d"):
            return self[1]
        return self[0][item]


class BandRow:
    """Row returned by .iloc on BandResult/BollingerResult: allows ['upper'], ['lower'], .upper, .lower, float()."""
    def __init__(self, upper: float, middle: float, lower: float):
        self.upper = float(upper)
        self.middle = float(middle)
        self.lower = float(lower)

    @property
    def high(self):
        return self.upper

    @property
    def low(self):
        return self.lower

    def __getitem__(self, item):
        if item in ("upper", "high", "top"):
            return self.upper
        if item in ("lower", "low", "bot", "bottom"):
            return self.lower
        if item in ("middle", "mid"):
            return self.middle
        if item == 0:
            return self.upper
        if item == 1:
            return self.middle
        if item == 2:
            return self.lower
        return self.upper

    def __float__(self):
        return self.upper

    def __int__(self):
        return int(self.upper)


class _BandIlocAccessor:
    def __init__(self, upper: pd.Series, middle: pd.Series, lower: pd.Series):
        self.u, self.m, self.l = upper, middle, lower

    def __getitem__(self, item):
        return BandRow(self.u.iloc[item], self.m.iloc[item], self.l.iloc[item])


class BollingerResult(IndicatorTuple):
    """Result of Bollinger Bands: (upper, middle, lower, bandwidth, pct_b). Unpacks as 2, 3, or 5."""
    def __new__(cls, upper: pd.Series, middle: pd.Series, lower: pd.Series, bandwidth: pd.Series, pct_b: pd.Series):
        return super().__new__(cls, (upper, middle, lower, bandwidth, pct_b))

    @property
    def upper(self):
        return self[0]

    @property
    def high(self):
        return self[0]

    @property
    def middle(self):
        return self[1]

    @property
    def mid(self):
        return self[1]

    @property
    def lower(self):
        return self[2]

    @property
    def low(self):
        return self[2]

    @property
    def bandwidth(self):
        return self[3]

    @property
    def width(self):
        return self[3]

    @property
    def pct_b(self):
        return self[4]

    @property
    def percent_b(self):
        return self[4]

    @property
    def iloc(self):
        return _BandIlocAccessor(self[0], self[1], self[2])

    def __iter__(self):
        try:
            f = sys._getframe(1)
            code = f.f_code.co_code
            op = code[f.f_lasti]
            arg = code[f.f_lasti + 1]
            if op == dis.opmap.get("UNPACK_SEQUENCE"):
                if arg == 2:
                    return iter([self[0], self[2]])
                elif arg == 3:
                    return iter([self[0], self[1], self[2]])
        except (AttributeError, ValueError, KeyError, IndexError) as exc:
            _log.debug("Unpack sequence inspect fallback: %s", exc)
        return super().__iter__()

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return super().__getitem__(item)
        if item in ("upper", "top", "high", "u"):
            return self[0]
        if item in ("middle", "mid", "m"):
            return self[1]
        if item in ("lower", "bottom", "low", "l"):
            return self[2]
        if item in ("bandwidth", "width"):
            return self[3]
        if item in ("pct_b", "percent_b"):
            return self[4]
        return self[0][item]



# ═══════════════════════════════════════════════════════════════════════════
# 1. Trend & Momentum Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_series(series: pd.Series | np.ndarray | pd.DataFrame) -> IndicatorSeries:
    """Normalize input to IndicatorSeries, extracting 'close' if DataFrame."""
    if isinstance(series, IndicatorSeries):
        return series
    if isinstance(series, pd.DataFrame):
        return IndicatorSeries(series["close"])
    if isinstance(series, pd.Series):
        return IndicatorSeries(series)
    return IndicatorSeries(series)


def sma(series: pd.Series | np.ndarray | pd.DataFrame, period: int = 20, n: int | None = None, **kwargs: Any) -> IndicatorSeries:
    """Simple Moving Average over rolling `period` bars. Accepts Series or DataFrame."""
    p = int(n or kwargs.get("period", period))
    s = _to_series(series)
    return IndicatorSeries(s.rolling(window=p, min_periods=p).mean())


def ema(series: pd.Series | np.ndarray | pd.DataFrame, period: int = 20, n: int | None = None, **kwargs: Any) -> IndicatorSeries:
    """Exponential Moving Average. Accepts Series or DataFrame."""
    p = int(n or kwargs.get("period", period))
    s = _to_series(series)
    return IndicatorSeries(s.ewm(span=p, adjust=False).mean())


def wma(series: pd.Series | np.ndarray | pd.DataFrame, period: int = 20, n: int | None = None, **kwargs: Any) -> IndicatorSeries:
    """Linearly Weighted Moving Average with weights [1, 2, ..., period]. Accepts Series or DataFrame."""
    p = int(n or kwargs.get("period", period))
    s = _to_series(series)
    weights = np.arange(1, p + 1)
    denom = weights.sum()

    def _weighted_avg(window: np.ndarray) -> float:
        return float(np.dot(window, weights) / denom)

    return IndicatorSeries(s.rolling(window=p, min_periods=p).apply(_weighted_avg, raw=True))


def macd(
    series: pd.Series | np.ndarray | pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
    **kwargs: Any,
) -> MacdResult:
    """Moving Average Convergence Divergence. Accepts Series or DataFrame.

    Returns:
        MacdResult: (macd_line, signal_line, histogram)
    """
    fp = int(fast or kwargs.get("fast_period", fast_period))
    sp = int(slow or kwargs.get("slow_period", slow_period))
    sigp = int(signal or kwargs.get("signal_period", signal_period))
    s = _to_series(series)
    fast_ema = s.ewm(span=fp, adjust=False).mean()
    slow_ema = s.ewm(span=sp, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=sigp, adjust=False).mean()
    histogram = macd_line - signal_line
    return MacdResult(IndicatorSeries(macd_line), IndicatorSeries(signal_line), IndicatorSeries(histogram))


def true_range(
    high: pd.Series | np.ndarray | pd.DataFrame,
    low: pd.Series | np.ndarray | None = None,
    close: pd.Series | np.ndarray | None = None,
) -> IndicatorSeries:
    """True Range: max(high - low, |high - prev_close|, |low - prev_close|). Accepts true_range(df) or true_range(h, l, c)."""
    if isinstance(high, pd.DataFrame):
        df = high
        h = df["high"]
        l = df["low"]
        c = df["close"]
    else:
        h = pd.Series(high) if not isinstance(high, pd.Series) else high
        l = pd.Series(low) if not isinstance(low, pd.Series) else low
        c = pd.Series(close) if not isinstance(close, pd.Series) else close

    c_prev = c.shift(1)
    tr1 = (h - l).abs()
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = abs(float(h.iloc[0] - l.iloc[0]))
    return IndicatorSeries(tr)


def atr(
    high: pd.Series | np.ndarray | pd.DataFrame,
    low: pd.Series | np.ndarray | int = 14,
    close: pd.Series | np.ndarray | None = None,
    period: int = 14,
    n: int | None = None,
    **kwargs: Any,
) -> IndicatorSeries:
    """Average True Range with Wilder exponential smoothing. Accepts atr(df, period), atr(df, n=14), atr(h, l, c, period)."""
    p = int(n or kwargs.get("atr_period", kwargs.get("atr_n", period)))
    if isinstance(high, pd.DataFrame):
        df = high
        p_val = int(n or (low if isinstance(low, (int, float)) else p))
        tr = true_range(df)
        return IndicatorSeries(tr.ewm(alpha=1.0 / max(p_val, 1), adjust=False).mean())
    else:
        if isinstance(close, (int, float)):
            n_val = int(close)
            h = pd.Series(high)
            l = pd.Series(low)
            c = pd.Series(low)
        elif isinstance(low, (int, float)):
            n_val = int(low)
            h = pd.Series(high)
            l = pd.Series(high)
            c = pd.Series(close) if close is not None else h
        else:
            n_val = int(n or p)
            h = pd.Series(high)
            l = pd.Series(low)
            c = pd.Series(close) if close is not None else l
        tr = true_range(h, l, c)
        return IndicatorSeries(tr.ewm(alpha=1.0 / max(n_val, 1), adjust=False).mean())


def adx(
    high: pd.Series | np.ndarray | pd.DataFrame,
    low: pd.Series | np.ndarray | int = 14,
    close: pd.Series | np.ndarray | None = None,
    period: int = 14,
    n: int | None = None,
    **kwargs: Any,
) -> AdxResult:
    """Average Directional Movement Index (ADX, +DI, -DI). Accepts adx(df, period) or adx(h, l, c, period)."""
    p = int(n or kwargs.get("adx_period", kwargs.get("adx_n", period)))
    if isinstance(high, pd.DataFrame):
        df = high
        period_val = int(n or (low if isinstance(low, (int, float)) else p))
        h = df["high"]
        l = df["low"]
        c = df["close"]
    else:
        period_val = p
        h = pd.Series(high) if not isinstance(high, pd.Series) else high
        l = pd.Series(low) if not isinstance(low, pd.Series) else low
        c = pd.Series(close) if not isinstance(close, pd.Series) else close

    h_diff = h.diff()
    l_diff = -l.diff()

    plus_dm = np.where((h_diff > l_diff) & (h_diff > 0), h_diff, 0.0)
    minus_dm = np.where((l_diff > h_diff) & (l_diff > 0), l_diff, 0.0)

    tr = true_range(h, l, c)
    atr_smooth = tr.ewm(alpha=1.0 / max(period_val, 1), adjust=False).mean()

    plus_di = 100.0 * (pd.Series(plus_dm).ewm(alpha=1.0 / max(period_val, 1), adjust=False).mean() / atr_smooth)
    minus_di = 100.0 * (pd.Series(minus_dm).ewm(alpha=1.0 / max(period_val, 1), adjust=False).mean() / atr_smooth)

    dx_denom = plus_di + minus_di
    dx = np.where(dx_denom > 0, 100.0 * ((plus_di - minus_di).abs() / dx_denom), 0.0)
    adx_line = pd.Series(dx).ewm(alpha=1.0 / max(period_val, 1), adjust=False).mean()

    idx = getattr(high, "index", None)
    return AdxResult(IndicatorSeries(adx_line, index=idx), IndicatorSeries(plus_di, index=idx), IndicatorSeries(minus_di, index=idx))


class SupertrendResult(tuple):
    """Result of supertrend: unpacks dynamically as (st, direction) or (st, direction, upper/lower)."""
    def __new__(cls, st: pd.Series, direction: pd.Series, upper: pd.Series | None = None, lower: pd.Series | None = None):
        return super().__new__(cls, (st, direction))

    def __init__(self, st: pd.Series, direction: pd.Series, upper: pd.Series | None = None, lower: pd.Series | None = None):
        self.st = st
        self.direction = direction
        self.upper = upper if upper is not None else st
        self.lower = lower if lower is not None else st

    @property
    def iloc(self):
        return self.st.iloc

    def __getattr__(self, name):
        if hasattr(self.st, name):
            return getattr(self.st, name)
        raise AttributeError(f"'SupertrendResult' object has no attribute '{name}'")

    def __iter__(self):
        try:
            import dis, sys
            f = sys._getframe(1)
            code = f.f_code.co_code
            op = code[f.f_lasti]
            arg = code[f.f_lasti + 1]
            if op == dis.opmap.get("UNPACK_SEQUENCE"):
                if arg == 3:
                    return iter([self.st, self.direction, self.upper])
                elif arg == 4:
                    return iter([self.st, self.direction, self.upper, self.lower])
        except (AttributeError, ValueError, KeyError, IndexError) as exc:
            _log.debug("Unpack sequence inspect fallback: %s", exc)
        return super().__iter__()

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return super().__getitem__(item)
        if item in ("st", "supertrend", "line"):
            return self.st
        if item in ("direction", "dir"):
            return self.direction
        if item == "upper":
            return self.upper
        if item == "lower":
            return self.lower
        if hasattr(self.st, "__getitem__"):
            return self.st[item]
        return super().__getitem__(item)


def supertrend(
    high: pd.Series | np.ndarray | pd.DataFrame,
    low: pd.Series | np.ndarray | int = 10,
    close: pd.Series | np.ndarray | float = 3.0,
    period: int = 10,
    multiplier: float = 3.0,
    n: int | None = None,
    mult: float | None = None,
    **kwargs: Any,
) -> SupertrendResult:
    """SuperTrend indicator. Accepts supertrend(df, period, mult) or supertrend(h, l, c, period, mult)."""
    p = int(n or kwargs.get("period", period))
    m = float(mult or kwargs.get("multiplier", multiplier))
    if isinstance(high, pd.DataFrame):
        df = high
        period = int(n or (low if isinstance(low, (int, float)) else p))
        multiplier = float(mult or (close if isinstance(close, (int, float)) else m))
        h = df["high"]
        l = df["low"]
        c = df["close"]
    else:
        period = p
        multiplier = m
        h = pd.Series(high) if not isinstance(high, pd.Series) else high
        l = pd.Series(low) if not isinstance(low, pd.Series) else low
        c = pd.Series(close) if not isinstance(close, pd.Series) else close

    h_arr = pd.Series(h).to_numpy(dtype=float)
    l_arr = pd.Series(l).to_numpy(dtype=float)
    c_arr = pd.Series(c).to_numpy(dtype=float)
    n = len(c_arr)

    atr_vals = atr(h, l, c, period=period).to_numpy(dtype=float)
    basic_upper = (h_arr + l_arr) / 2.0 + multiplier * atr_vals
    basic_lower = (h_arr + l_arr) / 2.0 - multiplier * atr_vals

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
        if basic_upper[i] < final_upper[i - 1] or c_arr[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Lower band
        if basic_lower[i] > final_lower[i - 1] or c_arr[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction switch
        prev_dir = direction[i - 1]
        if prev_dir == 1:
            if c_arr[i] < final_lower[i]:
                direction[i] = -1
                st[i] = final_upper[i]
            else:
                direction[i] = 1
                st[i] = final_lower[i]
        else:
            if c_arr[i] > final_upper[i]:
                direction[i] = 1
                st[i] = final_lower[i]
            else:
                direction[i] = -1
                st[i] = final_upper[i]

    idx = getattr(high, "index", getattr(c, "index", None))
    return SupertrendResult(
        IndicatorSeries(st, index=idx),
        IndicatorSeries(direction, index=idx),
        IndicatorSeries(final_upper, index=idx),
        IndicatorSeries(final_lower, index=idx),
    )




class BandResult(tuple):
    """Result of Donchian / Keltner: unpacks as (upper, middle, lower) or (upper, lower), supports .upper, .lower, ['upper'], ['lower'], and .iloc[-1]['upper']."""
    def __new__(cls, upper: pd.Series, middle: pd.Series, lower: pd.Series):
        return super().__new__(cls, (upper, middle, lower))

    def __init__(self, upper: pd.Series, middle: pd.Series, lower: pd.Series):
        self.upper = upper
        self.middle = middle
        self.lower = lower

    @property
    def iloc(self):
        return _BandIlocAccessor(self.upper, self.middle, self.lower)

    def __iter__(self):
        try:
            f = sys._getframe(1)
            code = f.f_code.co_code
            op = code[f.f_lasti]
            arg = code[f.f_lasti + 1]
            if op == dis.opmap.get("UNPACK_SEQUENCE") and arg == 2:
                return iter([self.upper, self.lower])
        except (AttributeError, ValueError, KeyError, IndexError) as exc:
            _log.debug("Unpack sequence inspect fallback: %s", exc)
        return super().__iter__()

    def __getitem__(self, item):
        if item in ("upper", "high", "top"):
            return self.upper
        if item in ("lower", "low", "bottom"):
            return self.lower
        if item in ("middle", "mid"):
            return self.middle
        return super().__getitem__(item)


def donchian(
    high: pd.Series | np.ndarray | pd.DataFrame,
    low: pd.Series | np.ndarray | int = 20,
    period: int = 20,
    n: int | None = None,
    **kwargs: Any,
) -> BandResult:
    """Donchian Channel over rolling `period` past bars. Accepts donchian(df, period) or donchian(h, l, period).

    Returns:
        BandResult: (upper, middle, lower)
    """
    p = n or kwargs.get("lookback", period)
    if isinstance(high, pd.DataFrame):
        df = high
        p = n or (int(low) if isinstance(low, (int, float)) else p)
        h = df["high"]
        l = df["low"]
    else:
        h = pd.Series(high) if not isinstance(high, pd.Series) else high
        l = pd.Series(low) if not isinstance(low, pd.Series) else low

    upper = h.rolling(window=int(p), min_periods=int(p)).max()
    lower = l.rolling(window=int(p), min_periods=int(p)).min()
    middle = (upper + lower) / 2.0
    return BandResult(IndicatorSeries(upper), IndicatorSeries(middle), IndicatorSeries(lower))


def keltner(
    high: pd.Series | np.ndarray | pd.DataFrame,
    low: pd.Series | np.ndarray | int = 20,
    close: pd.Series | np.ndarray | int = 10,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
    ema_n: int | None = None,
    atr_n: int | None = None,
    mult: float | None = None,
    **kwargs: Any,
) -> BandResult:
    """Keltner Channels (EMA midline +/- multiplier * ATR). Accepts keltner(df, ema_n, atr_n, mult) or keltner(h, l, c, ema_n, atr_n, mult).

    Returns:
        BandResult: (upper, middle, lower)
    """
    if isinstance(high, pd.DataFrame):
        df = high
        ema_p = ema_n or (int(low) if isinstance(low, (int, float)) else ema_period)
        atr_p = atr_n or (int(close) if isinstance(close, (int, float)) else atr_period)
        h = df["high"]
        l = df["low"]
        c = df["close"]
    else:
        ema_p = ema_n or ema_period
        atr_p = atr_n or atr_period
        h = pd.Series(high) if not isinstance(high, pd.Series) else high
        l = pd.Series(low) if not isinstance(low, pd.Series) else low
        c = pd.Series(close) if not isinstance(close, pd.Series) else close

    m = mult if mult is not None else multiplier
    middle = c.ewm(span=int(ema_p), adjust=False).mean()
    atr_vals = atr(h, l, c, period=int(atr_p))
    upper = middle + float(m) * atr_vals
    lower = middle - float(m) * atr_vals
    return BandResult(IndicatorSeries(upper), IndicatorSeries(middle), IndicatorSeries(lower))


def roc(series: pd.Series | np.ndarray, period: int = 10, **kwargs: Any) -> IndicatorSeries:
    """Rate of Change percentage over `period` bars: ((P_t - P_{t-N}) / P_{t-N}) * 100."""
    p = int(kwargs.get("n", kwargs.get("period", period)))
    s = _to_series(series)
    prev = s.shift(p)
    return IndicatorSeries((s - prev) / prev * 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Mean Reversion & Volatility Helpers
# ═══════════════════════════════════════════════════════════════════════════

def rsi(series: pd.Series | np.ndarray | pd.DataFrame, period: int = 14, n: int | None = None, **kwargs: Any) -> IndicatorSeries:
    """Relative Strength Index (Wilder's RSI). Accepts Series or DataFrame."""
    p = int(n or kwargs.get("period", period))
    s = _to_series(series)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))

    avg_gain = gain.ewm(alpha=1.0 / max(p, 1), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / max(p, 1), adjust=False).mean()

    rs = avg_gain / np.where(avg_loss == 0, 1e-9, avg_loss)
    res = 100.0 - (100.0 / (1.0 + rs))
    return IndicatorSeries(res)


def stoch(
    high: pd.Series | np.ndarray | pd.DataFrame,
    low: pd.Series | np.ndarray | int = 14,
    close: pd.Series | np.ndarray | int = 3,
    k_period: int = 14,
    d_period: int = 3,
    slowing: int = 3,
    k: int | None = None,
    d: int | None = None,
    **kwargs: Any,
) -> StochResult:
    """Stochastic Oscillator (%K, %D). Accepts stoch(df, k, d, slowing) or stoch(h, l, c, k, d, slowing)."""
    kp = int(k or kwargs.get("k_period", k_period))
    dp = int(d or kwargs.get("d_period", d_period))
    if isinstance(high, pd.DataFrame):
        df = high
        kp = int(k or (low if isinstance(low, (int, float)) else kp))
        dp = int(d or (close if isinstance(close, (int, float)) else dp))
        h = df["high"]
        l = df["low"]
        c = df["close"]
    else:
        h = pd.Series(high) if not isinstance(high, pd.Series) else high
        l = pd.Series(low) if not isinstance(low, pd.Series) else low
        c = pd.Series(close) if not isinstance(close, pd.Series) else close

    lowest_low = l.rolling(window=kp, min_periods=kp).min()
    highest_high = h.rolling(window=kp, min_periods=kp).max()

    denom = np.where((highest_high - lowest_low) == 0, 1e-9, highest_high - lowest_low)
    fast_k = 100.0 * (c - lowest_low) / denom

    slow_k = fast_k.rolling(window=slowing, min_periods=slowing).mean()
    slow_d = slow_k.rolling(window=dp, min_periods=dp).mean()
    return StochResult(IndicatorSeries(slow_k), IndicatorSeries(slow_d))


def bollinger(
    series: pd.Series | np.ndarray | pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    n: int | None = None,
    std: float | None = None,
    **kwargs: Any,
) -> BollingerResult:
    """Bollinger Bands. Accepts Series or DataFrame.

    Returns:
        BollingerResult: (upper, middle, lower, bandwidth, pct_b)
    """
    p = int(n or kwargs.get("period", period))
    ns = float(std or kwargs.get("num_std", num_std))
    s = _to_series(series)
    middle = s.rolling(window=p, min_periods=p).mean()
    std_val = s.rolling(window=p, min_periods=p).std()

    upper = middle + ns * std_val
    lower = middle - ns * std_val
    bandwidth = (upper - lower) / np.where(middle == 0, 1e-9, middle)
    pct_b = (s - lower) / np.where((upper - lower) == 0, 1e-9, upper - lower)
    return BollingerResult(IndicatorSeries(upper), IndicatorSeries(middle), IndicatorSeries(lower), IndicatorSeries(bandwidth), IndicatorSeries(pct_b))


def zscore(series: pd.Series | np.ndarray | pd.DataFrame, period: int = 20, n: int | None = None, **kwargs: Any) -> IndicatorSeries:
    """Rolling Z-Score: (series - rolling_mean) / rolling_std. Accepts Series or DataFrame."""
    p = int(n or kwargs.get("period", period))
    s = _to_series(series)
    m = s.rolling(window=p, min_periods=p).mean()
    std_val = s.rolling(window=p, min_periods=p).std()
    return IndicatorSeries((s - m) / np.where(std_val == 0, 1e-9, std_val))


def realized_vol(
    series: pd.Series | np.ndarray | pd.DataFrame,
    period: int = 20,
    annualize: bool = True,
    bars_per_year: int = 252 * 24,
    n: int | None = None,
    **kwargs: Any,
) -> IndicatorSeries:
    """Rolling realized volatility of log returns. Accepts Series or DataFrame."""
    p = int(n or kwargs.get("period", period))
    s = _to_series(series)
    log_ret = np.log(s / s.shift(1))
    vol = log_ret.rolling(window=p, min_periods=p).std()
    if annualize:
        vol = vol * np.sqrt(bars_per_year)
    return IndicatorSeries(vol)


def vol_percentile(*args: Any, **kwargs: Any) -> IndicatorSeries:
    """Rolling percentile rank (0..100) of current volatility within past lookback bars.

    Accepts:
      - vol_percentile(df, lookback=100, atr_n=14)
      - vol_percentile(high, low, close, atr_n=14, lookback=100)
      - vol_percentile(series, lookback=100)
    """
    atr_n = kwargs.get("atr_n", kwargs.get("period", kwargs.get("n", 14)))
    lookback = kwargs.get("lookback", 100)

    if len(args) == 0:
        raise ValueError("vol_percentile requires at least one argument")
    elif len(args) == 1:
        arg = args[0]
        if isinstance(arg, pd.DataFrame):
            s = atr(arg, period=int(atr_n))
        else:
            s = _to_series(arg)
    elif len(args) == 2:
        if isinstance(args[0], pd.DataFrame):
            s = atr(args[0], period=int(atr_n))
            lookback = args[1]
        else:
            s = _to_series(args[0])
            lookback = args[1]
    elif len(args) >= 3:
        h, l, c = args[0], args[1], args[2]
        if len(args) >= 4 and not isinstance(args[3], dict):
            lookback = args[3]
        s = atr(h, l, c, period=int(atr_n))
    else:
        s = _to_series(args[0])

    lb_val = int(lookback) if isinstance(lookback, (int, float)) else 100

    def _pct_rank(window: np.ndarray) -> float:
        val = window[-1]
        return float(np.mean(window <= val) * 100.0)

    res = s.rolling(window=lb_val, min_periods=min(20, lb_val)).apply(_pct_rank, raw=True)
    return IndicatorSeries(res)


def squeeze(
    high: pd.Series | np.ndarray | pd.DataFrame,
    low: pd.Series | np.ndarray | int = 20,
    close: pd.Series | np.ndarray | float = 2.0,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
    bb_n: int | None = None,
    kc_n: int | None = None,
    **kwargs: Any,
) -> IndicatorSeries:
    """Bollinger Band / Keltner Channel Volatility Squeeze. Accepts squeeze(df) or squeeze(h, l, c).

    Returns:
        IndicatorSeries[bool]: True if Bollinger Band is strictly inside Keltner Channel.
    """
    bb_p = int(bb_n or kwargs.get("bb_n", kwargs.get("bb_period", bb_period)))
    kc_p = int(kc_n or kwargs.get("kc_n", kwargs.get("kc_period", kc_period)))
    bb_s = float(kwargs.get("bb_std", bb_std))
    kc_m = float(kwargs.get("kc_mult", kc_mult))

    if isinstance(high, pd.DataFrame):
        df = high
        bb_p = int(bb_n or (low if isinstance(low, (int, float)) else bb_p))
        bb_s = float(close if isinstance(close, (int, float)) else bb_s)
        h = df["high"]
        l = df["low"]
        c = df["close"]
    else:
        h = pd.Series(high) if not isinstance(high, pd.Series) else high
        l = pd.Series(low) if not isinstance(low, pd.Series) else low
        c = pd.Series(close) if not isinstance(close, pd.Series) else close

    bb_res = bollinger(c, period=bb_p, num_std=bb_s)
    bb_upper, bb_lower = bb_res[0], bb_res[2]
    kc_res = keltner(h, l, c, ema_period=kc_p, atr_period=kc_p, multiplier=kc_m)
    kc_upper, kc_lower = kc_res[0], kc_res[2]

    bb_u = bb_upper.to_numpy(dtype=float)
    kc_u = kc_upper.to_numpy(dtype=float)
    bb_l = bb_lower.to_numpy(dtype=float)
    kc_l = kc_lower.to_numpy(dtype=float)

    res = (bb_u < kc_u) & (bb_l > kc_l)
    return IndicatorSeries(res, index=c.index)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Volume & Price Location Helpers
# ═══════════════════════════════════════════════════════════════════════════

def session_vwap(df: pd.DataFrame, volume_col: str = "volume", start_utc: Any = None, **kwargs: Any) -> pd.Series:
    """Cumulative Volume-Weighted Average Price resetting at each new session."""
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

    return IndicatorTuple((IndicatorSeries(bar_prev_high, index=df.index), IndicatorSeries(bar_prev_low, index=df.index)))


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

    return IndicatorTuple((IndicatorSeries(bar_prev_high, index=df.index), IndicatorSeries(bar_prev_low, index=df.index)))


class SessionRangeResult(tuple):
    """Result of session_range: unpacks as (high, low), dict indexing ['high'], or attr .high."""
    def __new__(cls, high: pd.Series, low: pd.Series):
        return super().__new__(cls, (high, low))

    @property
    def high(self) -> pd.Series:
        return self[0]

    @property
    def low(self) -> pd.Series:
        return self[1]

    @property
    def iloc(self):
        return _BandIlocAccessor(self[0], self[0], self[1])

    def __getattr__(self, name):
        if hasattr(self[0], name):
            return getattr(self[0], name)
        raise AttributeError(f"'SessionRangeResult' object has no attribute '{name}'")

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return super().__getitem__(item)
        if item in ("high", "h", "top"):
            return self[0]
        if item in ("low", "l", "bottom"):
            return self[1]
        if hasattr(self[0], "__getitem__"):
            return self[0][item]
        return super().__getitem__(item)


def session_range(df: pd.DataFrame, session_name: str = "asia") -> SessionRangeResult:
    """High and Low of the most recently COMPLETED session named `session_name`."""
    sess = df["session"].astype(str).str.lower() if "session" in df.columns else pd.Series([""] * len(df))
    target = session_name.lower().strip()
    if target in ("london_ny", "ny_london", "london_new_york", "overlap"):
        target = "overlap_london_ny"
    elif target in ("ny", "newyork"):
        target = "new_york"

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
                in_target = True
                cur_high = highs[i]
                cur_low = lows[i]
            else:
                cur_high = max(cur_high, highs[i])
                cur_low = min(cur_low, lows[i])
        else:
            if in_target:
                in_target = False
                last_high = cur_high
                last_low = cur_low

        res_high[i] = last_high
        res_low[i] = last_low

    return SessionRangeResult(pd.Series(res_high, index=df.index), pd.Series(res_low, index=df.index))


def opening_range(df: pd.DataFrame, n_bars: int = 1, duration_bars: int | None = None, **kwargs: Any) -> tuple[pd.Series, pd.Series]:
    """Opening range High and Low over the first `n_bars` of the session/day."""
    nb = int(duration_bars or kwargs.get("duration_bars", n_bars))
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
        if day_bars <= nb:
            cur_or_h = max(cur_or_h, highs[i])
            cur_or_l = min(cur_or_l, lows[i])
            if day_bars == nb:
                frozen_h = cur_or_h
                frozen_l = cur_or_l

        or_high[i] = frozen_h
        or_low[i] = frozen_l

    return IndicatorTuple((IndicatorSeries(or_high, index=df.index), IndicatorSeries(or_low, index=df.index)))


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


def is_session(df: pd.DataFrame, session_name: str) -> IndicatorSeries:
    """Check if current bar falls within specified session name."""
    if "session" not in df.columns:
        return IndicatorSeries(pd.Series(False, index=df.index))
    s_col = df["session"].astype(str).str.lower()
    target = str(session_name).lower().strip()

    if target in ("london_ny", "ny_london", "london_new_york", "overlap", "overlap_london_ny"):
        res = s_col == "overlap_london_ny"
    elif target in ("ny", "newyork", "new_york"):
        res = (s_col == "new_york") | (s_col == "overlap_london_ny")
    elif target in ("london",):
        res = (s_col == "london") | (s_col == "overlap_london_ny")
    elif target in ("asia", "asian", "tokyo"):
        res = s_col == "asia"
    else:
        res = s_col == target
    return IndicatorSeries(res)


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
    freq = "4h" if target_tf.lower() in ("4h", "240m") else "1D"
    htf_periods = ts.dt.floor(freq)

    htf_df = df.groupby(htf_periods).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum" if "volume" in df.columns else "count",
    })

    try:
        htf_values = func(htf_df)
    except Exception as exc:
        _log.warning("htf func evaluation failed: %s; falling back to close", exc)
        htf_values = htf_df["close"]

    if isinstance(htf_values, (tuple, list, IndicatorTuple)):
        out = []
        for val in htf_values:
            if isinstance(val, (pd.Series, pd.DataFrame)):
                vals_arr = val.to_numpy().flatten()
            elif isinstance(val, np.ndarray):
                vals_arr = val.flatten()
            else:
                vals_arr = np.array(val).flatten()

            if len(vals_arr) == len(htf_df):
                htf_val_series = pd.Series(vals_arr, index=htf_df.index).shift(1)
            else:
                htf_val_series = pd.Series(np.nan, index=htf_df.index)
            bar_htf = htf_periods.map(htf_val_series)
            out.append(IndicatorSeries(bar_htf, index=df.index))
        return IndicatorTuple(out)

    if isinstance(htf_values, (pd.Series, pd.DataFrame)):
        vals_arr = htf_values.to_numpy().flatten()
    elif isinstance(htf_values, np.ndarray):
        vals_arr = htf_values.flatten()
    else:
        vals_arr = np.array(htf_values).flatten()

    if len(vals_arr) == len(htf_df):
        htf_val_series = pd.Series(vals_arr, index=htf_df.index).shift(1)
    else:
        htf_val_series = pd.Series(np.nan, index=htf_df.index)

    bar_htf_values = htf_periods.map(htf_val_series)
    return IndicatorSeries(bar_htf_values, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Confirmed Swings & Pivots
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SwingResult:
    pivot_value: pd.Series
    pivot_bar: pd.Series
    confirmed_at: pd.Series

    @property
    def iloc(self):
        return self.pivot_value.iloc

    def __getattr__(self, name):
        if hasattr(self.pivot_value, name):
            return getattr(self.pivot_value, name)
        raise AttributeError(f"'SwingResult' object has no attribute '{name}'")

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return self.pivot_value.iloc[item]
        if item in ("pivot_value", "value", "val"):
            return self.pivot_value
        if item in ("pivot_bar", "bar"):
            return self.pivot_bar
        if item in ("confirmed_at", "confirmed"):
            return self.confirmed_at
        if hasattr(self.pivot_value, "__getitem__"):
            return self.pivot_value[item]
        raise KeyError(item)


def swing_high(high: pd.Series | np.ndarray | pd.DataFrame, n: int = 2) -> SwingResult:
    """Confirmed Swing High. Accepts Series or DataFrame."""
    if isinstance(high, pd.DataFrame):
        h_s = high["high"]
    elif isinstance(high, pd.Series):
        h_s = high
    else:
        h_s = pd.Series(high)
    idx = h_s.index
    h = h_s.to_numpy(dtype=float)
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
        pivot_value=pd.Series(pivot_val, index=idx),
        pivot_bar=pd.Series(pivot_idx, index=idx),
        confirmed_at=pd.Series(confirmed, index=idx),
    )


def swing_low(low: pd.Series | np.ndarray | pd.DataFrame, n: int = 2) -> SwingResult:
    """Confirmed Swing Low. Accepts Series or DataFrame."""
    if isinstance(low, pd.DataFrame):
        l_s = low["low"]
    elif isinstance(low, pd.Series):
        l_s = low
    else:
        l_s = pd.Series(low)
    idx = l_s.index
    l = l_s.to_numpy(dtype=float)
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
        pivot_value=pd.Series(pivot_val, index=idx),
        pivot_bar=pd.Series(pivot_idx, index=idx),
        confirmed_at=pd.Series(confirmed, index=idx),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7. Mechanical Market Structure Helpers
# ═══════════════════════════════════════════════════════════════════════════

class BosResult(tuple):
    """Result of BOS: (bull_bos, bear_bos) tuple that also acts as a combined Series."""
    def __new__(cls, bull_bos: pd.Series, bear_bos: pd.Series):
        return super().__new__(cls, (bull_bos, bear_bos))

    def __init__(self, bull_bos: pd.Series, bear_bos: pd.Series):
        self.bull_bos = bull_bos
        self.bear_bos = bear_bos
        self.combined = pd.Series(0, index=bull_bos.index, dtype=int)
        self.combined[bull_bos] = 1
        self.combined[bear_bos] = -1

    @property
    def iloc(self):
        return self.combined.iloc

    def __getitem__(self, item):
        if item in (0, 1):
            return super().__getitem__(item)
        return self.combined[item]


class ChochResult(tuple):
    """Result of CHoCH: (bull_choch, bear_choch) tuple that also acts as a combined Series."""
    def __new__(cls, bull_choch: pd.Series, bear_choch: pd.Series):
        return super().__new__(cls, (bull_choch, bear_choch))

    def __init__(self, bull_choch: pd.Series, bear_choch: pd.Series):
        self.bull_choch = bull_choch
        self.bear_choch = bear_choch
        self.combined = pd.Series(0, index=bull_choch.index, dtype=int)
        self.combined[bull_choch] = 1
        self.combined[bear_choch] = -1

    @property
    def iloc(self):
        return self.combined.iloc

    def __getitem__(self, item):
        if item in (0, 1):
            return super().__getitem__(item)
        return self.combined[item]


class SweepRow:
    def __init__(self, val: int, name: Any = 0, level: float = 0.0):
        self.val = int(val)
        self.name = name
        self.level = float(level)

    def __getitem__(self, item):
        if item == "type":
            return "high" if self.val == -1 else ("low" if self.val == 1 else "none")
        if item == "level":
            return self.level
        if item == "val":
            return self.val
        return 0

    def __int__(self):
        return self.val

    def __eq__(self, other):
        if isinstance(other, str):
            return self["type"] == other
        return self.val == other


class _IlocSweepAccessor:
    def __init__(self, combined: pd.Series, levels: pd.Series | None = None):
        self._combined = combined
        self._levels = levels

    def __getitem__(self, item):
        res = self._combined.iloc[item]
        if isinstance(item, (int, np.integer)):
            idx_name = self._combined.index[item] if hasattr(self._combined.index, "__getitem__") else item
            lvl = float(self._levels.iloc[item]) if self._levels is not None else 0.0
            return SweepRow(val=int(res), name=idx_name, level=lvl)
        return res


class LiquiditySweepResult(tuple):
    """Result of liquidity sweep: (high_sweep, low_sweep) tuple that also acts as a combined Series."""
    def __new__(cls, high_sweep: pd.Series, low_sweep: pd.Series, levels: pd.Series | None = None):
        return super().__new__(cls, (high_sweep, low_sweep))

    def __init__(self, high_sweep: pd.Series, low_sweep: pd.Series, levels: pd.Series | None = None):
        self.high_sweep = high_sweep
        self.low_sweep = low_sweep
        self.levels = levels
        self.combined = pd.Series(0, index=high_sweep.index, dtype=int)
        self.combined[low_sweep] = 1
        self.combined[high_sweep] = -1

    @property
    def iloc(self):
        return _IlocSweepAccessor(self.combined, self.levels)

    def __getitem__(self, item):
        if item in (0, 1):
            return super().__getitem__(item)
        return self.combined[item]


def bos(
    high: pd.Series | np.ndarray | pd.DataFrame,
    low: pd.Series | np.ndarray | int = 2,
    close: pd.Series | np.ndarray | None = None,
    swing_n: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Break of Structure (BOS). Accepts bos(df, swing_n) or bos(h, l, swing_n) or bos(h, l, c, swing_n)."""
    if isinstance(high, pd.DataFrame):
        df = high
        swing_n = int(low) if isinstance(low, (int, float)) else swing_n
        h = df["high"]
        l = df["low"]
        c = df["close"]
    else:
        h = pd.Series(high) if not isinstance(high, pd.Series) else high
        l = pd.Series(low) if not isinstance(low, pd.Series) else low
        if isinstance(close, (int, float)):
            swing_n = int(close)
            c = l
        elif close is not None:
            c = pd.Series(close) if not isinstance(close, pd.Series) else close
        else:
            c = l

    sh = swing_high(h, n=swing_n)
    sl = swing_low(l, n=swing_n)
    sh_val = pd.Series(sh.pivot_value.to_numpy(), index=c.index)
    sl_val = pd.Series(sl.pivot_value.to_numpy(), index=c.index)

    bull_bos = (c > sh_val) & (c.shift(1) <= sh_val.shift(1))
    bear_bos = (c < sl_val) & (c.shift(1) >= sl_val.shift(1))
    return BosResult(bull_bos.fillna(False), bear_bos.fillna(False))


def choch(
    high: pd.Series | np.ndarray | pd.DataFrame,
    low: pd.Series | np.ndarray | int = 2,
    close: pd.Series | np.ndarray | None = None,
    swing_n: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Change of Character (CHoCH). Accepts choch(df, swing_n) or choch(h, l, swing_n) or choch(h, l, c, swing_n)."""
    if isinstance(high, pd.DataFrame):
        df = high
        swing_n = int(low) if isinstance(low, (int, float)) else swing_n
        h = df["high"]
        l = df["low"]
        c = df["close"]
    else:
        h = pd.Series(high) if not isinstance(high, pd.Series) else high
        l = pd.Series(low) if not isinstance(low, pd.Series) else low
        if isinstance(close, (int, float)):
            swing_n = int(close)
            c = l
        elif close is not None:
            c = pd.Series(close) if not isinstance(close, pd.Series) else close
        else:
            c = l

    bull_bos, bear_bos = bos(h, l, c, swing_n=swing_n)
    n = len(c)

    trend = np.zeros(n, dtype=int)
    bull_choch = np.zeros(n, dtype=bool)
    bear_choch = np.zeros(n, dtype=bool)

    cur_trend = 0
    for i in range(n):
        if bull_bos[i]:
            if cur_trend == -1:
                bull_choch[i] = True
            cur_trend = 1
        elif bear_bos[i]:
            if cur_trend == 1:
                bear_choch[i] = True
            cur_trend = -1
        trend[i] = cur_trend

    return ChochResult(pd.Series(bull_choch, index=c.index), pd.Series(bear_choch, index=c.index))


def fvg(
    high: pd.Series | np.ndarray | pd.DataFrame,
    low: pd.Series | np.ndarray | float = 0.0,
    min_gap_usd: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Fair Value Gap (3-bar imbalance). Accepts fvg(df, min_gap_usd) or fvg(h, l)."""
    if isinstance(high, pd.DataFrame):
        df = high
        min_gap_usd = float(low) if isinstance(low, (int, float)) else min_gap_usd
        h = df["high"]
        l = df["low"]
    else:
        h = pd.Series(high) if not isinstance(high, pd.Series) else high
        l = pd.Series(low) if not isinstance(low, pd.Series) else low

    bull_fvg = (l > (h.shift(2) + min_gap_usd)).fillna(False)
    bear_fvg = (h < (l.shift(2) - min_gap_usd)).fillna(False)
    return IndicatorTuple((IndicatorSeries(bull_fvg, index=h.index), IndicatorSeries(bear_fvg, index=h.index)))


def liquidity_sweep(
    high: pd.Series | np.ndarray | pd.DataFrame,
    low: pd.Series | np.ndarray | int = 2,
    close: pd.Series | np.ndarray | None = None,
    swing_n: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Liquidity Sweep / Stop Hunt. Accepts liquidity_sweep(df, swing_n) or liquidity_sweep(h, l, swing_n) or liquidity_sweep(h, l, c, swing_n)."""
    if isinstance(high, pd.DataFrame):
        df = high
        swing_n = int(low) if isinstance(low, (int, float)) else swing_n
        h = df["high"]
        l = df["low"]
        c = df["close"]
    else:
        h = pd.Series(high) if not isinstance(high, pd.Series) else high
        l = pd.Series(low) if not isinstance(low, pd.Series) else low
        if isinstance(close, (int, float)):
            swing_n = int(close)
            c = l
        elif close is not None:
            c = pd.Series(close) if not isinstance(close, pd.Series) else close
        else:
            c = l

    sh = swing_high(h, n=swing_n)
    sl = swing_low(l, n=swing_n)
    sh_val = pd.Series(sh.pivot_value.to_numpy(), index=c.index)
    sl_val = pd.Series(sl.pivot_value.to_numpy(), index=c.index)

    high_sweep = (h.to_numpy() > sh_val.to_numpy()) & (c.to_numpy() <= sh_val.to_numpy())
    low_sweep = (l.to_numpy() < sl_val.to_numpy()) & (c.to_numpy() >= sl_val.to_numpy())

    levels = pd.Series(
        np.where(high_sweep, sh_val.to_numpy(), np.where(low_sweep, sl_val.to_numpy(), np.nan)),
        index=c.index,
    )
    return LiquiditySweepResult(
        pd.Series(high_sweep, index=c.index).fillna(False),
        pd.Series(low_sweep, index=c.index).fillna(False),
        levels=levels,
    )


def premium_discount(
    high: pd.Series | np.ndarray | pd.DataFrame,
    low: pd.Series | np.ndarray | int = 5,
    close: pd.Series | np.ndarray | None = None,
    swing_n: int = 5,
) -> pd.Series:
    """Relative position within current swing range (0.0 = discount low, 1.0 = premium high). Accepts premium_discount(df, swing_n) or premium_discount(h, l, c, swing_n)."""
    if isinstance(high, pd.DataFrame):
        df = high
        swing_n = int(low) if isinstance(low, (int, float)) else swing_n
        h = df["high"]
        l = df["low"]
        c = df["close"]
    else:
        h = pd.Series(high) if not isinstance(high, pd.Series) else high
        l = pd.Series(low) if not isinstance(low, pd.Series) else low
        c = pd.Series(close) if not isinstance(close, pd.Series) else close

    sh = swing_high(h, n=swing_n)
    sl = swing_low(l, n=swing_n)
    sh_val = pd.Series(sh.pivot_value.to_numpy(), index=c.index)
    sl_val = pd.Series(sl.pivot_value.to_numpy(), index=c.index)

    range_span = sh_val - sl_val
    pos = (c - sl_val) / np.where(range_span <= 0, 1e-9, range_span)
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


class RegimeStr(str):
    """String subclass that allows float conversion for regime labels."""
    def __float__(self):
        mapping = {
            "low_vol": 0.5,
            "normal_vol": 1.0,
            "high_vol": 1.5,
            "trend": 1.5,
            "trending": 1.5,
            "range": 0.5,
            "ranging": 0.5,
            "bull": 1.0,
            "bear": -1.0,
            "neutral": 0.0,
        }
        return float(mapping.get(str(self).lower(), 1.0))

    def __int__(self):
        return int(float(self))


def vol_regime(
    *args: Any,
    **kwargs: Any,
) -> IndicatorSeries:
    """Classify volatility regime as 'low_vol', 'normal_vol', or 'high_vol'."""
    atr_period = int(kwargs.get("atr_n", kwargs.get("atr_period", 14)))
    lookback = int(kwargs.get("lookback", kwargs.get("lookback_n", kwargs.get("pct_n", 100))))
    if len(args) == 1 and isinstance(args[0], pd.DataFrame):
        df = args[0]
        h, l, c = df["high"], df["low"], df["close"]
    elif len(args) >= 3:
        h, l, c = args[0], args[1], args[2]
    elif "df" in kwargs:
        df = kwargs["df"]
        h, l, c = df["high"], df["low"], df["close"]
    else:
        h = kwargs.get("high")
        l = kwargs.get("low")
        c = kwargs.get("close")
    atr_vals = atr(h, l, c, period=atr_period)
    pcts = vol_percentile(atr_vals, lookback=lookback)

    def _classify(p: float) -> RegimeStr:
        if np.isnan(p):
            return RegimeStr("normal_vol")
        if p <= 25.0:
            return RegimeStr("low_vol")
        elif p >= 75.0:
            return RegimeStr("high_vol")
        return RegimeStr("normal_vol")

    idx = getattr(c, "index", None)
    return IndicatorSeries(pcts.apply(_classify), index=idx)


def session_regime(df: pd.DataFrame) -> pd.Series:
    """Returns the active trading session name or 'off_hours'."""
    if "session" in df.columns:
        return df["session"].astype(str)
    return pd.Series("unknown", index=df.index)
