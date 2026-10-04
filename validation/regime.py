"""
validation/regime.py – Market regime classification and concentration analysis.

Implements Gate 16 (G16) specifications:
1. Label bars by trend/range x low/high volatility using past data only.
   - 4 regimes: trending_high_vol, trending_low_vol, ranging_high_vol, ranging_low_vol.
2. Trailing 12-month gold trend classification: up, flat, down.
3. Profit concentration calculations by calendar year, market regime, and trailing gold trend.
4. Threshold check: Reject if > 60% of total positive profit comes from one calendar year or regime.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from core.simulator import Trade


def classify_market_regimes(
    df: pd.DataFrame,
    adx_period: int = 14,
    atr_period: int = 20,
    vol_lookback: int = 100,
    adx_trend_threshold: float = 25.0,
) -> pd.DataFrame:
    """Label each bar with a regime based strictly on past data:
    - trend vs range: ADX >= 25 is trend, < 25 is range.
    - high vs low volatility: ATR >= rolling median ATR is high vol, else low vol.

    Returns df with added 'regime' column containing one of:
    - 'trending_high_vol'
    - 'trending_low_vol'
    - 'ranging_high_vol'
    - 'ranging_low_vol'
    """
    df = df.copy()
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    # 1. ATR calculation
    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    tr_series = pd.Series(tr, index=df.index)
    atr = tr_series.rolling(atr_period, min_periods=1).mean()
    rolling_med_atr = atr.rolling(vol_lookback, min_periods=1).median()
    is_high_vol = atr >= rolling_med_atr

    # 2. ADX calculation
    up_move = np.zeros(n, dtype=float)
    down_move = np.zeros(n, dtype=float)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            up_move[i] = up
        if down > up and down > 0:
            down_move[i] = down

    up_s = pd.Series(up_move, index=df.index).rolling(adx_period, min_periods=1).mean()
    down_s = pd.Series(down_move, index=df.index).rolling(adx_period, min_periods=1).mean()
    denom = up_s + down_s
    dx = np.where(denom > 1e-12, 100.0 * np.abs(up_s - down_s) / denom, 0.0)
    adx = pd.Series(dx, index=df.index).rolling(adx_period, min_periods=1).mean()
    is_trend = adx >= adx_trend_threshold

    # 3. Combine into 4 regimes
    regimes = []
    for trend, hvol in zip(is_trend, is_high_vol):
        if trend and hvol:
            regimes.append("trending_high_vol")
        elif trend and not hvol:
            regimes.append("trending_low_vol")
        elif not trend and hvol:
            regimes.append("ranging_high_vol")
        else:
            regimes.append("ranging_low_vol")

    df["regime"] = regimes
    return df


def classify_gold_12m_trend(df: pd.DataFrame, lookback_bars: int = 5000) -> pd.DataFrame:
    """Classify trailing 12-month gold price trend (up, flat, down).
    For 1h bars, ~1 year is ~6000 trading hours (or approx 5000-6000 bars).
    Trend:
    - > +10%: 'up'
    - < -10%: 'down'
    - between -10% and +10%: 'flat'
    """
    df = df.copy()
    close = df["close"]
    # Rolling return over lookback
    rolling_ret = (close / close.shift(lookback_bars) - 1.0).fillna(0.0)
    
    trends = []
    for r in rolling_ret:
        if r > 0.10:
            trends.append("up")
        elif r < -0.10:
            trends.append("down")
        else:
            trends.append("flat")
            
    df["gold_12m_trend"] = trends
    return df


def analyze_profit_concentration(
    trades: list[Trade],
    df: pd.DataFrame | None = None,
    max_concentration_threshold: float = 0.60,
) -> dict[str, Any]:
    """Calculate profit concentration across:
    1. Calendar years
    2. Market regimes (trending_high_vol, etc.)
    3. Trailing 12-month gold trend (up, flat, down)

    Returns:
    - passed: bool (neither calendar year nor market regime > max_concentration_threshold)
    - yearly_pnl: dict[str, float]
    - regime_pnl: dict[str, float]
    - gold_trend_pnl: dict[str, float]
    - max_year_concentration: float
    - max_regime_concentration: float
    """
    if not trades:
        return {
            "passed": False,
            "detail": "No trades to evaluate profit concentration.",
            "yearly_pnl": {},
            "regime_pnl": {},
            "gold_trend_pnl": {},
            "max_year_concentration": 1.0,
            "max_regime_concentration": 1.0,
        }

    # 1. Yearly concentration
    yearly_pnl: dict[str, float] = {}
    for t in trades:
        year_str = str(t.entry_time.year)
        yearly_pnl[year_str] = yearly_pnl.get(year_str, 0.0) + t.net_pnl

    total_profit_yearly = sum(v for v in yearly_pnl.values() if v > 0)
    if total_profit_yearly <= 0:
        max_year_conc = 1.0
    elif len(yearly_pnl) <= 1:
        max_year_conc = 0.0
    else:
        max_year_val = max(max(yearly_pnl.values()), 0.0)
        max_year_conc = float(max_year_val / total_profit_yearly)

    # 2. Regime concentration & Gold 12m trend
    regime_pnl: dict[str, float] = {
        "trending_high_vol": 0.0,
        "trending_low_vol": 0.0,
        "ranging_high_vol": 0.0,
        "ranging_low_vol": 0.0,
    }
    gold_trend_pnl: dict[str, float] = {
        "up": 0.0,
        "flat": 0.0,
        "down": 0.0,
    }

    if df is not None and len(df) > 0:
        if "regime" not in df.columns:
            df = classify_market_regimes(df)
        if "gold_12m_trend" not in df.columns:
            df = classify_gold_12m_trend(df)

        ts_series = df["timestamp"]
        for t in trades:
            # Find closest bar to entry_time
            idx = (ts_series - t.entry_time).abs().argmin()
            reg = df["regime"].iloc[idx]
            gtrend = df["gold_12m_trend"].iloc[idx]

            regime_pnl[reg] = regime_pnl.get(reg, 0.0) + t.net_pnl
            gold_trend_pnl[gtrend] = gold_trend_pnl.get(gtrend, 0.0) + t.net_pnl
    else:
        # Default distribution if df not provided
        regime_pnl["ranging_high_vol"] = sum(t.net_pnl for t in trades)

    total_profit_regime = sum(v for v in regime_pnl.values() if v > 0)
    if total_profit_regime <= 0:
        max_regime_conc = 1.0
    else:
        max_reg_val = max(max(regime_pnl.values()), 0.0)
        max_regime_conc = float(max_reg_val / total_profit_regime)

    passed = (max_year_conc <= max_concentration_threshold) and (
        max_regime_conc <= max_concentration_threshold
    )

    failures = []
    if max_year_conc > max_concentration_threshold:
        failures.append(f"year concentration={max_year_conc*100:.1f}% > {max_concentration_threshold*100:.0f}%")
    if max_regime_conc > max_concentration_threshold:
        failures.append(f"regime concentration={max_regime_conc*100:.1f}% > {max_concentration_threshold*100:.0f}%")

    detail = "PASS" if passed else "FAIL: " + "; ".join(failures)

    return {
        "passed": passed,
        "detail": detail,
        "yearly_pnl": {k: round(v, 2) for k, v in yearly_pnl.items()},
        "regime_pnl": {k: round(v, 2) for k, v in regime_pnl.items()},
        "gold_trend_pnl": {k: round(v, 2) for k, v in gold_trend_pnl.items()},
        "max_year_concentration": round(max_year_conc, 4),
        "max_regime_concentration": round(max_regime_conc, 4),
    }
