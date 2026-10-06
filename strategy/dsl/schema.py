"""
strategy/dsl/schema.py – Strict Pydantic Schema for the Strategy Domain-Specific Language (DSL).

Guarantees (Sections 11, 12, 13, 62):
1. Pure deterministic schema without vague natural-language constraints.
2. Supports ICT/SMC concepts (BOS, CHOCH, FVG, liquidity sweeps) via exact quantitative rules.
3. Multi-timeframe awareness with explicit chronological alignment.
4. Parameterizable limits (up to 6 tunable parameters).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


class IndicatorDef(BaseModel):
    id: str = Field(..., description="Unique identifier for indicator instance, e.g. 'ema_fast'")
    type: str = Field(..., description="Indicator function: ema, sma, wma, rsi, macd, atr, adx, supertrend, bollinger, keltner, stoch, roc, donchian, session_range, htf, swing_high, swing_low, bos, choch, fvg, liquidity_sweep")
    timeframe: Optional[str] = Field(None, description="Optional HTF source (e.g. '1h' or '4h')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Indicator parameters, e.g. {'period': 20}")


class ConditionDef(BaseModel):
    left: str = Field(..., description="Left operand, e.g. 'close', 'ema_fast', 'rsi', 'bars.high'")
    operator: Literal[">", "<", ">=", "<=", "==", "!=", "cross_above", "cross_below", "is_true", "is_false"] = Field(...)
    right: Optional[Union[str, float, int]] = Field(None, description="Right operand, e.g. 'ema_slow', 70.0, or None if boolean flag")
    offset: int = Field(0, description="Bar offset for left operand (0=current closed bar, 1=previous closed bar)")


class EntryRule(BaseModel):
    direction: Literal["LONG", "SHORT"]
    all_conditions: List[ConditionDef] = Field(default_factory=list, description="All conditions must be met (logical AND)")
    any_conditions: List[ConditionDef] = Field(default_factory=list, description="At least one condition must be met (logical OR)")
    session_filter: Optional[Literal["all", "london", "ny", "asia", "london_ny"]] = Field("all")
    cooldown_bars: int = Field(2, ge=0, le=50, description="Minimum bars between new entries")


class ExitRule(BaseModel):
    stop_loss_type: Literal["atr", "points", "swing", "fixed"] = Field("atr")
    stop_loss_multiplier: float = Field(1.5, ge=0.1, le=20.0)
    take_profit_type: Literal["rr", "atr", "points", "trailing", "none"] = Field("rr")
    take_profit_ratio: Optional[float] = Field(2.0, ge=0.5, le=20.0, description="Risk-Reward ratio if type is 'rr'")
    take_profit_multiplier: Optional[float] = Field(3.0, ge=0.5, le=30.0, description="ATR multiplier if type is 'atr'")
    trailing_stop_atr: Optional[float] = Field(None, ge=0.5, le=10.0)
    breakeven_r: Optional[float] = Field(None, ge=0.5, le=10.0, description="Move SL to BE after reaching R profit")
    time_stop_bars: Optional[int] = Field(None, ge=1, le=500, description="Close trade after N bars held")


class RiskRule(BaseModel):
    risk_per_trade: float = Field(0.01, ge=0.001, le=0.05, description="Fraction of equity risked per trade (0.01 = 1%)")
    max_spread_usd: float = Field(1.0, ge=0.1, le=5.0, description="Max acceptable spread in USD before entry is suppressed")


class TimeframeConfig(BaseModel):
    primary: Literal["5m", "15m", "1h", "4h"] = Field("15m")
    context: Optional[Literal["15m", "1h", "4h"]] = Field("1h")


class ParameterRange(BaseModel):
    default: Union[int, float]
    min: Union[int, float]
    max: Union[int, float]
    step: Optional[Union[int, float]] = None


ParameterDef = ParameterRange


class StrategyDSL(BaseModel):
    """Complete, self-contained Strategy Specification in QuantForge DSL."""
    name: str = Field(..., description="Descriptive strategy name")
    market: Literal["XAUUSD"] = Field("XAUUSD")
    family: Literal[
        "TREND", "MOMENTUM", "MEAN_REVERSION", "BREAKOUT", "VOLATILITY",
        "MARKET_STRUCTURE", "LIQUIDITY", "PRICE_ACTION", "SESSION",
        "MULTI_TIMEFRAME", "REGIME_SWITCHING", "STATISTICAL", "HYBRID"
    ] = Field(..., description="Strategy family taxonomy category")
    hypothesis: str = Field(..., description="Economic and microstructure rationale for why the edge exists")
    timeframes: TimeframeConfig = Field(default_factory=TimeframeConfig)
    indicators: List[IndicatorDef] = Field(default_factory=list)
    entry_rules: List[EntryRule] = Field(..., description="List of entry rules for LONG and/or SHORT")
    exit: ExitRule = Field(default_factory=ExitRule)
    risk: RiskRule = Field(default_factory=RiskRule)
    parameters: Dict[str, ParameterRange] = Field(default_factory=dict, description="Tunable parameters, max 6")

    @model_validator(mode="after")
    def validate_parameter_cap(self) -> StrategyDSL:
        if len(self.parameters) > 6:
            raise ValueError(f"DSL enforces maximum 6 tunable parameters, got {len(self.parameters)}")
        return self
