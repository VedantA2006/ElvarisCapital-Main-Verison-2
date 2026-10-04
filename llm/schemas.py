"""
llm/schemas.py – Pydantic contract and validation schemas for LLM strategy generation.

Enforces:
- Compulsory economic hypothesis (why the edge exists and who is on the other side).
- Hard cap of at most 6 tunable strategy parameters (SBX-5).
- Parameter bounds sanity (min <= default <= max).
- Valid Python code defining class Strategy with on_bar.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


ConceptFamilyType = Literal[
    "trend",
    "mean_reversion",
    "breakout",
    "volatility",
    "session",
    "time_effect",
    "market_structure",
    "multi_timeframe",
    "intermarket",
    "event",
    "regime",
    "other",
]


class ParameterDef(BaseModel):
    """Specification of a single tunable strategy parameter."""

    default: float
    min: float
    max: float
    step: float | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> ParameterDef:
        if self.min > self.max:
            raise ValueError(f"Parameter min ({self.min}) cannot exceed max ({self.max})")
        if not (self.min <= self.default <= self.max):
            raise ValueError(
                f"Parameter default ({self.default}) must lie within [{self.min}, {self.max}]"
            )
        return self


class StrategySpec(BaseModel):
    """Structured specification of a strategy ideated by the LLM."""

    name: str = Field(..., min_length=3, max_length=60)
    timeframe: Literal["1h", "4h"]
    hypothesis: str = Field(
        ...,
        min_length=15,
        description="Economic rationale: why the edge exists and who is on the other side of the trade.",
    )
    concept_family: ConceptFamilyType
    indicators_used: list[str] = Field(default_factory=list)
    entry_logic: str = Field(..., min_length=5)
    exit_logic: str = Field(..., min_length=5)
    filters: list[str] = Field(default_factory=list)
    session_filter: str | None = None
    direction: Literal["long_only", "short_only", "both"] = "both"
    parameters: dict[str, ParameterDef] = Field(default_factory=dict)
    expected_trades_per_year: int = Field(default=50, ge=5)
    expected_failure_conditions: str = Field(default="")

    @field_validator("hypothesis")
    @classmethod
    def validate_hypothesis_non_trivial(cls, v: str) -> str:
        s = v.strip()
        if len(s) < 15:
            raise ValueError("Hypothesis is too short. State who is on the other side and why the edge exists.")
        trivial_phrases = ["make money", "buy low sell high", "profit", "win trades"]
        if s.lower() in trivial_phrases:
            raise ValueError("Hypothesis must state a concrete structural or behavioural market phenomenon.")
        return s

    @field_validator("parameters")
    @classmethod
    def validate_parameter_cap(cls, v: dict[str, ParameterDef]) -> dict[str, ParameterDef]:
        if len(v) > 6:
            raise ValueError(f"Exceeded hard cap of 6 parameters: received {len(v)} parameters.")
        return v


class StrategyResponse(BaseModel):
    """Complete strategy payload returned by the LLM (spec + executable code)."""

    spec: StrategySpec
    code: str = Field(..., min_length=20)

    @field_validator("code")
    @classmethod
    def validate_code_structure(cls, v: str) -> str:
        if "class Strategy" not in v:
            raise ValueError("Strategy code must define a class named 'Strategy'.")
        if "def on_bar" not in v:
            raise ValueError("Strategy class must define an 'on_bar(self, bars)' method.")
        return v


class StrategyImproveResponse(BaseModel):
    """Payload returned by the LLM during structural strategy refinement."""

    spec: StrategySpec
    code: str = Field(..., min_length=20)
    refinement_reason: str = Field(..., min_length=10)

    @field_validator("code")
    @classmethod
    def validate_code_structure(cls, v: str) -> str:
        if "class Strategy" not in v:
            raise ValueError("Strategy code must define a class named 'Strategy'.")
        if "def on_bar" not in v:
            raise ValueError("Strategy class must define an 'on_bar(self, bars)' method.")
        return v
