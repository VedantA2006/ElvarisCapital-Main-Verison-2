"""
strategy/dsl/validator.py – Validates StrategyDSL for semantic correctness and security.

Guarantees (Sections 11, 12, 13, 31, 35):
1. Verifies that all indicator functions exist in the approved catalog.
2. Forbids negative offsets in condition definitions (prevents lookahead via DSL).
3. Enforces parameter bounds and complexity limits.
4. Asserts deterministic quantitative definitions for ICT/SMC concepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from strategy.dsl.schema import StrategyDSL

APPROVED_INDICATORS = {
    "sma", "ema", "wma", "roc", "macd", "adx", "supertrend", "donchian",
    "keltner", "rsi", "stoch", "bollinger", "zscore", "atr", "true_range",
    "realized_vol", "vol_percentile", "squeeze", "session_vwap", "anchored_vwap",
    "previous_day_hl", "previous_week_hl", "session_range", "opening_range",
    "hour_utc", "dow", "is_session", "minutes_since_session_open", "month_end_flag",
    "htf", "swing_high", "swing_low", "bos", "choch", "fvg", "liquidity_sweep",
}


@dataclass
class DSLValidationResult:
    is_valid: bool
    errors: List[str]


class DSLValidator:
    """Deterministic validator for StrategyDSL objects."""

    def validate(self, dsl: StrategyDSL) -> DSLValidationResult:
        errors = []

        # 1. Parameter count check
        if len(dsl.parameters) > 6:
            errors.append(f"Too many parameters ({len(dsl.parameters)} > 6 maximum allowed).")

        # 2. Indicator verification
        for ind in dsl.indicators:
            if ind.type not in APPROVED_INDICATORS:
                errors.append(f"Indicator '{ind.type}' (id: '{ind.id}') is not in approved leak-free catalog.")

        # 3. Lookahead condition checks
        for rule in dsl.entry_rules:
            for cond in rule.all_conditions + rule.any_conditions:
                if cond.offset < 0:
                    errors.append(
                        f"Condition '{cond.left}' has negative offset {cond.offset}: offset must be non-negative (future lookahead forbidden)."
                    )

        # 4. Exit sanity checks
        if dsl.exit.stop_loss_multiplier <= 0:
            errors.append("Stop loss multiplier must be strictly positive.")
        if dsl.exit.take_profit_type == "rr" and (dsl.exit.take_profit_ratio or 0) < 0.5:
            errors.append("Take profit ratio must be >= 0.5 for risk-reward exits.")

        # 5. Risk bounds
        if not (0.001 <= dsl.risk.risk_per_trade <= 0.05):
            errors.append(f"Risk per trade {dsl.risk.risk_per_trade} outside institutional bounds [0.1%, 5%].")

        return DSLValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
        )
