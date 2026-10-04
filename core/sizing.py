"""
core/sizing.py – Position sizing and margin management (Phase F2).

Implements strict risk budgeting, lot quantization, margin utilization limits,
and stop-out thresholds as specified in Section F2.3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class PositionSizeResult:
    """Outcome of position sizing calculation."""
    lots: float
    risk_usd: float
    required_margin: float
    skipped: bool
    skip_reason: str = ""  # "skipped_min_lot", "skipped_margin", "invalid_input"


def calculate_position_size(
    equity: float,
    sl_distance: float,
    price: float,
    spread: float = 0.25,
    expected_slippage: float = 0.0,
    risk_fraction: float = 0.01,
    contract_size: float = 100.0,
    lot_step: float = 0.01,
    min_lot: float = 0.01,
    max_lot: float = 50.0,
    leverage: float = 100.0,
    max_margin_utilisation: float = 0.80,
    min_lot_budget_tolerance: float = 1.50,
) -> PositionSizeResult:
    """Calculate position size in lots based on risk and margin constraints.

    Formula:
      risk_budget = risk_fraction * equity
      cost_per_oz = sl_distance + spread + expected_slippage
      risk_per_lot = cost_per_oz * contract_size
      raw_lots = risk_budget / risk_per_lot
      lots = floor(raw_lots / lot_step) * lot_step

    Rules:
      1. Round DOWN to nearest lot_step.
      2. If lots < min_lot:
         - If risk at min_lot <= min_lot_budget_tolerance * risk_budget:
             lots = min_lot
         - Else:
             skip trade, reason = 'skipped_min_lot'
      3. Apply max_lot cap: lots = min(lots, max_lot)
      4. Margin check:
         required_margin = lots * contract_size * price / leverage
         if required_margin > max_margin_utilisation * equity:
             skip trade, reason = 'skipped_margin'
    """
    if equity <= 0 or sl_distance <= 0 or price <= 0:
        return PositionSizeResult(
            lots=0.0,
            risk_usd=0.0,
            required_margin=0.0,
            skipped=True,
            skip_reason="invalid_input",
        )

    risk_budget = risk_fraction * equity
    effective_sl_per_oz = sl_distance + spread + expected_slippage
    risk_per_lot = effective_sl_per_oz * contract_size

    if risk_per_lot <= 0:
        return PositionSizeResult(
            lots=0.0,
            risk_usd=0.0,
            required_margin=0.0,
            skipped=True,
            skip_reason="invalid_input",
        )

    raw_lots = risk_budget / risk_per_lot
    
    # Step precision (e.g. lot_step=0.01 -> round down to 2 decimal places)
    steps = math.floor(raw_lots / lot_step + 1e-12)
    lots = steps * lot_step
    # Quantize to avoid floating-point representation artifacts (e.g. 0.010000000000000002)
    step_decimals = max(0, int(round(-math.log10(lot_step)))) if lot_step < 1 else 0
    lots = round(lots, step_decimals)

    if lots < min_lot:
        min_lot_risk = min_lot * risk_per_lot
        if min_lot_risk <= min_lot_budget_tolerance * risk_budget + 1e-9:
            lots = min_lot
        else:
            return PositionSizeResult(
                lots=0.0,
                risk_usd=0.0,
                required_margin=0.0,
                skipped=True,
                skip_reason="skipped_min_lot",
            )

    lots = min(lots, max_lot)
    lots = round(lots, step_decimals)

    # Margin requirement
    required_margin = (lots * contract_size * price) / leverage

    if required_margin > (max_margin_utilisation * equity + 1e-9):
        return PositionSizeResult(
            lots=0.0,
            risk_usd=0.0,
            required_margin=required_margin,
            skipped=True,
            skip_reason="skipped_margin",
        )

    actual_risk = lots * risk_per_lot
    return PositionSizeResult(
        lots=lots,
        risk_usd=actual_risk,
        required_margin=required_margin,
        skipped=False,
        skip_reason="",
    )


def check_margin_call(
    equity: float,
    required_margin: float,
    stop_out_level: float = 0.50,
) -> bool:
    """Return True if equity falls to or below stop_out_level of required margin.

    Example: stop_out_level = 0.50 means margin call if margin level <= 50%.
    """
    if required_margin <= 0:
        return False
    margin_level = equity / required_margin
    return margin_level <= (stop_out_level + 1e-9)
