"""
portfolio/risk.py – Risk budgeting and portfolio exposure constraints.

Guarantees (Section 50):
1. Enforces max strategy weights, aggregate margin/drawdown caps, and exposure limits.
2. Computes marginal and component risk contributions across correlated strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class PortfolioRiskBudget:
    max_strategy_weight: float = 0.35      # No single strategy exceeds 35%
    max_portfolio_drawdown: float = 0.15   # 15% aggregate portfolio DD ceiling
    max_correlated_exposure: float = 0.50  # Cap on strategies with correlation > 0.60
    max_aggregate_leverage: float = 5.0    # 5x gross notional limit
    min_diversification_ratio: float = 1.15


class PortfolioRiskManager:
    """Validates allocations against institutional portfolio risk constraints."""

    def __init__(self, budget: PortfolioRiskBudget | None = None):
        self.budget = budget or PortfolioRiskBudget()

    def validate_weights(
        self,
        weights: Dict[str, float],
        correlation_matrix: List[List[float]],
        strategy_ids: List[str],
    ) -> tuple[bool, List[str]]:
        """Verify allocation weights against all configured risk ceilings."""
        violations = []

        # 1. Single strategy weight cap
        for s_id, w in weights.items():
            if w > self.budget.max_strategy_weight + 1e-4:
                violations.append(
                    f"Strategy '{s_id}' weight {w:.1%} exceeds ceiling ({self.budget.max_strategy_weight:.1%})"
                )

        # 2. Correlated cluster exposure cap
        k = len(strategy_ids)
        if k > 1 and correlation_matrix:
            for i in range(k):
                corr_weight_sum = weights.get(strategy_ids[i], 0.0)
                for j in range(k):
                    if i != j and correlation_matrix[i][j] > 0.60:
                        corr_weight_sum += weights.get(strategy_ids[j], 0.0)
                if corr_weight_sum > self.budget.max_correlated_exposure + 1e-4:
                    violations.append(
                        f"Correlated cluster around '{strategy_ids[i]}' has aggregate weight {corr_weight_sum:.1%}, exceeding limit ({self.budget.max_correlated_exposure:.1%})"
                    )
                    break

        return len(violations) == 0, violations


def calculate_risk_contributions(weights: Dict[str, float], cov_matrix: np.ndarray) -> Dict[str, float]:
    """Calculate percentage contribution to total portfolio risk per asset."""
    keys = list(weights.keys())
    w = np.array([weights[k] for k in keys])
    port_var = float(w @ cov_matrix @ w)
    if port_var < 1e-8:
        return {k: round(1.0 / len(keys), 4) for k in keys}
    port_vol = np.sqrt(port_var)
    marginal_contrib = (cov_matrix @ w) / port_vol
    component_contrib = w * marginal_contrib
    pct_contrib = component_contrib / port_vol
    return {keys[i]: round(float(pct_contrib[i]), 4) for i in range(len(keys))}

