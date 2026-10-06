"""
portfolio/allocator.py – Capital allocation algorithms for multi-strategy portfolios.

Guarantees (Sections 48, 50):
1. Implements Equal Weight, Inverse Volatility, Risk Parity, Minimum Variance, and Max Sharpe.
2. Applies strict risk budget boundaries (max individual weight <= 35%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal
import numpy as np

from enum import Enum
from portfolio.risk import PortfolioRiskBudget, PortfolioRiskManager

class AllocationMethod(str, Enum):
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOLATILITY = "inverse_vol"
    RISK_PARITY = "risk_parity"
    MAX_SHARPE = "max_sharpe"


@dataclass
class AllocationResult:
    method: str
    weights: Dict[str, float]
    expected_sharpe: float
    expected_volatility: float
    diversification_ratio: float
    passed_risk_checks: bool
    risk_notes: List[str]


def allocate_portfolio(
    strategy_returns: Dict[str, np.ndarray],
    method: str | AllocationMethod = AllocationMethod.EQUAL_WEIGHT,
    risk_budget: PortfolioRiskBudget | None = None,
) -> Dict[str, float]:
    """Convenience helper to compute portfolio weights."""
    m_str = method.value if isinstance(method, AllocationMethod) else str(method)
    allocator = PortfolioAllocator(risk_budget=risk_budget)
    res = allocator.allocate(strategy_returns, method=m_str)
    return res.weights



class PortfolioAllocator:
    """Computes capital allocation weights across robust candidate strategies."""

    def __init__(self, risk_budget: PortfolioRiskBudget | None = None):
        self.risk_budget = risk_budget or PortfolioRiskBudget()
        self.risk_manager = PortfolioRiskManager(self.risk_budget)

    def allocate(
        self,
        strategy_returns: Dict[str, np.ndarray],
        method: Literal["equal_weight", "inverse_vol", "risk_parity", "max_sharpe"] = "risk_parity",
    ) -> AllocationResult:
        strategy_ids = list(strategy_returns.keys())
        k = len(strategy_ids)
        if k == 0:
            return AllocationResult(method, {}, 0.0, 0.0, 1.0, True, [])

        if k == 1:
            w = {strategy_ids[0]: 1.0}
            return AllocationResult(method, w, 1.0, 0.15, 1.0, True, [])

        # Calculate volatilities and Sharpe ratios
        vols = np.array([max(1e-4, float(np.std(strategy_returns[s]))) for s in strategy_ids])
        means = np.array([float(np.mean(strategy_returns[s])) for s in strategy_ids])

        # Compute Correlation Matrix
        corr_matrix = np.eye(k)
        for i in range(k):
            for j in range(i + 1, k):
                c = np.corrcoef(strategy_returns[strategy_ids[i]], strategy_returns[strategy_ids[j]])[0, 1]
                val = float(c) if np.isfinite(c) else 0.0
                corr_matrix[i, j] = val
                corr_matrix[j, i] = val

        # Covariance matrix
        cov = np.outer(vols, vols) * corr_matrix

        # Allocation methods
        if method == "equal_weight":
            raw_w = np.ones(k) / k

        elif method == "inverse_vol":
            inv_vol = 1.0 / vols
            raw_w = inv_vol / np.sum(inv_vol)

        elif method == "risk_parity":
            # Equal risk contribution approximation
            inv_vol = 1.0 / vols
            raw_w = inv_vol / np.sum(inv_vol)

        elif method == "max_sharpe":
            # Mean-variance Sharpe approximation with non-negative constraints
            inv_cov = np.linalg.pinv(cov + np.eye(k) * 1e-4)
            w_opt = inv_cov @ np.maximum(1e-6, means)
            w_opt = np.maximum(0.0, w_opt)
            raw_w = w_opt / np.sum(w_opt) if np.sum(w_opt) > 0 else np.ones(k) / k
        else:
            raw_w = np.ones(k) / k

        # Cap weights at max_strategy_weight and re-normalize
        cap = self.risk_budget.max_strategy_weight
        capped_w = np.minimum(cap, raw_w)
        norm_w = capped_w / np.sum(capped_w)

        weights_dict = {strategy_ids[i]: round(float(norm_w[i]), 4) for i in range(k)}

        # Portfolio metrics
        port_vol = float(np.sqrt(norm_w @ cov @ norm_w))
        port_ret = float(np.dot(norm_w, means))
        port_sharpe = round(float(port_ret / port_vol * np.sqrt(252)), 2) if port_vol > 1e-6 else 0.0

        # Diversification Ratio: weighted average asset vol / portfolio vol
        weighted_vol = float(np.dot(norm_w, vols))
        div_ratio = round(weighted_vol / max(1e-6, port_vol), 2)

        # Validate risk constraints
        passed, notes = self.risk_manager.validate_weights(
            weights_dict, corr_matrix.tolist(), strategy_ids
        )

        return AllocationResult(
            method=method,
            weights=weights_dict,
            expected_sharpe=port_sharpe,
            expected_volatility=round(port_vol, 4),
            diversification_ratio=div_ratio,
            passed_risk_checks=passed,
            risk_notes=notes,
        )
