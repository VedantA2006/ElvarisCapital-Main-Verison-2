"""
portfolio – Institutional Quantitative Portfolio Construction Engine.

Includes:
- correlation.py: Multi-strategy return correlation, trade overlap, and position alignment
- risk.py: Risk budgeting, marginal risk contribution, and aggregate exposure limits
- allocator.py: Capital allocation models (Equal Weight, Inverse Vol, Risk Parity, Max Sharpe)
"""

from portfolio.correlation import StrategyCorrelationMatrix, compute_return_correlation
from portfolio.risk import PortfolioRiskBudget
from portfolio.allocator import PortfolioAllocator, AllocationResult

__all__ = [
    "StrategyCorrelationMatrix",
    "compute_return_correlation",
    "PortfolioRiskBudget",
    "PortfolioAllocator",
    "AllocationResult",
]
