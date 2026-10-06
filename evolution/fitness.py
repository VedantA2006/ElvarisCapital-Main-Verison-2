"""
evolution/fitness.py – Multi-Objective Pareto Fitness and Complexity Penalty Engine.

Guarantees (Sections 35, 44, 66):
1. Never optimizes net profit alone.
2. Evaluates Sharpe, Sortino, PF, Max Drawdown, Trade count sufficiency, Cost Resilience.
3. Incorporates structural complexity penalties and non-dominated Pareto ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List
from evolution.genome import StrategyGenome


@dataclass
class FitnessScores:
    profit_factor: float
    sharpe: float
    sortino: float
    max_drawdown: float
    trade_count: int
    cost_resilience: float
    complexity_penalty: float
    composite_score: float
    pareto_rank: int = 1


class MultiObjectiveFitness:
    """Computes multi-dimensional quantitative fitness and Pareto dominance."""

    def evaluate(self, genome: StrategyGenome, backtest_metrics: Dict[str, Any]) -> FitnessScores:
        """Compute composite fitness score and vector."""
        pf = max(0.0, float(backtest_metrics.get("profit_factor", 0.0)))
        sharpe = float(backtest_metrics.get("sharpe", -2.0))
        sortino = float(backtest_metrics.get("sortino", -2.0))
        max_dd = max(0.001, float(backtest_metrics.get("max_drawdown", 1.0)))
        trades = int(backtest_metrics.get("total_trades", 0))
        cgr = float(backtest_metrics.get("cost_gross_ratio", 1.0))

        # Sample sufficiency factor (smooth sigmoid around 80 trades)
        sample_factor = 1.0 / (1.0 + math.exp(-0.05 * (trades - 80)))

        # Cost resilience: higher is better (costs < 40% of gross profits)
        cost_resilience = max(0.0, 1.0 - cgr)

        # Complexity penalty (Section 35)
        # Each condition / indicator beyond a minimal baseline adds a small discount
        comp = genome.complexity
        complexity_discount = max(0.60, 1.0 - (comp * 0.015))

        # Composite score
        # Weighted combination of risk-adjusted return, downside protection, and sample sufficiency
        risk_adj = (max(0.0, sharpe + 1.0) * 0.35) + (min(3.0, pf) * 0.35)
        dd_factor = max(0.1, 1.0 - min(1.0, max_dd * 2.0)) * 0.15
        cost_factor = cost_resilience * 0.15

        raw_score = (risk_adj + dd_factor + cost_factor) * sample_factor * complexity_discount
        composite = round(max(0.0, raw_score * 100.0), 2)

        return FitnessScores(
            profit_factor=round(pf, 3),
            sharpe=round(sharpe, 3),
            sortino=round(sortino, 3),
            max_drawdown=round(max_dd, 4),
            trade_count=trades,
            cost_resilience=round(cost_resilience, 3),
            complexity_penalty=round(1.0 - complexity_discount, 3),
            composite_score=composite,
        )

    def assign_pareto_ranks(self, candidates: List[Dict[str, Any]]) -> None:
        """Assign Non-Dominated Pareto ranks to a population of evaluated candidates."""
        # Candidate i dominates candidate j if i is >= in all objectives and > in at least one
        for i, cand_i in enumerate(candidates):
            f_i = cand_i["fitness"]
            domination_count = 0
            for j, cand_j in enumerate(candidates):
                if i == j:
                    continue
                f_j = cand_j["fitness"]
                # Objectives to maximize: sharpe, pf, cost_resilience. Objective to minimize: max_drawdown
                better_or_equal = (
                    f_j.sharpe >= f_i.sharpe and
                    f_j.profit_factor >= f_i.profit_factor and
                    f_j.max_drawdown <= f_i.max_drawdown and
                    f_j.cost_resilience >= f_i.cost_resilience
                )
                strictly_better = (
                    f_j.sharpe > f_i.sharpe or
                    f_j.profit_factor > f_i.profit_factor or
                    f_j.max_drawdown < f_i.max_drawdown
                )
            f_i.pareto_rank = domination_count + 1


def evaluate_candidate_fitness(genome: StrategyGenome, backtest_metrics: Dict[str, Any]) -> FitnessScores:
    """Convenience helper to compute multi-objective fitness scores."""
    return MultiObjectiveFitness().evaluate(genome, backtest_metrics)


def pareto_rank(candidates: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute non-dominated Pareto ranks for a list of candidate dictionaries.

    Candidate j dominates candidate i if j is >= in all reward metrics,
    <= in risk metrics (max_drawdown), and strictly better in at least one.
    """
    ranks: Dict[str, int] = {}
    n = len(candidates)

    for i in range(n):
        cand_i = candidates[i]
        c_id = cand_i.get("id", str(i))
        sh_i = float(cand_i.get("sharpe", 0.0))
        pf_i = float(cand_i.get("profit_factor", 0.0))
        dd_i = float(cand_i.get("max_drawdown", 1.0))

        domination_count = 0
        for j in range(n):
            if i == j:
                continue
            cand_j = candidates[j]
            sh_j = float(cand_j.get("sharpe", 0.0))
            pf_j = float(cand_j.get("profit_factor", 0.0))
            dd_j = float(cand_j.get("max_drawdown", 1.0))

            better_or_equal = (sh_j >= sh_i and pf_j >= pf_i and dd_j <= dd_i)
            strictly_better = (sh_j > sh_i or pf_j > pf_i or dd_j < dd_i)

            if better_or_equal and strictly_better:
                domination_count += 1

        ranks[c_id] = domination_count + 1

    return ranks

