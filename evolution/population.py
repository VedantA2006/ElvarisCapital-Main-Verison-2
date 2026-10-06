"""
evolution/population.py – Population management, tournament selection, and elitism.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List
from evolution.genome import StrategyGenome


class Population:
    """Manages an active population pool of strategy genomes."""

    def __init__(self, target_size: int = 30, elite_count: int = 4):
        self.target_size = target_size
        self.elite_count = elite_count
        self.members: List[Dict[str, Any]] = []  # list of {"genome": StrategyGenome, "fitness": FitnessScores}

    def add(self, genome: StrategyGenome, fitness: Any) -> None:
        self.members.append({"genome": genome, "fitness": fitness})

    def get_elites(self) -> List[StrategyGenome]:
        """Return the top elite genomes based on Pareto rank and composite score."""
        sorted_members = sorted(
            self.members,
            key=lambda m: (m["fitness"].pareto_rank, -m["fitness"].composite_score)
        )
        return [m["genome"] for m in sorted_members[:self.elite_count]]

    def select_parent(self, tournament_k: int = 3) -> StrategyGenome:
        """Tournament selection favoring Pareto non-dominated individuals."""
        if not self.members:
            raise ValueError("Cannot select from an empty population.")
        candidates = random.sample(self.members, min(tournament_k, len(self.members)))
        best = min(candidates, key=lambda c: (c["fitness"].pareto_rank, -c["fitness"].composite_score))
        return best["genome"]

    def cull(self) -> None:
        """Trim population to target_size retaining top performers."""
        if len(self.members) > self.target_size:
            self.members = sorted(
                self.members,
                key=lambda m: (m["fitness"].pareto_rank, -m["fitness"].composite_score)
            )[:self.target_size]
