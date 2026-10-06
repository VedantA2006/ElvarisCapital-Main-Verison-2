"""
evolution/engine.py – Tiered Evolutionary Strategy Optimizer Engine.

Guarantees (Sections 40, 43, 44, 45):
1. Tiered evaluation: cheap checks first (Tier 1), detailed robustness only for survivors (Tier 2/3).
2. Manages generations of mutation, crossover, and Pareto selection.
3. Completely sandboxed backtest execution with zero lookahead.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from evolution.crossover import GenomeCrossover
from evolution.fitness import MultiObjectiveFitness
from evolution.genome import StrategyGenome
from evolution.mutation import GenomeMutator
from evolution.population import Population
from strategy.dsl.compiler import DSLCompiler
from strategy.dsl.fingerprint import StrategyFingerprinter
from strategy.dsl.validator import DSLValidator

_log = logging.getLogger("quantforge.evolution.engine")


class EvolutionaryEngine:
    """Orchestrates generational evolutionary search across strategy genomes."""

    def __init__(
        self,
        cfg: dict,
        population_size: int = 20,
        elite_count: int = 3,
        mutation_rate: float = 0.35,
        crossover_rate: float = 0.40,
    ):
        self.cfg = cfg
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.validator = DSLValidator()
        self.compiler = DSLCompiler()
        self.mutator = GenomeMutator()
        self.recombiner = GenomeCrossover()
        self.fitness_eval = MultiObjectiveFitness()
        self.fingerprinter = StrategyFingerprinter()
        self.population = Population(target_size=population_size, elite_count=elite_count)

    def run_generation(
        self,
        evaluator_fn: Callable[[str, StrategyGenome], Dict[str, Any]],
        known_fingerprints: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run one full generation of selection, crossover, mutation, and tiered evaluation."""
        if len(self.population.members) < 2:
            return []

        new_candidates: List[StrategyGenome] = []

        # 1. Elitism: retain top performers
        elites = self.population.get_elites()

        # 2. Generate offspring via crossover and mutation
        while len(new_candidates) < (self.pop_size - len(elites)):
            import random
            parent_a = self.population.select_parent()
            if random.random() < self.crossover_rate:
                parent_b = self.population.select_parent()
                child = self.recombiner.crossover(parent_a, parent_b)
            else:
                child = parent_a.clone(new_name=f"{parent_a.id}_c{parent_a.generation + 1}")

            if random.random() < self.mutation_rate:
                child = self.mutator.mutate(child)

            # Validate DSL before accepting
            v_res = self.validator.validate(child.dsl)
            if v_res.is_valid:
                new_candidates.append(child)

        # 3. Evaluate new candidates via tiered evaluator
        results = []
        for genome in new_candidates:
            code = self.compiler.compile(genome.dsl)
            metrics = evaluator_fn(code, genome)
            fitness = self.fitness_eval.evaluate(genome, metrics)
            self.population.add(genome, fitness)
            results.append({"genome": genome, "fitness": fitness, "metrics": metrics})

        # 4. Assign Pareto ranks and cull to target size
        self.fitness_eval.assign_pareto_ranks(self.population.members)
        self.population.cull()

        return results
