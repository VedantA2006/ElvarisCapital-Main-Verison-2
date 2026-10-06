"""
evolution – Evolutionary Genetic Algorithm Engine for Quantitative Strategy Search.

Includes:
- genome.py: Genome representation of StrategyDSL rules and parameters
- mutation.py: Valid semantic mutations for indicators, periods, thresholds, rules
- crossover.py: Recombination of entry rules, filters, and exit systems
- fitness.py: Multi-objective Pareto fitness and complexity penalties
- population.py: Population pool management and tournament selection
- engine.py: Tiered evolutionary search optimizer
"""

from evolution.genome import StrategyGenome
from evolution.mutation import GenomeMutator
from evolution.crossover import GenomeCrossover
from evolution.fitness import MultiObjectiveFitness, FitnessScores
from evolution.population import Population
from evolution.engine import EvolutionaryEngine

__all__ = [
    "StrategyGenome",
    "GenomeMutator",
    "GenomeCrossover",
    "MultiObjectiveFitness",
    "FitnessScores",
    "Population",
    "EvolutionaryEngine",
]
