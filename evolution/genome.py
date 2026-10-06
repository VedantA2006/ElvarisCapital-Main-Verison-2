"""
evolution/genome.py – Genetic representation of quantitative strategies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from strategy.dsl.schema import StrategyDSL, IndicatorDef, ParameterDef, ExitRule, EntryRule, TimeframeConfig


@dataclass
class ParameterGene:
    name: str
    value: float
    min_val: float = 0.0
    max_val: float = 100.0
    step: float = 1.0


@dataclass
class Gene:
    gene_type: str
    name: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Genome:
    """Alternative modular gene representation used in GA search."""
    genome_id: str
    family: str
    timeframe: str = "1h"
    parameters: Dict[str, ParameterGene] = field(default_factory=dict)
    genes: List[Gene] = field(default_factory=list)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)


@dataclass
class StrategyGenome:
    """Genotype wrapper around StrategyDSL supporting genetic manipulation."""
    dsl: StrategyDSL
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.dsl.name

    @property
    def complexity(self) -> int:
        """Compute structural complexity penalty score (Section 35)."""
        indicator_count = len(self.dsl.indicators)
        parameter_count = len(self.dsl.parameters)
        condition_count = sum(
            len(r.all_conditions) + len(r.any_conditions)
            for r in self.dsl.entry_rules
        )
        return indicator_count * 2 + parameter_count + condition_count * 3

    def clone(self, new_name: str) -> StrategyGenome:
        """Deep copy genome with a new strategy identifier."""
        new_dsl_dict = copy.deepcopy(self.dsl.model_dump())
        new_dsl_dict["name"] = new_name
        new_dsl = StrategyDSL(**new_dsl_dict)
        return StrategyGenome(
            dsl=new_dsl,
            generation=self.generation + 1,
            parent_ids=[self.id],
            mutation_history=list(self.mutation_history),
        )
