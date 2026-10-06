"""
evolution/crossover.py – Recombination (crossover) of two parent strategy genomes.

Guarantees (Section 42):
1. Combines entry rules, filters, and exit regimes from two parent genomes.
2. Clamps total parameters to maximum 6 to maintain complexity limits.
"""

from __future__ import annotations

import copy
import random
from typing import Union
from evolution.genome import StrategyGenome, Genome, ParameterGene
from strategy.dsl.schema import StrategyDSL, EntryRule, IndicatorDef


class GenomeCrossover:
    """Recombines two parent strategy genomes into a valid child genome."""

    def crossover(
        self,
        parent_a: Union[StrategyGenome, Genome],
        parent_b: Union[StrategyGenome, Genome],
    ) -> Union[StrategyGenome, Genome]:
        """Produce a child genome by recombining parent A and parent B."""
        if isinstance(parent_a, Genome) and isinstance(parent_b, Genome):
            child_id = f"{parent_a.genome_id}_x_{parent_b.genome_id}"
            child_params = {}
            # Inherit parameters randomly from either parent
            all_keys = set(parent_a.parameters.keys()).union(parent_b.parameters.keys())
            for k in all_keys:
                if k in parent_a.parameters and k in parent_b.parameters:
                    child_params[k] = copy.deepcopy(
                        parent_a.parameters[k] if random.random() < 0.5 else parent_b.parameters[k]
                    )
                elif k in parent_a.parameters:
                    child_params[k] = copy.deepcopy(parent_a.parameters[k])
                else:
                    child_params[k] = copy.deepcopy(parent_b.parameters[k])

            child_genes = list(parent_a.genes) + [g for g in parent_b.genes if g not in parent_a.genes]

            return Genome(
                genome_id=child_id,
                family=parent_a.family,
                timeframe=parent_a.timeframe,
                parameters=child_params,
                genes=child_genes,
                generation=max(parent_a.generation, parent_b.generation) + 1,
                parent_ids=[parent_a.genome_id, parent_b.genome_id],
            )

        # StrategyGenome
        child_name = f"{parent_a.id}_x_{parent_b.id}"

        # Combine indicators (deduplicated by id)
        ind_map = {ind.id: copy.deepcopy(ind) for ind in parent_a.dsl.indicators}
        for ind in parent_b.dsl.indicators:
            if ind.id not in ind_map and len(ind_map) < 5:
                ind_map[ind.id] = copy.deepcopy(ind)
        child_indicators = list(ind_map.values())

        # Combine entry rules
        rules_a = {r.direction: copy.deepcopy(r) for r in parent_a.dsl.entry_rules}
        rules_b = {r.direction: copy.deepcopy(r) for r in parent_b.dsl.entry_rules}

        child_rules = []
        if "LONG" in rules_a and random.random() < 0.5:
            child_rules.append(rules_a["LONG"])
        elif "LONG" in rules_b:
            child_rules.append(rules_b["LONG"])
        elif "LONG" in rules_a:
            child_rules.append(rules_a["LONG"])

        if "SHORT" in rules_b and random.random() < 0.5:
            child_rules.append(rules_b["SHORT"])
        elif "SHORT" in rules_a:
            child_rules.append(rules_a["SHORT"])
        elif "SHORT" in rules_b:
            child_rules.append(rules_b["SHORT"])

        # Inherit exit from the parent
        child_exit = copy.deepcopy(parent_a.dsl.exit if random.random() < 0.5 else parent_b.dsl.exit)

        # Merge parameters, capped at 6
        param_map = copy.deepcopy(parent_a.dsl.parameters)
        for k, v in parent_b.dsl.parameters.items():
            if k not in param_map and len(param_map) < 6:
                param_map[k] = copy.deepcopy(v)

        child_dsl = StrategyDSL(
            name=child_name,
            market="XAUUSD",
            family=parent_a.dsl.family if random.random() < 0.5 else parent_b.dsl.family,
            hypothesis=f"Hybrid crossover edge: [{parent_a.dsl.name}] + [{parent_b.dsl.name}]",
            timeframes=copy.deepcopy(parent_a.dsl.timeframes),
            indicators=child_indicators,
            entry_rules=child_rules,
            exit=child_exit,
            risk=copy.deepcopy(parent_a.dsl.risk),
            parameters=param_map,
        )

        return StrategyGenome(
            dsl=child_dsl,
            generation=max(parent_a.generation, parent_b.generation) + 1,
            parent_ids=[parent_a.id, parent_b.id],
            mutation_history=[f"crossover({parent_a.id}, {parent_b.id})"],
        )


def crossover_genomes(
    parent_a: Union[StrategyGenome, Genome],
    parent_b: Union[StrategyGenome, Genome],
) -> Union[StrategyGenome, Genome]:
    """Convenience helper to recombine two parent genomes."""
    return GenomeCrossover().crossover(parent_a, parent_b)
