"""
evolution/mutation.py – Genetic mutation operators preserving StrategyDSL validity.

Guarantees (Section 41):
1. All mutations produce syntactically and semantically valid StrategyDSL specifications.
2. Supports parameter perturbation, indicator swaps, threshold tweaks, and rule edits.
"""

from __future__ import annotations

import copy
import random
from typing import List, Union
from evolution.genome import StrategyGenome, Genome, ParameterGene
from strategy.dsl.schema import ConditionDef, IndicatorDef


class GenomeMutator:
    """Applies valid genetic mutations to a strategy genome."""

    INDICATOR_SWAPS = {
        "ema": ["sma", "wma"],
        "sma": ["ema", "wma"],
        "wma": ["ema", "sma"],
        "rsi": ["roc", "zscore"],
        "roc": ["rsi"],
        "bollinger": ["keltner", "donchian"],
        "keltner": ["bollinger", "donchian"],
        "donchian": ["keltner", "bollinger"],
    }

    def mutate(self, genome: Union[StrategyGenome, Genome], mutation_rate: float = 0.3) -> Union[StrategyGenome, Genome]:
        """Create a mutated clone of the genome."""
        if isinstance(genome, Genome):
            # Mutate modular Genome
            new_id = f"{genome.genome_id}_m{genome.generation + 1}"
            new_params = copy.deepcopy(genome.parameters)
            for p_name, p_gene in new_params.items():
                if random.random() < mutation_rate:
                    step = p_gene.step if p_gene.step > 0 else 1.0
                    delta = random.choice([-step, step, -step * 2, step * 2])
                    val = p_gene.value + delta
                    p_gene.value = max(p_gene.min_val, min(p_gene.max_val, val))

            return Genome(
                genome_id=new_id,
                family=genome.family,
                timeframe=genome.timeframe,
                parameters=new_params,
                genes=copy.deepcopy(genome.genes),
                generation=genome.generation + 1,
                parent_ids=[genome.genome_id],
            )

        # StrategyGenome
        child_name = f"{genome.id}_m{genome.generation + 1}"
        child = genome.clone(new_name=child_name)
        dsl = child.dsl

        mutations_applied = []

        # 1. Mutate parameters (period / threshold perturbation)
        if dsl.parameters and random.random() < mutation_rate:
            p_key = random.choice(list(dsl.parameters.keys()))
            param = dsl.parameters[p_key]
            delta = random.choice([-0.20, -0.10, 0.10, 0.20])
            new_val = param.default * (1.0 + delta)
            if isinstance(param.default, int):
                new_val = max(int(param.min), min(int(param.max), round(new_val)))
            else:
                new_val = max(float(param.min), min(float(param.max), round(new_val, 2)))
            param.default = new_val
            mutations_applied.append(f"param_perturb({p_key}->{new_val})")

        # 2. Mutate exit stop loss / take profit
        if random.random() < mutation_rate:
            choice = random.choice(["sl", "tp"])
            if choice == "sl":
                mult = round(max(0.8, min(4.0, dsl.exit.stop_loss_multiplier + random.choice([-0.2, 0.2]))), 2)
                dsl.exit.stop_loss_multiplier = mult
                mutations_applied.append(f"sl_mult({mult})")
            elif choice == "tp" and dsl.exit.take_profit_ratio:
                ratio = round(max(1.0, min(5.0, dsl.exit.take_profit_ratio + random.choice([-0.3, 0.3]))), 2)
                dsl.exit.take_profit_ratio = ratio
                mutations_applied.append(f"tp_ratio({ratio})")

        # 3. Indicator swap
        if dsl.indicators and random.random() < 0.2:
            ind = random.choice(dsl.indicators)
            if ind.type in self.INDICATOR_SWAPS:
                new_type = random.choice(self.INDICATOR_SWAPS[ind.type])
                ind.type = new_type
                mutations_applied.append(f"swap_ind({ind.id}:{new_type})")

        # 4. Session filter modification
        if dsl.entry_rules and random.random() < 0.15:
            rule = random.choice(dsl.entry_rules)
            new_session = random.choice(["all", "london", "ny", "london_ny"])
            rule.session_filter = new_session
            mutations_applied.append(f"session_filter({new_session})")

        if not mutations_applied:
            dsl.exit.stop_loss_multiplier = round(max(1.0, dsl.exit.stop_loss_multiplier * 1.1), 2)
            mutations_applied.append("sl_nudge")

        child.mutation_history.extend(mutations_applied)
        return child


def mutate_genome(genome: Union[StrategyGenome, Genome], mutation_rate: float = 0.3) -> Union[StrategyGenome, Genome]:
    """Convenience helper to apply mutation."""
    return GenomeMutator().mutate(genome, mutation_rate=mutation_rate)
