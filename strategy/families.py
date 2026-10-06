"""
strategy/families.py – Strategy Family Taxonomy and Adaptive Research Budget Allocation.

Guarantees (Sections 21, 22):
1. Comprehensive 13-family taxonomy.
2. Dynamic budget allocation preventing repetitive exploration of single concepts.
3. Adaptive exploration with minimum exploration floors (prevents winner-take-all starvation).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class StrategyFamily(str, Enum):
    TREND = "TREND"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    BREAKOUT = "BREAKOUT"
    VOLATILITY = "VOLATILITY"
    MARKET_STRUCTURE = "MARKET_STRUCTURE"
    LIQUIDITY = "LIQUIDITY"
    PRICE_ACTION = "PRICE_ACTION"
    SESSION = "SESSION"
    MULTI_TIMEFRAME = "MULTI_TIMEFRAME"
    REGIME_SWITCHING = "REGIME_SWITCHING"
    STATISTICAL = "STATISTICAL"
    HYBRID = "HYBRID"


DEFAULT_FAMILY_WEIGHTS: Dict[StrategyFamily, float] = {
    StrategyFamily.LIQUIDITY: 0.20,
    StrategyFamily.MARKET_STRUCTURE: 0.15,
    StrategyFamily.TREND: 0.15,
    StrategyFamily.VOLATILITY: 0.15,
    StrategyFamily.BREAKOUT: 0.10,
    StrategyFamily.SESSION: 0.10,
    StrategyFamily.MEAN_REVERSION: 0.05,
    StrategyFamily.MULTI_TIMEFRAME: 0.05,
    StrategyFamily.PRICE_ACTION: 0.02,
    StrategyFamily.MOMENTUM: 0.01,
    StrategyFamily.REGIME_SWITCHING: 0.01,
    StrategyFamily.STATISTICAL: 0.005,
    StrategyFamily.HYBRID: 0.005,
}


@dataclass
class FamilyResearchTracker:
    total_cycles: int = 0
    cycles_by_family: Dict[StrategyFamily, int] = field(
        default_factory=lambda: {f: 0 for f in StrategyFamily}
    )
    successes_by_family: Dict[StrategyFamily, int] = field(
        default_factory=lambda: {f: 0 for f in StrategyFamily}
    )
    weights: Dict[StrategyFamily, float] = field(
        default_factory=lambda: dict(DEFAULT_FAMILY_WEIGHTS)
    )

    def record_attempt(self, family: StrategyFamily, succeeded: bool = False) -> None:
        """Record a research cycle attempt and update success tracking."""
        self.total_cycles += 1
        self.cycles_by_family[family] = self.cycles_by_family.get(family, 0) + 1
        if succeeded:
            self.successes_by_family[family] = self.successes_by_family.get(family, 0) + 1

    def choose_next_family(self) -> StrategyFamily:
        """Select next research family using adaptive diversity weighting with floors."""
        families = list(StrategyFamily)
        adj_weights = []

        for f in families:
            base_w = self.weights.get(f, 0.05)
            attempts = self.cycles_by_family.get(f, 0)
            succ = self.successes_by_family.get(f, 0)

            # Compute empirical success rate (with laplace smoothing)
            win_rate = (succ + 1) / (attempts + 5)
            # Penalize over-explored families relative to target
            target_share = base_w
            actual_share = (attempts / max(1, self.total_cycles))

            # Adaptive dynamic adjustment: penalize oversaturated families
            exploration_boost = max(0.2, 1.0 + (target_share - actual_share) * 2.0)
            final_w = max(0.02, base_w * (0.5 + win_rate) * exploration_boost)
            adj_weights.append(final_w)

        # Normalize weights
        total = sum(adj_weights)
        norm_weights = [w / total for w in adj_weights]

        return random.choices(families, weights=norm_weights, k=1)[0]

    def get_target_allocations(self) -> Dict[StrategyFamily, float]:
        """Return target research allocations across all families."""
        return dict(self.weights)

    def record_experiment(self, family: StrategyFamily, passed: bool = False) -> None:
        """Alias for record_attempt."""
        self.record_attempt(family, succeeded=passed)

    def select_next_family(self) -> StrategyFamily:
        """Alias for choose_next_family."""
        return self.choose_next_family()


DiversityManager = FamilyResearchTracker

