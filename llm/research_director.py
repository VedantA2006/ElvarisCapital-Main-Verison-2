"""
llm/research_director.py – Autonomous Research Director Decision Engine.

Guarantees (Sections 18, 60, 91, 92, 93):
1. Governs high-level autonomous exploration policy across strategy families.
2. Decides among: NEW_HYPOTHESIS, IMPROVE_EXISTING, EXPLORE_NEW_FAMILY,
   EVOLVE_PROMISING_STRATEGY, ABANDON_LINEAGE, RUN_ROBUSTNESS, FREEZE_CANDIDATE.
3. Enforces strict session budgets (max_llm_calls, max_tokens, max_depth).
4. Strictly subordinate to deterministic safety (cannot bypass gates or unseal holdout).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from llm.failure_analyzer import FailureCategory, StrategyAutopsy
from llm.prompt_firewall import PromptFirewall
from llm.research_memory import LineageNode, ResearchHypothesis, ResearchMemory
from strategy.families import FamilyResearchTracker, StrategyFamily

_log = logging.getLogger("quantforge.llm.director")


class DirectorActionType(str, Enum):
    NEW_HYPOTHESIS = "NEW_HYPOTHESIS"
    IMPROVE_EXISTING = "IMPROVE_EXISTING"
    EXPLORE_NEW_FAMILY = "EXPLORE_NEW_FAMILY"
    RUN_ROBUSTNESS = "RUN_ROBUSTNESS"
    ABANDON_LINEAGE = "ABANDON_LINEAGE"
    FREEZE_CANDIDATE = "FREEZE_CANDIDATE"
    STOP_BUDGET_EXHAUSTED = "STOP_BUDGET_EXHAUSTED"


@dataclass
class ResearchAction:
    action_type: DirectorActionType
    family: StrategyFamily
    timeframe: str
    target_strategy_id: Optional[str] = None
    hypothesis: Optional[str] = None
    autopsy: Optional[StrategyAutopsy] = None
    reason: str = ""


@dataclass
class ResearchBudget:
    max_llm_calls: int = 100
    max_tokens: int = 500_000
    max_candidates: int = 50
    max_backtests: int = 200
    max_improvement_depth: int = 5

    # Telemetry
    used_llm_calls: int = 0
    used_tokens: int = 0
    completed_backtests: int = 0
    candidates_promoted: int = 0

    def is_exhausted(self) -> bool:
        return (
            self.used_llm_calls >= self.max_llm_calls
            or self.used_tokens >= self.max_tokens
            or self.completed_backtests >= self.max_backtests
        )


class ResearchDirector:
    """The autonomous laboratory director steering strategy research."""

    def __init__(
        self,
        cfg: dict,
        memory: Optional[ResearchMemory] = None,
        budget: Optional[ResearchBudget] = None,
        firewall: Optional[PromptFirewall] = None,
    ):
        self._cfg = cfg
        self._memory = memory or ResearchMemory()
        self._family_tracker = FamilyResearchTracker()
        self._budget = budget or ResearchBudget()
        self._firewall = firewall or PromptFirewall()
        self._active_lineage_stack: List[str] = []

    @property
    def budget(self) -> ResearchBudget:
        return self._budget

    def select_next_research(self) -> ResearchAction:
        """Evaluate deterministic state and choose the next scientific action."""
        if self._budget.is_exhausted():
            return ResearchAction(
                action_type=DirectorActionType.STOP_BUDGET_EXHAUSTED,
                family=StrategyFamily.TREND,
                timeframe="15m",
                reason="Session research budget exhausted.",
            )

        # 1. Check if an active promising lineage can be improved
        promising_nodes = self._memory.get_promising_unexhausted_lineages(
            max_depth=self._budget.max_improvement_depth
        )
        if promising_nodes:
            candidate = promising_nodes[0]
            gen = int(candidate.get("generation", 0))
            strat_id = candidate.get("strategy_id", "")
            fam_str = candidate.get("family", "TREND")
            family = StrategyFamily(fam_str) if fam_str in StrategyFamily.__members__ else StrategyFamily.TREND
            tf = candidate.get("timeframe", "15m")

            # Verify improvement limits (Section 27)
            if gen < self._budget.max_improvement_depth:
                return ResearchAction(
                    action_type=DirectorActionType.IMPROVE_EXISTING,
                    family=family,
                    timeframe=tf,
                    target_strategy_id=strat_id,
                    reason=f"Candidate {strat_id} (Gen {gen}) has promising near-miss metrics; requesting structural improvement.",
                )

        # 2. Select next research family via adaptive diversity allocation
        target_family = self._family_tracker.choose_next_family()
        timeframe = "15m" if target_family in (StrategyFamily.LIQUIDITY, StrategyFamily.PRICE_ACTION) else "1h"

        return ResearchAction(
            action_type=DirectorActionType.NEW_HYPOTHESIS,
            family=target_family,
            timeframe=timeframe,
            reason=f"Exploring under-represented family {target_family.value} to maintain research diversity.",
        )

    def record_autopsy_and_decide(
        self,
        strategy_id: str,
        autopsy: StrategyAutopsy,
        generation: int = 0,
        root_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> DirectorActionType:
        """Record backtest autopsy in memory and decide on lineage continuation."""
        fam_enum = StrategyFamily(autopsy.family) if autopsy.family in StrategyFamily.__members__ else StrategyFamily.TREND
        self._family_tracker.record_attempt(fam_enum, succeeded=autopsy.status == "PASS")
        self._budget.completed_backtests += 1

        # Store autopsy in memory
        self._memory.record_autopsy(autopsy.to_dict())

        # Record lineage node
        node = LineageNode(
            strategy_id=strategy_id,
            parent_id=parent_id,
            root_id=root_id or strategy_id,
            generation=generation,
            family=autopsy.family,
            timeframe="15m",
            mutation_type="root" if generation == 0 else "improvement",
            status="candidate" if autopsy.status == "PASS" else ("repairable" if autopsy.is_repairable else "abandoned"),
            metrics={
                "profit_factor": autopsy.profit_factor,
                "sharpe": autopsy.sharpe,
                "total_trades": autopsy.total_trades,
                "cost_gross_ratio": autopsy.cost_gross_ratio,
            },
            primary_failure=autopsy.primary_failure,
        )
        self._memory.record_lineage_node(node)

        if autopsy.status == "PASS":
            return DirectorActionType.RUN_ROBUSTNESS

        if autopsy.is_repairable and generation < self._budget.max_improvement_depth:
            return DirectorActionType.IMPROVE_EXISTING

        return DirectorActionType.ABANDON_LINEAGE
