"""
llm/research_memory.py – Persistent Structured Research Memory for Autonomous R&D.

Guarantees (Sections 19, 46, 68):
1. Stores hypotheses, experiments, failure autopsies, and strategy lineages.
2. Derives knowledge conclusions strictly from recorded historical experiments.
3. Provides contextual memory retrieval for the Research Director without data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from storage import mongo

_log = logging.getLogger("quantforge.llm.memory")


@dataclass
class ResearchHypothesis:
    hypothesis_id: str
    family: str
    timeframe: str
    hypothesis_text: str
    market_rationale: str
    created_at: str
    status: str = "active"  # active, validated, abandoned
    tested_strategies: List[str] = field(default_factory=list)
    best_sharpe: float = 0.0
    best_pf: float = 0.0


@dataclass
class LineageNode:
    strategy_id: str
    parent_id: Optional[str]
    root_id: str
    generation: int
    family: str
    timeframe: str
    mutation_type: str
    status: str  # candidate, rejected, improved, abandoned
    metrics: Dict[str, Any] = field(default_factory=dict)
    primary_failure: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ResearchMemory:
    """Manages empirical memory of hypotheses, lineages, and strategy autopsies."""

    def __init__(self, db=None):
        self._db = db if db is not None else mongo.get_db()
        self._col_hypotheses = self._db["hypotheses"]
        self._col_lineage = self._db["strategy_lineage"]
        self._col_autopsies = self._db["strategy_autopsies"]

    def record_hypothesis(self, hyp: ResearchHypothesis) -> None:
        """Store or update a research hypothesis."""
        doc = asdict(hyp)
        self._col_hypotheses.replace_one({"hypothesis_id": hyp.hypothesis_id}, doc, upsert=True)

    def record_lineage_node(self, node: LineageNode) -> None:
        """Record an evolutionary node in a strategy lineage tree."""
        doc = asdict(node)
        self._col_lineage.replace_one({"strategy_id": node.strategy_id}, doc, upsert=True)

    def record_autopsy(self, autopsy_dict: Dict[str, Any]) -> None:
        """Persist a strategy autopsy diagnostic report."""
        strategy_id = autopsy_dict.get("strategy_id", "unknown")
        autopsy_dict["recorded_at"] = datetime.now(timezone.utc).isoformat()
        self._col_autopsies.replace_one({"strategy_id": strategy_id}, autopsy_dict, upsert=True)

    def get_lineage(self, root_strategy_id: str) -> List[Dict[str, Any]]:
        """Retrieve all generations for a strategy lineage."""
        return list(self._col_lineage.find({"root_id": root_strategy_id}).sort("generation", 1))

    def get_recent_failures(self, family: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent failure causes for context-aware ideation."""
        query: Dict[str, Any] = {}
        if family:
            query["family"] = family
        return list(self._col_autopsies.find(query).sort("recorded_at", -1).limit(limit))

    def get_promising_unexhausted_lineages(self, max_depth: int = 5) -> List[Dict[str, Any]]:
        """Find promising candidate strategies eligible for further improvement."""
        # Find strategies that achieved positive expectancy or near-miss PF but haven't reached max depth
        pipeline = [
            {"$match": {"generation": {"$lt": max_depth}, "status": {"$in": ["repairable", "rejected"]}}},
            {"$sort": {"metrics.profit_factor": -1}},
            {"$limit": 10},
        ]
        return list(self._col_lineage.aggregate(pipeline))

    def summarize_empirical_learnings(self, family: str) -> str:
        """Synthesize honest research observations derived strictly from stored data (Section 68)."""
        autopsies = list(self._col_autopsies.find({"family": family}).limit(50))
        if not autopsies:
            return f"Family '{family}' has no prior evaluated experiments on this dataset."

        total = len(autopsies)
        fail_reasons: Dict[str, int] = {}
        cost_sensitive_count = 0
        low_trades_count = 0

        for a in autopsies:
            p_fail = a.get("primary_failure", "UNKNOWN")
            fail_reasons[p_fail] = fail_reasons.get(p_fail, 0) + 1
            if a.get("cost_gross_ratio", 0) > 0.40:
                cost_sensitive_count += 1
            if a.get("total_trades", 0) < 50:
                low_trades_count += 1

        top_fails = sorted(fail_reasons.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{k} ({v}/{total})" for k, v in top_fails)

        summary_lines = [
            f"Empirical memory for {family} ({total} trials recorded):",
            f"- Primary failure modes: {top_str}",
            f"- Cost-sensitive fraction: {cost_sensitive_count}/{total} ({cost_sensitive_count/total:.1%})",
            f"- Insufficient sample fraction: {low_trades_count}/{total} ({low_trades_count/total:.1%})",
        ]
        return "\n".join(summary_lines)
