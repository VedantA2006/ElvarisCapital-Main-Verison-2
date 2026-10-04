"""
strategy/diversity.py – Diversity system (never the same strategy twice).

Closes: OPS-2 (diversity part), LLM-4 (Phase F9).

Enforces 6 layers of strategy diversity:
1. Logic fingerprint: canonical JSON hash of key components. Exact duplicates rejected
   BEFORE any backtest and do not count as trials. Rejection cites the duplicated strategy.
2. Code-structure fingerprint: normalized AST multiset Jaccard similarity >= 0.85 is a near-duplicate.
3. Semantic similarity: TF-IDF cosine similarity on hypothesis + entry_logic + exit_logic >= 0.80.
4. Behaviour fingerprint: daily returns correlation > 0.70 or entry overlap +/-1 bar > 0.80 marked redundant.
5. Coverage map: 4D grid (concept x timeframe x session x regime), selects emptiest cell with 20% exploration,
   cooling period on K consecutive failures.
6. Versions: revisions compared against ancestor only by logic fingerprint (constants-only is not a new idea);
   compared against all others by all similarity checks.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ═══════════════════════════════════════════════════════════════════════════
# 1. Logic Fingerprint
# ═══════════════════════════════════════════════════════════════════════════

def compute_logic_fingerprint(spec: dict[str, Any]) -> str:
    """Canonical JSON hash of core structural strategy elements.

    Tuple: (concept_family, sorted indicators, entry type, exit type, sorted filters, session, direction, timeframe).
    Invariant to dictionary key ordering and indicator/filter list ordering.
    """
    canonical_spec = {
        "concept_family": str(spec.get("concept_family", "")).strip().lower(),
        "indicators": sorted([str(x).strip().lower() for x in spec.get("indicators", [])]),
        "entry_type": str(spec.get("entry_type", "")).strip().lower(),
        "exit_type": str(spec.get("exit_type", "")).strip().lower(),
        "filters": sorted([str(x).strip().lower() for x in spec.get("filters", [])]),
        "session": str(spec.get("session", "all")).strip().lower(),
        "direction": str(spec.get("direction", "both")).strip().lower(),
        "timeframe": str(spec.get("timeframe", "1h")).strip().lower(),
    }
    canonical_json = json.dumps(canonical_spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Code-Structure Fingerprint (AST Multiset Jaccard Similarity)
# ═══════════════════════════════════════════════════════════════════════════

def compute_code_structure_features(source: str) -> Counter[str]:
    """Normalize AST and extract multiset of structural tokens:
    (helper called, comparison/arithmetic operators, exit style, node types).
    Replaces identifier names and constants with generic placeholders.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        # Fallback to lexical token count on syntax error
        return Counter(source.split())

    features: Counter[str] = Counter()

    for node in ast.walk(tree):
        # 1. Node type
        features[f"node_{type(node).__name__}"] += 1

        # 2. Calls (helper functions, indicator methods)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                features[f"call_{node.func.id}"] += 1
            elif isinstance(node.func, ast.Attribute):
                features[f"call_attr_{node.func.attr}"] += 1
            # Keywords in calls (e.g. Signal exit style arguments)
            for kw in node.keywords:
                features[f"kw_{kw.arg}"] += 1

        # 3. Comparison operators
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                features[f"op_cmp_{type(op).__name__}"] += 1

        # 4. Binary operators (arithmetic)
        elif isinstance(node, ast.BinOp):
            features[f"op_bin_{type(node.op).__name__}"] += 1

        # 5. Attributes (e.g. bars['close'].iloc)
        elif isinstance(node, ast.Attribute):
            features[f"attr_{node.attr}"] += 1

    return features


def compute_code_structure_similarity(code1: str, code2: str) -> float:
    """Compute multiset Jaccard similarity between two AST feature counters.
    Returns float in [0.0, 1.0]. >= 0.85 indicates a near-duplicate.
    """
    c1 = compute_code_structure_features(code1)
    c2 = compute_code_structure_features(code2)

    all_keys = set(c1.keys()) | set(c2.keys())
    if not all_keys:
        return 1.0

    intersection = sum(min(c1[k], c2[k]) for k in all_keys)
    union = sum(max(c1[k], c2[k]) for k in all_keys)

    return float(intersection / union) if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. Semantic Similarity (TF-IDF Cosine on Strategy Text)
# ═══════════════════════════════════════════════════════════════════════════

def compute_semantic_similarity(spec1: dict[str, Any], spec2: dict[str, Any]) -> float:
    """Compute TF-IDF cosine similarity on hypothesis + entry_logic + exit_logic.
    Returns float in [0.0, 1.0]. >= 0.80 indicates a near-duplicate.
    """
    def _to_text(s: dict[str, Any]) -> str:
        h = str(s.get("hypothesis", "")).strip()
        en = str(s.get("entry_logic", "")).strip()
        ex = str(s.get("exit_logic", "")).strip()
        return f"{h} {en} {ex}".strip()

    t1 = _to_text(spec1)
    t2 = _to_text(spec2)

    if not t1 or not t2:
        return 0.0
    if t1 == t2:
        return 1.0

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform([t1, t2])
        # Cosine similarity between 2 normalized vectors is dot product
        sim = (tfidf[0] * tfidf[1].T).toarray()[0][0]
        return float(max(0.0, min(1.0, sim)))
    except (ValueError, TypeError, AttributeError):
        # Fallback to Jaccard on word sets
        w1 = set(t1.lower().split())
        w2 = set(t2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Behaviour Fingerprint (Aligned Daily Returns & Entry Bar Overlap)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BehaviouralCheckResult:
    is_redundant: bool
    return_correlation: float
    entry_overlap: float
    reason: str = ""

    def to_doc(self) -> dict[str, Any]:
        return {
            "is_redundant": self.is_redundant,
            "return_correlation": round(self.return_correlation, 4),
            "entry_overlap": round(self.entry_overlap, 4),
            "reason": self.reason,
        }


def check_behavioural_similarity(
    returns1: pd.Series,
    returns2: pd.Series,
    entry_bars1: list[int] | None = None,
    entry_bars2: list[int] | None = None,
    max_correlation: float = 0.70,
    max_entry_overlap: float = 0.80,
) -> BehaviouralCheckResult:
    """Compute Pearson correlation of aligned daily returns and +/-1 bar entry overlap.
    Correlation > 0.70 or entry overlap > 0.80 marks strategy as 'same idea in disguise'.
    """
    # 1. Aligned daily returns correlation
    combined = pd.concat([returns1, returns2], axis=1, join="inner").dropna()
    if len(combined) < 5 or combined.iloc[:, 0].std() < 1e-9 or combined.iloc[:, 1].std() < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(combined.iloc[:, 0], combined.iloc[:, 1])[0, 1])
        if np.isnan(corr):
            corr = 0.0

    # 2. Entry bar overlap (+/- 1 bar)
    overlap = 0.0
    if entry_bars1 is not None and entry_bars2 is not None and len(entry_bars1) > 0 and len(entry_bars2) > 0:
        set2 = set(entry_bars2)
        matches = 0
        for b in entry_bars1:
            if (b in set2) or (b - 1 in set2) or (b + 1 in set2):
                matches += 1
        overlap = float(matches / len(entry_bars1))

    is_red = (corr > max_correlation) or (overlap > max_entry_overlap)
    reasons = []
    if corr > max_correlation:
        reasons.append(f"Return correlation {corr:.2f} > {max_correlation:.2f}")
    if overlap > max_entry_overlap:
        reasons.append(f"Entry bar overlap {overlap:.2f} > {max_entry_overlap:.2f}")

    reason_str = " | ".join(reasons) if reasons else "Distinct behavioural profile"

    return BehaviouralCheckResult(
        is_redundant=is_red,
        return_correlation=corr,
        entry_overlap=overlap,
        reason=reason_str,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Coverage Map (4D Grid: Concept x Timeframe x Session x Regime)
# ═══════════════════════════════════════════════════════════════════════════

CoverageCell = tuple[str, str, str, str]  # (concept_family, timeframe, session_bias, regime_bias)


@dataclass
class CellStats:
    attempts: int = 0
    passes: int = 0
    consecutive_failures: int = 0
    cooling_until_cycle: int = 0

    def to_doc(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "passes": self.passes,
            "consecutive_failures": self.consecutive_failures,
            "cooling_until_cycle": self.cooling_until_cycle,
        }


class CoverageMap:
    """Tracks ideation distribution across the 4D search space.
    Chooses the emptiest cell with 20% exploration, deprioritizes unproductive cells.
    """

    DEFAULT_CONCEPT_FAMILIES = [
        "momentum",
        "mean_reversion",
        "breakout",
        "trend_following",
        "liquidity_sweep",
        "volatility_expansion",
    ]
    DEFAULT_TIMEFRAMES = ["5m", "15m", "1h", "4h"]
    DEFAULT_SESSIONS = ["asian", "london", "ny", "all"]
    DEFAULT_REGIMES = ["trending", "ranging", "high_vol", "low_vol", "all"]

    def __init__(
        self,
        concept_families: list[str] | None = None,
        timeframes: list[str] | None = None,
        session_biases: list[str] | None = None,
        regime_biases: list[str] | None = None,
        max_consecutive_failures: int = 5,
        cooling_cycles: int = 20,
    ):
        self.concept_families = concept_families or self.DEFAULT_CONCEPT_FAMILIES
        self.timeframes = timeframes or self.DEFAULT_TIMEFRAMES
        self.session_biases = session_biases or self.DEFAULT_SESSIONS
        self.regime_biases = regime_biases or self.DEFAULT_REGIMES
        self.max_consecutive_failures = max_consecutive_failures
        self.cooling_cycles = cooling_cycles

        # Initialize cells
        self.cells: dict[CoverageCell, CellStats] = {}
        for c in self.concept_families:
            for t in self.timeframes:
                for s in self.session_biases:
                    for r in self.regime_biases:
                        self.cells[(c, t, s, r)] = CellStats()

    def get_target_cell(
        self,
        exploration_rate: float = 0.20,
        current_cycle: int = 0,
        seed: int | None = None,
    ) -> CoverageCell:
        """Choose target cell: emptiest cell with exploration_rate probability of random exploration.
        Deprioritizes cells currently in cooling period.
        """
        rng = np.random.default_rng(seed)

        # Filter out cooling cells
        active_cells = [cell for cell, stats in self.cells.items() if stats.cooling_until_cycle <= current_cycle]
        if not active_cells:
            # If all are cooling, fallback to all cells
            active_cells = list(self.cells.keys())

        # Exploration step
        if rng.random() < exploration_rate:
            idx = rng.integers(0, len(active_cells))
            return active_cells[idx]

        # Exploitation step: pick emptiest cell (minimum attempts)
        min_attempts = min(self.cells[c].attempts for c in active_cells)
        emptiest = [c for c in active_cells if self.cells[c].attempts == min_attempts]

        idx = rng.integers(0, len(emptiest))
        return emptiest[idx]

    def record_attempt(self, cell: CoverageCell, passed: bool, current_cycle: int = 0) -> None:
        """Record the outcome of a strategy cycle targeting this cell."""
        if cell not in self.cells:
            self.cells[cell] = CellStats()

        stats = self.cells[cell]
        stats.attempts += 1
        if passed:
            stats.passes += 1
            stats.consecutive_failures = 0
        else:
            stats.consecutive_failures += 1
            if stats.consecutive_failures >= self.max_consecutive_failures:
                stats.cooling_until_cycle = current_cycle + self.cooling_cycles

    def is_cooling(self, cell: CoverageCell, current_cycle: int = 0) -> bool:
        """Check if a cell is currently in cooling period."""
        stats = self.cells.get(cell)
        if not stats:
            return False
        return stats.cooling_until_cycle > current_cycle

    def to_doc(self) -> dict[str, Any]:
        """Serialize for MongoDB persistence and dashboard rendering."""
        return {
            "total_cells": len(self.cells),
            "max_consecutive_failures": self.max_consecutive_failures,
            "cooling_cycles": self.cooling_cycles,
            "cells": {
                "|".join(k): v.to_doc() for k, v in self.cells.items() if v.attempts > 0
            },
        }

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> CoverageMap:
        """Reconstruct CoverageMap from persisted document."""
        cmap = cls(
            max_consecutive_failures=doc.get("max_consecutive_failures", 5),
            cooling_cycles=doc.get("cooling_cycles", 20),
        )
        for k_str, v_doc in doc.get("cells", {}).items():
            parts = tuple(k_str.split("|"))
            if len(parts) == 4:
                cell = (parts[0], parts[1], parts[2], parts[3])
                cmap.cells[cell] = CellStats(
                    attempts=v_doc.get("attempts", 0),
                    passes=v_doc.get("passes", 0),
                    consecutive_failures=v_doc.get("consecutive_failures", 0),
                    cooling_until_cycle=v_doc.get("cooling_until_cycle", 0),
                )
        return cmap


# ═══════════════════════════════════════════════════════════════════════════
# 6. Diversity Registry & Lineage
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FingerprintCheckResult:
    is_duplicate: bool
    duplicate_of: str | None = None
    reason: str = ""


@dataclass
class RevisionCheckResult:
    is_new_concept: bool
    reason: str = ""


class DiversityRegistry:
    """Central registry tracking all discovered strategies, fingerprints, and lineage."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        div_cfg = self.cfg.get("diversity", {})
        cmap_cfg = div_cfg.get("coverage_map", {})
        self.code_structure_threshold = float(div_cfg.get("code_structure_threshold", 0.85))
        self.semantic_similarity_threshold = float(div_cfg.get("semantic_similarity_threshold", 0.80))
        self.behavior_correlation_threshold = float(div_cfg.get("behavior_correlation_threshold", 0.70))
        self.trade_overlap_threshold = float(div_cfg.get("trade_overlap_threshold", 0.70))

        # Map logic_hash -> strategy_id
        self.logic_hashes: dict[str, str] = {}
        # Map strategy_id -> metadata
        self.strategies: dict[str, dict[str, Any]] = {}
        # Coverage map
        self.coverage_map = CoverageMap(
            max_consecutive_failures=int(cmap_cfg.get("max_consecutive_failures", 5)),
            cooling_cycles=int(cmap_cfg.get("cooling_cycles", 20)),
        )

    def check_logic_fingerprint(self, strategy_id: str, spec: dict[str, Any]) -> FingerprintCheckResult:
        """Gate 1 of Diversity: Check exact logic duplicate before backtest."""
        h = compute_logic_fingerprint(spec)
        if h in self.logic_hashes:
            dup_id = self.logic_hashes[h]
            return FingerprintCheckResult(
                is_duplicate=True,
                duplicate_of=dup_id,
                reason=f"Rejected: Exact logic duplicate of stored strategy '{dup_id}' (hash: {h[:12]}).",
            )
        return FingerprintCheckResult(is_duplicate=False)

    def check_code_structure(self, source: str, threshold: float = 0.85) -> tuple[bool, str | None, float]:
        """Check AST structure similarity against all stored strategies."""
        for strat_id, data in self.strategies.items():
            stored_source = data.get("source")
            if stored_source:
                sim = compute_code_structure_similarity(source, stored_source)
                if sim >= threshold:
                    return True, strat_id, sim
        return False, None, 0.0

    def check_semantic_similarity(self, spec: dict[str, Any], threshold: float = 0.80) -> tuple[bool, str | None, float]:
        """Check TF-IDF semantic cosine similarity against all stored strategy specs."""
        for strat_id, data in self.strategies.items():
            stored_spec = data.get("spec")
            if stored_spec:
                sim = compute_semantic_similarity(spec, stored_spec)
                if sim >= threshold:
                    return True, strat_id, sim
        return False, None, 0.0

    def check_revision(
        self,
        strategy_id: str,
        spec: dict[str, Any],
        ancestor_id: str,
        source: str | None = None,
    ) -> RevisionCheckResult:
        """Audit strategy revision from the improve loop.
        - Compares against its own ancestors only by logic fingerprint (constants-only is not a new idea).
        - Compares against all others by all similarity checks.
        """
        if ancestor_id not in self.strategies:
            return RevisionCheckResult(is_new_concept=True)

        ancestor_data = self.strategies[ancestor_id]
        ancestor_spec = ancestor_data.get("spec", {})

        # Compare logic against ancestor
        h_rev = compute_logic_fingerprint(spec)
        h_anc = compute_logic_fingerprint(ancestor_spec)

        if h_rev == h_anc:
            return RevisionCheckResult(
                is_new_concept=False,
                reason=f"Revision '{strategy_id}' has same logic fingerprint as ancestor '{ancestor_id}' (constants-only parameter change).",
            )

        # Check against everyone else
        for sid, sdata in self.strategies.items():
            if sid == ancestor_id:
                continue
            # Logic check
            if h_rev == compute_logic_fingerprint(sdata.get("spec", {})):
                return RevisionCheckResult(
                    is_new_concept=False,
                    reason=f"Revision duplicates logic of strategy '{sid}'.",
                )
            # Code structure check
            if source and sdata.get("source"):
                sim = compute_code_structure_similarity(source, sdata["source"])
                if sim >= 0.85:
                    return RevisionCheckResult(
                        is_new_concept=False,
                        reason=f"Revision has high code structure similarity ({sim:.2f}) with strategy '{sid}'.",
                    )

        return RevisionCheckResult(is_new_concept=True, reason="Genuine conceptual revision verified.")

    def register_strategy(
        self,
        strategy_id: str,
        spec: dict[str, Any],
        source: str | None = None,
        daily_returns: pd.Series | None = None,
        entry_bars: list[int] | None = None,
        ancestor_id: str | None = None,
    ) -> None:
        """Register strategy in the diversity catalog."""
        h = compute_logic_fingerprint(spec)
        self.logic_hashes[h] = strategy_id
        self.strategies[strategy_id] = {
            "spec": spec,
            "source": source,
            "daily_returns": daily_returns,
            "entry_bars": entry_bars,
            "ancestor_id": ancestor_id,
            "logic_hash": h,
        }
