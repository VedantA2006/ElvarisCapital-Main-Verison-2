"""
strategy/dsl/fingerprint.py – Three-Tier Strategy Fingerprinting and Duplicate Detection.

Guarantees (Section 20):
- Level 1 (Exact): Normalized DSL and code SHA-256 hash.
- Level 2 (Logical): Structural comparison of indicators, parameters, entry/exit trees.
- Level 3 (Behavioral): Correlation of executed position sequences and trade vectors (90–95% threshold).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np

from strategy.dsl.schema import StrategyDSL


@dataclass
class StrategyFingerprint:
    exact_hash: str
    logical_hash: str
    family: str
    timeframe: str
    indicator_signature: str
    condition_signature: str


class StrategyFingerprinter:
    """Computes exact, logical, and behavioral fingerprints for candidate strategies."""

    def __init__(self, behavioral_threshold: float = 0.90):
        self.behavioral_threshold = behavioral_threshold

    def exact_hash(self, code_or_dsl: Any) -> str:
        """Level 1 exact hash invariant to formatting and comments."""
        if isinstance(code_or_dsl, str):
            no_comments = re.sub(r"#.*", "", code_or_dsl)
            normalized = re.sub(r"\s+", " ", no_comments).strip()
            return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        normalized_json = json.dumps(
            code_or_dsl.model_dump() if hasattr(code_or_dsl, "model_dump") else code_or_dsl,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(normalized_json.encode("utf-8")).hexdigest()

    def logical_hash(self, spec_or_dict: Any) -> str:
        """Level 2 logical hash invariant to key and item ordering."""
        def _canonical(o: Any) -> Any:
            if isinstance(o, dict):
                return {k: _canonical(v) for k, v in sorted(o.items())}
            elif isinstance(o, list):
                sub = [_canonical(x) for x in o]
                return sorted(sub, key=lambda x: json.dumps(x, sort_keys=True, default=str))
            return o

        normalized_json = json.dumps(_canonical(spec_or_dict), sort_keys=True, default=str)
        return hashlib.sha256(normalized_json.encode("utf-8")).hexdigest()

    def behavioral_similarity(self, positions_a: np.ndarray, positions_b: np.ndarray) -> float:
        """Alias for compute_behavioral_similarity."""
        return self.compute_behavioral_similarity(positions_a, positions_b)

    def compute_fingerprint(self, dsl: StrategyDSL, code: str = "") -> StrategyFingerprint:
        """Compute Level 1 (Exact) and Level 2 (Logical) fingerprints."""
        # Level 1: Exact Normalized JSON Hash
        normalized_json = json.dumps(dsl.model_dump(), sort_keys=True, default=str)
        exact_hash = hashlib.sha256(normalized_json.encode("utf-8")).hexdigest()

        # If compiled code provided, incorporate into exact signature
        if code:
            code_hash = hashlib.sha256(code.strip().encode("utf-8")).hexdigest()
            exact_hash = hashlib.sha256(f"{exact_hash}_{code_hash}".encode("utf-8")).hexdigest()

        # Level 2: Logical Signature
        # Sort indicators by type + params
        ind_tokens = sorted([f"{i.type}:{sorted(i.params.items())}" for i in dsl.indicators])
        indicator_sig = ";".join(ind_tokens)

        # Condition tokens
        cond_tokens = []
        for r in dsl.entry_rules:
            for c in r.all_conditions:
                cond_tokens.append(f"{r.direction}:{c.left}{c.operator}{c.right}")
        condition_sig = ";".join(sorted(cond_tokens))

        logical_payload = f"{dsl.family}|{dsl.timeframes.primary}|{indicator_sig}|{condition_sig}"
        logical_hash = hashlib.sha256(logical_payload.encode("utf-8")).hexdigest()

        return StrategyFingerprint(
            exact_hash=exact_hash,
            logical_hash=logical_hash,
            family=dsl.family,
            timeframe=dsl.timeframes.primary,
            indicator_signature=indicator_sig,
            condition_signature=condition_sig,
        )

    def is_exact_duplicate(self, candidate_fp: StrategyFingerprint, known_fps: List[StrategyFingerprint]) -> bool:
        """Check for Level 1 exact duplicate."""
        return any(candidate_fp.exact_hash == k.exact_hash for k in known_fps)

    def is_logical_duplicate(self, candidate_fp: StrategyFingerprint, known_fps: List[StrategyFingerprint]) -> bool:
        """Check for Level 2 logical duplicate."""
        return any(candidate_fp.logical_hash == k.logical_hash for k in known_fps)

    def compute_behavioral_similarity(
        self,
        positions_a: np.ndarray,
        positions_b: np.ndarray,
    ) -> float:
        """Level 3: Compare bar-by-bar position sequences (-1, 0, 1) to determine behavioral overlap."""
        if len(positions_a) == 0 or len(positions_b) == 0:
            return 0.0

        n = min(len(positions_a), len(positions_b))
        a = positions_a[:n]
        b = positions_b[:n]

        # Calculate Cohen's Kappa or exact position agreement
        matching = np.sum(a == b)
        overlap_ratio = float(matching / n)

        # Non-zero active trade agreement (where at least one was in a trade)
        active_mask = (a != 0) | (b != 0)
        if np.any(active_mask):
            active_agreement = float(np.sum(a[active_mask] == b[active_mask]) / np.sum(active_mask))
            # Weighted average between overall agreement and active trade alignment
            return 0.4 * overlap_ratio + 0.6 * active_agreement

        return overlap_ratio

    def is_behavioral_duplicate(
        self,
        candidate_positions: np.ndarray,
        existing_positions_list: List[np.ndarray],
    ) -> bool:
        """Level 3: True if candidate is >= behavioral_threshold similar to any existing strategy."""
        for existing in existing_positions_list:
            sim = self.compute_behavioral_similarity(candidate_positions, existing)
            if sim >= self.behavioral_threshold:
                return True
        return False
