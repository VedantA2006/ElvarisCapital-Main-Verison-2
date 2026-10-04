"""
tests/regression/test_f9_diversity.py – Regression tests for Phase F9 Diversity System.

Tests all requirements from AUDIT_PROMPT.md Phase F9:
1. Logic fingerprint: canonical JSON hash of key components. Exact duplicates rejected before backtest.
2. Code-structure fingerprint: normalized AST multiset Jaccard similarity >= 0.85.
3. Semantic similarity: TF-IDF cosine similarity >= 0.80.
4. Behaviour fingerprint: daily returns correlation > 0.70 or entry overlap +/-1 bar.
5. Coverage map: 4D grid, emptiest cell selection with 20% exploration, cooling period on K consecutive failures.
6. Versioning: constants-only revision flagged against ancestor; full checks against all other strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestLogicFingerprint:
    def test_identical_spec_produces_identical_hash(self):
        from strategy.diversity import compute_logic_fingerprint

        spec1 = {
            "concept_family": "momentum",
            "indicators": ["rsi", "ema", "atr"],
            "entry_type": "breakout",
            "exit_type": "trailing_stop",
            "filters": ["session_filter", "adx_trend"],
            "session": "london",
            "direction": "long",
            "timeframe": "1h",
        }
        spec2 = {
            # Same content with different key order and indicator order
            "timeframe": "1h",
            "session": "london",
            "filters": ["adx_trend", "session_filter"],
            "direction": "long",
            "concept_family": "momentum",
            "exit_type": "trailing_stop",
            "indicators": ["atr", "rsi", "ema"],
            "entry_type": "breakout",
        }

        h1 = compute_logic_fingerprint(spec1)
        h2 = compute_logic_fingerprint(spec2)
        assert h1 == h2, "Canonical JSON hashing must be invariant to key and list order"

    def test_different_spec_produces_different_hash(self):
        from strategy.diversity import compute_logic_fingerprint

        spec1 = {
            "concept_family": "momentum",
            "indicators": ["rsi"],
            "entry_type": "breakout",
            "exit_type": "atr_tp",
            "filters": [],
            "session": "all",
            "direction": "both",
            "timeframe": "1h",
        }
        spec2 = dict(spec1, concept_family="mean_reversion")
        assert compute_logic_fingerprint(spec1) != compute_logic_fingerprint(spec2)

    def test_exact_duplicate_rejected_before_backtest(self):
        from strategy.diversity import DiversityRegistry

        registry = DiversityRegistry()
        spec = {
            "concept_family": "breakout",
            "indicators": ["donchian"],
            "entry_type": "channel_break",
            "exit_type": "midpoint",
            "filters": [],
            "session": "ny",
            "direction": "long",
            "timeframe": "15m",
        }
        # First registration succeeds
        res1 = registry.check_logic_fingerprint("strat_001", spec)
        assert res1.is_duplicate is False
        registry.register_strategy("strat_001", spec=spec)

        # Duplicate registration fails and points to strat_001
        res2 = registry.check_logic_fingerprint("strat_002", spec)
        assert res2.is_duplicate is True
        assert res2.duplicate_of == "strat_001"
        assert "strat_001" in res2.reason


class TestCodeStructureFingerprint:
    def test_renamed_identifiers_and_changed_constants_flagged(self):
        from strategy.diversity import compute_code_structure_similarity

        code1 = """
class Strategy:
    def __init__(self):
        self.fast_ma = 14
        self.slow_ma = 50
    def on_bar(self, bars):
        c = bars['close'].iloc[-1]
        m1 = bars['close'].rolling(self.fast_ma).mean().iloc[-1]
        m2 = bars['close'].rolling(self.slow_ma).mean().iloc[-1]
        if m1 > m2 and c > m1:
            return Signal(direction=Direction.LONG, stop_loss=c * 0.99, take_profit=c * 1.02)
        return None
"""
        code2 = """
class Strategy:
    def __init__(self):
        self.period_alpha = 21
        self.period_beta = 100
    def on_bar(self, dataframe):
        price_val = dataframe['close'].iloc[-1]
        val1 = dataframe['close'].rolling(self.period_alpha).mean().iloc[-1]
        val2 = dataframe['close'].rolling(self.period_beta).mean().iloc[-1]
        if val1 > val2 and price_val > val1:
            return Signal(direction=Direction.LONG, stop_loss=price_val * 0.98, take_profit=price_val * 1.03)
        return None
"""
        sim = compute_code_structure_similarity(code1, code2)
        assert sim >= 0.85, f"Structural similarity {sim:.3f} was expected >= 0.85 for variable-renamed code"

    def test_different_code_structures_produce_low_similarity(self):
        from strategy.diversity import compute_code_structure_similarity

        ma_code = """
class Strategy:
    def on_bar(self, bars):
        p = bars['close'].iloc[-1]
        ma = bars['close'].mean()
        if p > ma:
            return Signal(direction=Direction.LONG, stop_loss=p * 0.99, take_profit=p * 1.01)
        return None
"""
        vol_code = """
class Strategy:
    def on_bar(self, bars):
        atr = bars['high'] - bars['low']
        v = bars['volume'].iloc[-1]
        if v > 1000 and atr.iloc[-1] < 5.0:
            return Signal(direction=Direction.SHORT, stop_loss=bars['close'].iloc[-1] + 10.0)
        return None
"""
        sim = compute_code_structure_similarity(ma_code, vol_code)
        assert sim < 0.70, f"Different code structure similarity {sim:.3f} was expected < 0.70"


class TestSemanticSimilarity:
    def test_reworded_idea_flagged(self):
        from strategy.diversity import compute_semantic_similarity

        t1 = {
            "hypothesis": "Gold momentum breakout during the US session open when price crosses above the 20 EMA.",
            "entry_logic": "Enter long at 13:00 UTC when current close crosses above the 20 period EMA.",
            "exit_logic": "Exit after 4 bars or when a trailing stop of 1.5 ATR is reached.",
        }
        t2 = {
            # Reworded copy of the same idea
            "hypothesis": "Gold momentum breakout during US session open after price crosses above the 20 EMA.",
            "entry_logic": "Enter long at 13:00 UTC once current close crosses above the 20-period EMA.",
            "exit_logic": "Exit after 4 bars or when a trailing stop of 1.5 ATR is hit.",
        }

        sim = compute_semantic_similarity(t1, t2)
        assert sim >= 0.80, f"Reworded semantic similarity was {sim:.3f}, expected >= 0.80"

    def test_distinct_idea_passes_semantic_check(self):
        from strategy.diversity import compute_semantic_similarity

        t1 = {
            "hypothesis": "US open momentum breakout.",
            "entry_logic": "Enter long at 13:00 UTC breakout.",
            "exit_logic": "Exit on 4-bar time stop.",
        }
        t2 = {
            "hypothesis": "Asian session quiet range mean-reversion using Bollinger Bands.",
            "entry_logic": "Sell when price touches upper Bollinger Band with RSI above 70 during Asian hours.",
            "exit_logic": "Take profit at middle Bollinger band or stop loss at 2x ATR.",
        }

        sim = compute_semantic_similarity(t1, t2)
        assert sim < 0.50, f"Distinct ideas had semantic similarity {sim:.3f}, expected < 0.50"


class TestBehaviouralSimilarity:
    def test_correlated_returns_flagged_as_redundant(self):
        from strategy.diversity import check_behavioural_similarity

        rng = np.random.default_rng(42)
        dates = pd.date_range("2021-01-01", periods=100, freq="1D")
        ret1 = rng.normal(0.001, 0.01, size=100)
        # Highly correlated copy (> 0.7 correlation)
        ret2 = ret1 * 0.85 + rng.normal(0, 0.002, size=100)

        s1 = pd.Series(ret1, index=dates)
        s2 = pd.Series(ret2, index=dates)

        res = check_behavioural_similarity(s1, s2, entry_bars1=[10, 20, 30], entry_bars2=[10, 21, 30])
        assert res.is_redundant is True
        assert res.return_correlation > 0.70

    def test_independent_returns_pass_behaviour_gate(self):
        from strategy.diversity import check_behavioural_similarity

        rng = np.random.default_rng(99)
        dates = pd.date_range("2021-01-01", periods=100, freq="1D")
        s1 = pd.Series(rng.normal(0.001, 0.01, size=100), index=dates)
        s2 = pd.Series(rng.normal(0.001, 0.01, size=100), index=dates)

        res = check_behavioural_similarity(s1, s2, entry_bars1=[10, 25, 40], entry_bars2=[15, 35, 55])
        assert res.is_redundant is False
        assert abs(res.return_correlation) < 0.30


class TestCoverageMap:
    def test_coverage_map_targets_emptiest_cells(self):
        from strategy.diversity import CoverageMap

        cmap = CoverageMap(
            concept_families=["momentum", "mean_reversion", "breakout"],
            timeframes=["15m", "1h"],
            session_biases=["london", "ny", "asian"],
            regime_biases=["trending", "ranging"],
        )

        # Fill some cells
        for _ in range(5):
            cmap.record_attempt(("momentum", "1h", "ny", "trending"), passed=True)
            cmap.record_attempt(("breakout", "1h", "london", "trending"), passed=False)

        # Target cell with exploration=0.0 must pick an empty cell
        cell = cmap.get_target_cell(exploration_rate=0.0, seed=123)
        assert cell not in [("momentum", "1h", "ny", "trending"), ("breakout", "1h", "london", "trending")]

    def test_cooling_period_deprioritises_failing_cells(self):
        from strategy.diversity import CoverageMap

        cmap = CoverageMap(
            concept_families=["momentum", "breakout"],
            timeframes=["1h"],
            session_biases=["ny"],
            regime_biases=["trending"],
            max_consecutive_failures=3,
            cooling_cycles=10,
        )

        cell_failing = ("momentum", "1h", "ny", "trending")
        cell_empty = ("breakout", "1h", "ny", "trending")

        for _ in range(3):
            cmap.record_attempt(cell_failing, passed=False)

        assert cmap.is_cooling(cell_failing) is True

        # When requesting target with exploration=0, failing cell should be deprioritized
        target = cmap.get_target_cell(exploration_rate=0.0, seed=42)
        assert target == cell_empty


class TestVersionLineage:
    def test_constants_only_revision_flagged_against_ancestor(self):
        from strategy.diversity import DiversityRegistry

        registry = DiversityRegistry()
        ancestor_spec = {
            "concept_family": "momentum",
            "indicators": ["rsi"],
            "entry_type": "threshold",
            "exit_type": "fixed_tp",
            "filters": [],
            "session": "london",
            "direction": "long",
            "timeframe": "1h",
            "params": {"rsi_period": 14, "rsi_threshold": 30},
        }
        registry.register_strategy("strat_v1", spec=ancestor_spec)

        # Revision changing only parameters (constants)
        revision_spec = {
            "concept_family": "momentum",
            "indicators": ["rsi"],
            "entry_type": "threshold",
            "exit_type": "fixed_tp",
            "filters": [],
            "session": "london",
            "direction": "long",
            "timeframe": "1h",
            "params": {"rsi_period": 21, "rsi_threshold": 25},
        }

        # Compared against its own ancestor: must detect same logic
        check = registry.check_revision("strat_v2", revision_spec, ancestor_id="strat_v1")
        assert check.is_new_concept is False
        assert "same logic fingerprint as ancestor" in check.reason

    def test_genuine_conceptual_revision_passes_ancestor_check(self):
        from strategy.diversity import DiversityRegistry

        registry = DiversityRegistry()
        ancestor_spec = {
            "concept_family": "momentum",
            "indicators": ["rsi"],
            "entry_type": "threshold",
            "exit_type": "fixed_tp",
            "filters": [],
            "session": "london",
            "direction": "long",
            "timeframe": "1h",
        }
        registry.register_strategy("strat_v1", spec=ancestor_spec)

        # Genuine change: adds ATR exit and trend filter
        revision_spec = {
            "concept_family": "momentum",
            "indicators": ["rsi", "atr"],
            "entry_type": "threshold",
            "exit_type": "atr_trailing",
            "filters": ["adx_trend"],
            "session": "london",
            "direction": "long",
            "timeframe": "1h",
        }
        check = registry.check_revision("strat_v2", revision_spec, ancestor_id="strat_v1")
        assert check.is_new_concept is True
