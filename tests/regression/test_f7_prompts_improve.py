"""
tests/regression/test_f7_prompts_improve.py – Regression tests for Phase F7:
Strategy contract, Pydantic schemas, Jinja2 prompts, train-only diagnostics, and improve loop.

Covers Defect IDs:
- LLM-3: No improve loop (backtest feedback, structural refinement, retry on error).
- LLM-4: Prompts lack hypotheses, past logic summaries, helper library, and parameter contracts.
- BT-12: Strategy exceptions feedback (clean tracebacks sent to LLM for code repair).
- SBX-5: Parameter contract (PARAMS declaration, default/min/max, capped at 6).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError


class TestF7StrategyContractSchemas:
    """Verifies F7.1: Pydantic schemas, hypothesis requirements, and parameter cap."""

    def test_schema_requires_hypothesis(self):
        """Reject strategy specs that omit or have trivial hypothesis."""
        from llm.schemas import StrategySpec, ParameterDef

        # Missing hypothesis
        with pytest.raises(ValidationError) as exc_info:
            StrategySpec(
                name="NoHypoStrat",
                timeframe="1h",
                hypothesis="",  # Empty
                concept_family="trend",
                indicators_used=["sma"],
                entry_logic="SMA cross",
                exit_logic="SL at 20 pips",
                filters=[],
                direction="both",
                parameters={"fast": ParameterDef(default=10, min=5, max=20)},
                expected_trades_per_year=50,
                expected_failure_conditions="Ranging market",
            )
        assert "hypothesis" in str(exc_info.value).lower()

    def test_schema_enforces_parameter_cap_of_six(self):
        """Reject strategy specs with more than 6 parameters (hard cap)."""
        from llm.schemas import StrategySpec, ParameterDef

        params = {f"p_{i}": ParameterDef(default=10, min=1, max=50) for i in range(7)}
        with pytest.raises(ValidationError) as exc_info:
            StrategySpec(
                name="TooManyParamsStrat",
                timeframe="1h",
                hypothesis="Gold trends strongly during London session due to institutional order flow.",
                concept_family="trend",
                indicators_used=["sma"],
                entry_logic="SMA cross",
                exit_logic="SL at 20 pips",
                filters=[],
                direction="both",
                parameters=params,  # 7 parameters > 6 cap
                expected_trades_per_year=50,
                expected_failure_conditions="Ranging market",
            )
        assert "parameter" in str(exc_info.value).lower()

    def test_valid_strategy_payload_validates(self):
        """Valid StrategyResponse with spec and code parses successfully."""
        from llm.schemas import StrategyResponse, StrategySpec, ParameterDef

        payload = {
            "spec": {
                "name": "LondonBreakout",
                "timeframe": "1h",
                "hypothesis": "Asian range wicks are liquidity pools swept before London trend continuation.",
                "concept_family": "breakout",
                "indicators_used": ["session_range", "atr"],
                "entry_logic": "Buy on breakout of Asian high with London session filter",
                "exit_logic": "SL at Asian low, TP at 2x Asian range",
                "filters": ["session == london"],
                "session_filter": "london",
                "direction": "long_only",
                "parameters": {
                    "buffer_atr": {"default": 0.2, "min": 0.1, "max": 0.5},
                    "tp_multiple": {"default": 2.0, "min": 1.0, "max": 4.0},
                },
                "expected_trades_per_year": 120,
                "expected_failure_conditions": "Low volatility bank holidays",
            },
            "code": "class Strategy:\n    PARAMS = {}\n    def __init__(self, params):\n        pass\n    def on_bar(self, bars):\n        return None\n",
        }
        res = StrategyResponse.model_validate(payload)
        assert res.spec.name == "LondonBreakout"
        assert len(res.spec.parameters) == 2
        assert "Strategy" in res.code


class TestF7PromptTemplates:
    """Verifies F7.2: Jinja2 templates, helper reference library, and past ideas formatting."""

    def test_ideation_prompt_renders_helpers_and_past_logics(self):
        """Ideate prompt must include core.indicators catalog and past idea logic summaries."""
        from llm.prompts import render_ideation_prompt

        past_accepted = [
            {"name": "AsianSweep", "concept_family": "session", "logic_summary": "Fades Asian extremes with 1.5x ATR stop"}
        ]
        past_rejected = [
            {"name": "PureRsi", "concept_family": "mean_reversion", "reason": "Failed walk-forward efficiency (<0.5)"}
        ]

        messages = render_ideation_prompt(
            timeframe="1h",
            target_cell={"concept_family": "market_structure", "session": "london"},
            past_accepted=past_accepted,
            past_rejected=past_rejected,
        )

        prompt_text = "\n".join(m["content"] for m in messages)
        # Must include helper indicators
        assert "core.indicators" in prompt_text or "supertrend" in prompt_text or "swing_high" in prompt_text
        # Must include past logic summary
        assert "AsianSweep" in prompt_text
        assert "Fades Asian extremes" in prompt_text
        assert "PureRsi" in prompt_text
        assert "market_structure" in prompt_text

    def test_fix_code_prompt_includes_clean_traceback(self):
        """Fix code prompt provides line-numbered source and truncated traceback without absolute paths."""
        from llm.prompts import render_code_fix_prompt

        raw_code = "def on_bar(self, bars):\n    x = 1 / 0\n    return None"
        tb = 'Traceback (most recent call last):\n  File "c:\\secret\\path\\strategy.py", line 2, in on_bar\n    x = 1 / 0\nZeroDivisionError: division by zero'

        messages = render_code_fix_prompt(code=raw_code, error=tb)
        prompt_text = "\n".join(m["content"] for m in messages)

        assert "ZeroDivisionError" in prompt_text
        assert "line 2" in prompt_text
        # Absolute path should be sanitized
        assert "c:\\secret\\path" not in prompt_text


class TestF7DiagnosticsAndImproveLoop:
    """Verifies F7.3 & BT-12: Train-only diagnostics and autonomous structural improvement."""

    def test_train_diagnostics_builder(self):
        """Builds compact markdown diagnostics strictly from train metrics and trades."""
        from llm.diagnostics import build_train_diagnostics

        metrics = {
            "sharpe": 1.45,
            "sortino": 1.82,
            "profit_factor": 1.62,
            "max_drawdown_pct": 8.5,
            "win_rate": 54.2,
            "total_trades": 180,
            "cost_gross_ratio": 0.18,
        }
        trades_summary = {
            "by_session": {"asia": {"trades": 40, "pnl": -200}, "london": {"trades": 90, "pnl": 1500}, "ny": {"trades": 50, "pnl": 800}},
            "by_weekday": {"mon": 200, "tue": 400, "wed": 500, "thu": -100, "fri": 1100},
            "by_exit_reason": {"tp": 80, "sl": 75, "trailing": 25},
            "long_short_split": {"long_trades": 100, "long_win_rate": 58.0, "short_trades": 80, "short_win_rate": 49.0},
        }

        diag_md = build_train_diagnostics(metrics, trades_summary)
        assert "Sharpe: 1.45" in diag_md or "1.45" in diag_md
        assert "london" in diag_md.lower()
        assert "tp" in diag_md.lower()
        # Verify no mention of validation or holdout data
        assert "validation" not in diag_md.lower()
        assert "holdout" not in diag_md.lower()

    def test_improve_loop_creates_child_version_with_structural_refinement(self, base_cfg, fresh_db):
        """Improve loop takes a parent strategy, asks LLM for structural refinement, increments version and links parent_id."""
        from llm.client import LLMClient
        from llm.improve_loop import StrategyImproveLoop
        from llm.key_manager import KeyManager

        parent_spec = {
            "name": "TrendSma",
            "timeframe": "1h",
            "hypothesis": "Trend following gold using fast/slow SMA momentum.",
            "concept_family": "trend",
            "indicators_used": ["sma"],
            "entry_logic": "Fast SMA crosses slow SMA",
            "exit_logic": "Opposite cross or SL at 2x ATR",
            "filters": [],
            "direction": "both",
            "parameters": {"fast": {"default": 10, "min": 5, "max": 20}},
            "expected_trades_per_year": 60,
            "expected_failure_conditions": "Chop",
        }
        parent_code = "class Strategy:\n    PARAMS={'fast': {'default': 10, 'min': 5, 'max': 20}}\n    def __init__(self, p): pass\n    def on_bar(self, b): return None\n"

        train_diagnostics = "### Train Performance\nSharpe: 1.1, Win Rate: 48%. Weakness: 60% of losses occur during low-vol Asian hours."

        # Mock LLM response for improve step
        improved_response = {
            "spec": {
                **parent_spec,
                "name": "TrendSma_v2",
                "hypothesis": "Trend following gold filtered by London volatility to eliminate Asian chop losses.",
                "filters": ["session != asia"],
                "session_filter": "london_ny",
                "parameters": {
                    "fast": {"default": 10, "min": 5, "max": 20},
                    "min_atr": {"default": 2.0, "min": 1.0, "max": 5.0},
                },
            },
            "code": "class Strategy:\n    PARAMS={'fast': {'default': 10, 'min': 5, 'max': 20}, 'min_atr': {'default': 2.0, 'min': 1.0, 'max': 5.0}}\n    def __init__(self, p): pass\n    def on_bar(self, b): return None\n",
            "refinement_reason": "Filtered Asian session to avoid choppy false breakouts identified in diagnostics.",
        }

        from llm.schemas import StrategyImproveResponse

        mock_llm_client = MagicMock()
        mock_llm_client.chat_and_validate.return_value = StrategyImproveResponse.model_validate(improved_response)

        improver = StrategyImproveLoop(cfg=base_cfg, client=mock_llm_client, db=fresh_db)
        child = improver.improve_strategy(
            parent_strategy_id="strat_parent_001",
            parent_version=1,
            parent_spec=parent_spec,
            parent_code=parent_code,
            train_diagnostics=train_diagnostics,
        )

        assert child["parent_id"] == "strat_parent_001"
        assert child["version"] == 2
        assert child["spec"]["name"] == "TrendSma_v2"
        assert "min_atr" in child["spec"]["parameters"]
        assert len(child["spec"]["parameters"]) <= 6
        assert child["refinement_reason"] != ""
