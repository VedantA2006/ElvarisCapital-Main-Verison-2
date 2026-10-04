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


# ═══════════════════════════════════════════════════════════════════════════
# F7 "Must pass" suite (AUDIT_PROMPT.md F7.3)
# ═══════════════════════════════════════════════════════════════════════════

import re as _re
from datetime import datetime as _dt, timedelta as _td, timezone as _tz

_BASE_SPEC = {
    "name": "SessionTrend",
    "timeframe": "1h",
    "hypothesis": "Gold trends during London/NY liquidity as real-money flows hit the book.",
    "concept_family": "trend",
    "indicators_used": ["ema", "atr"],
    "entry_logic": "EMA slope positive and close above EMA",
    "exit_logic": "SL 1.5 ATR, TP 3 ATR",
    "filters": [],
    "direction": "both",
    "parameters": {"ema_n": {"default": 20, "min": 10, "max": 50}},
    "expected_trades_per_year": 80,
    "expected_failure_conditions": "Low-volatility chop",
}
_BASE_CODE = (
    "class Strategy:\n    PARAMS = {'ema_n': {'default': 20, 'min': 10, 'max': 50}}\n"
    "    def __init__(self, params):\n        self.p = params\n"
    "    def on_bar(self, bars):\n        return None\n"
)


def _trade(ts: _dt, pnl: float, direction: str = "LONG", reason: str = "tp", bars: int = 5) -> dict:
    return {
        "entry_time": ts, "exit_time": ts + _td(hours=bars), "direction": direction,
        "gross_pnl": pnl + 2.0, "spread_cost": 1.0, "slippage_cost": 0.5, "commission": 0.5,
        "swap_cost": 0.0, "net_pnl": pnl, "exit_reason": reason, "bars_held": bars,
        "mae": abs(pnl) * 0.5, "mfe": abs(pnl) * 1.2,
    }


def _planted_trades(include_asia: bool) -> list[dict]:
    """London/NY trades win; Asia trades (02:00 UTC = 11:00 Tokyo) lose. Planted weakness."""
    trades = []
    start = _dt(2021, 1, 4, tzinfo=_tz.utc)
    for d in range(0, 700, 3):
        day = start + _td(days=d)
        if day.weekday() >= 5:
            continue
        trades.append(_trade(day.replace(hour=14), 30.0 if d % 2 else -12.0,
                             reason="tp" if d % 2 else "sl"))
        if include_asia:
            trades.append(_trade(day.replace(hour=2), -25.0, direction="SHORT", reason="sl"))
    return trades


def _metrics_from(trades: list[dict]) -> dict:
    import numpy as np
    pnl = np.array([t["net_pnl"] for t in trades], dtype=float)
    sd = pnl.std(ddof=1) if len(pnl) > 1 else 1.0
    return {"sharpe": float(pnl.mean() / sd * np.sqrt(252)), "expectancy_per_trade": float(pnl.mean()),
            "profit_factor": float(pnl[pnl > 0].sum() / max(1e-9, -pnl[pnl < 0].sum())),
            "total_trades": len(pnl), "win_rate": float((pnl > 0).mean() * 100),
            "max_drawdown_pct": 10.0, "cost_gross_ratio": 0.1}


class ScriptedLLM:
    """Fake LLM client that records every prompt and answers via a script."""

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[dict] = []

    def chat_and_validate(self, messages, schema, purpose="ideate", temperature=None, **_kw):
        text = "\n".join(m["content"] for m in messages)
        self.calls.append({"purpose": purpose, "temperature": temperature, "text": text})
        return schema.model_validate(self.responder(purpose, text, len(self.calls)))

    @property
    def all_prompt_text(self) -> str:
        return "\n".join(c["text"] for c in self.calls)


def _child_payload(name: str, code: str, reason: str = "Structural change addressing diagnosed weakness.") -> dict:
    return {"spec": {**_BASE_SPEC, "name": name}, "code": code, "refinement_reason": reason}


class TestF7DiagnosticsFromTrades:
    def test_diagnostics_cover_all_required_breakdowns(self, base_cfg):
        from llm.diagnostics import build_train_diagnostics_from_trades

        trades = _planted_trades(include_asia=True)
        md = build_train_diagnostics_from_trades(trades, _metrics_from(trades), cfg=base_cfg)
        low = md.lower()
        for section in ["by year", "by month", "by session", "by weekday", "by hour (utc)",
                        "exit reason", "mae", "mfe", "holding time", "long vs short",
                        "cost impact", "losing streak", "worst drawdown"]:
            assert section in low, f"missing diagnostics section: {section}"
        assert "asia" in low and "net negative" in low  # planted weakness is surfaced
        assert "validation" not in low and "holdout" not in low
        assert len(md) < 6000, "diagnostics must stay compact (small tables, never raw trades)"

    def test_diagnostics_refuse_non_train_split(self, base_cfg):
        from llm.diagnostics import build_train_diagnostics_from_trades

        with pytest.raises(ValueError):
            build_train_diagnostics_from_trades(_planted_trades(False), {}, cfg=base_cfg, split="validation")


class TestF7ImproveLoopMustPass:
    def _loop(self, base_cfg, fresh_db, llm, **overrides):
        from llm.improve_loop import StrategyImproveLoop
        cfg = {**base_cfg, "llm": {**base_cfg["llm"], **overrides}}
        return StrategyImproveLoop(cfg=cfg, client=llm, db=fresh_db)

    def test_planted_weakness_is_improved_in_round_2(self, base_cfg, fresh_db):
        from llm.improve_loop import EvalOutcome

        def responder(purpose, text, n):
            assert purpose == "improve"
            # The fake LLM only knows to filter Asia if the diagnostics told it so.
            if "asia" in text.lower() and "net negative" in text.lower():
                return _child_payload("SessionTrend_v2", _BASE_CODE + "# SESSION_FILTER: skip asia\n")
            return _child_payload("SessionTrend_vX", _BASE_CODE + "# unrelated tweak\n")

        def evaluate(spec, code):
            trades = _planted_trades(include_asia="SESSION_FILTER" not in code)
            return EvalOutcome(status="ok", train_metrics=_metrics_from(trades), train_trades=trades)

        llm = ScriptedLLM(responder)
        res = self._loop(base_cfg, fresh_db, llm).run("root_planted", _BASE_SPEC, _BASE_CODE, evaluate)

        assert res.versions[1].version == 2
        assert "SESSION_FILTER" in res.versions[1].code
        assert res.versions[1].score > res.versions[0].score
        assert res.best.version == 2

    def test_stops_at_max_five_rounds(self, base_cfg, fresh_db):
        from llm.improve_loop import EvalOutcome
        llm = ScriptedLLM(lambda p, t, n: _child_payload(f"Imp_v{n + 1}", _BASE_CODE + f"# v{n + 1}\n"))

        def evaluate(spec, code):  # every version strictly better
            v = int(_re.findall(r"# v(\d+)", code)[-1]) if "# v" in code else 1
            trades = _planted_trades(False)
            m = _metrics_from(trades)
            m["sharpe"] = float(v)
            return EvalOutcome(status="ok", train_metrics=m, train_trades=trades)

        res = self._loop(base_cfg, fresh_db, llm).run("root_max", _BASE_SPEC, _BASE_CODE, evaluate)
        assert res.stop_reason == "max_rounds"
        assert len([c for c in llm.calls if c["purpose"] == "improve"]) == 5
        assert [v.version for v in res.versions] == [1, 2, 3, 4, 5, 6]

    def test_stops_after_two_non_improving_rounds(self, base_cfg, fresh_db):
        from llm.improve_loop import EvalOutcome
        llm = ScriptedLLM(lambda p, t, n: _child_payload(f"Flat_v{n + 1}", _BASE_CODE + f"# v{n + 1}\n"))
        trades = _planted_trades(False)

        res = self._loop(base_cfg, fresh_db, llm).run(
            "root_flat", _BASE_SPEC, _BASE_CODE,
            lambda s, c: __import__("llm.improve_loop", fromlist=["EvalOutcome"]).EvalOutcome(
                status="ok", train_metrics=_metrics_from(trades), train_trades=trades),
        )
        assert res.stop_reason == "no_improvement"
        assert len(res.versions) == 3  # v1 + 2 non-improving children

    def test_versions_are_linked_and_each_counts_as_a_trial(self, base_cfg, fresh_db):
        from llm.improve_loop import EvalOutcome
        from storage import mongo

        llm = ScriptedLLM(lambda p, t, n: _child_payload(f"Link_v{n + 1}", _BASE_CODE + f"# v{n + 1}\n"))
        trades = _planted_trades(False)
        before = mongo.get_trial_count()
        res = self._loop(base_cfg, fresh_db, llm).run(
            "root_link", _BASE_SPEC, _BASE_CODE,
            lambda s, c: EvalOutcome(status="ok", train_metrics=_metrics_from(trades), train_trades=trades),
        )
        children = res.versions[1:]
        assert mongo.get_trial_count() - before == len(children)
        ids = {v.strategy_id for v in res.versions}
        for child in children:
            assert child.parent_id in ids and child.root_id == "root_link"
            assert child.version >= 2
        stored = list(fresh_db["strategies"].find({"root_id": "root_link"}))
        assert len(stored) == len(res.versions)

    def test_validation_and_robustness_numbers_never_reach_any_prompt(self, base_cfg, fresh_db):
        from llm.improve_loop import EvalOutcome

        secret_numbers = ["7.7777", "3.3333", "4.4444", "0.1234"]
        llm = ScriptedLLM(lambda p, t, n: _child_payload(f"Sec_v{n + 1}", _BASE_CODE + f"# v{n + 1}\n"))
        trades = _planted_trades(False)
        state = {"n": 0}

        def evaluate(spec, code):
            state["n"] += 1
            if state["n"] == 1:  # v1: robustness failure with private numbers
                return EvalOutcome(status="robustness_failed", category="parameter_fragility",
                                   train_metrics=_metrics_from(trades), train_trades=trades,
                                   private_details={"neighbour_sharpe": 4.4444, "pbo": 0.1234})
            return EvalOutcome(status="validation_failed", category="insufficient_sharpe",
                               train_metrics=_metrics_from(trades), train_trades=trades,
                               private_details={"val_sharpe": 7.7777, "val_pf": 3.3333})

        res = self._loop(base_cfg, fresh_db, llm).run("root_sec", _BASE_SPEC, _BASE_CODE, evaluate)

        text = llm.all_prompt_text
        for num in secret_numbers:
            assert num not in text, f"private number {num} leaked into a prompt"
        assert "validation" not in text.lower()
        assert "failed robustness: parameter_fragility" in text  # category only
        assert len([c for c in llm.calls if c["purpose"] == "rethink"]) == 1  # at most one rethink
        assert res.public_feedback == "validation failed: insufficient_sharpe"
        assert res.stop_reason == "validation_complete"

    def test_robustness_rethink_is_attempted_at_most_once(self, base_cfg, fresh_db):
        from llm.improve_loop import EvalOutcome
        llm = ScriptedLLM(lambda p, t, n: _child_payload(f"Rb_v{n + 1}", _BASE_CODE + f"# v{n + 1}\n"))
        res = self._loop(base_cfg, fresh_db, llm).run(
            "root_rb", _BASE_SPEC, _BASE_CODE,
            lambda s, c: EvalOutcome(status="robustness_failed", category="walk_forward"),
        )
        assert res.stop_reason == "robustness_rethink_exhausted"
        assert len(llm.calls) == 1

    def test_code_error_triggers_fix_code_not_rejection(self, base_cfg, fresh_db):
        from llm.improve_loop import EvalOutcome
        from storage import mongo

        fixed_code = _BASE_CODE + "# fixed\n"

        def responder(purpose, text, n):
            if purpose == "fix_code":
                return {"spec": _BASE_SPEC, "code": fixed_code}
            return _child_payload(f"Fx_v{n + 1}", fixed_code + f"# v{n + 1}\n")

        llm = ScriptedLLM(responder)
        trades = _planted_trades(False)

        def evaluate(spec, code):
            if "# fixed" not in code:
                raise ZeroDivisionError("division by zero in on_bar")
            return EvalOutcome(status="ok", train_metrics=_metrics_from(trades), train_trades=trades)

        before = mongo.get_trial_count()
        res = self._loop(base_cfg, fresh_db, llm, max_improve_rounds=1).run(
            "root_fix", _BASE_SPEC, _BASE_CODE, evaluate)

        fix_calls = [c for c in llm.calls if c["purpose"] == "fix_code"]
        assert len(fix_calls) == 1
        assert "ZeroDivisionError" in fix_calls[0]["text"]
        assert fix_calls[0]["temperature"] == pytest.approx(base_cfg["llm"]["code_fix_temperature"])
        assert res.versions[0].fix_attempts == 1 and res.versions[0].code == fixed_code
        assert res.stop_reason != "code_error_unfixed"
        # fix_code is not a trial: only the single improve child counts
        assert mongo.get_trial_count() - before == 1

    def test_unfixable_code_error_is_explicit_not_silent(self, base_cfg, fresh_db):
        llm = ScriptedLLM(lambda p, t, n: {"spec": _BASE_SPEC, "code": _BASE_CODE})

        def evaluate(spec, code):
            raise RuntimeError("always broken")

        res = self._loop(base_cfg, fresh_db, llm).run("root_broken", _BASE_SPEC, _BASE_CODE, evaluate)
        assert res.stop_reason == "code_error_unfixed"
        assert len(llm.calls) == base_cfg["llm"]["max_code_fix_attempts"]


class TestF7PromptsMustPass:
    def test_fix_lookahead_prompt_names_check_bar_field_and_rule(self):
        from llm.prompts import render_lookahead_fix_prompt
        msgs = render_lookahead_fix_prompt(
            code="class Strategy: ...", check="truncation", bar_index=1234,
            timestamp="2022-03-01 14:00", field="signal.sl_distance",
            rule="Signals for bar t may only use bars <= t.",
        )
        text = msgs[-1]["content"]
        for needle in ["truncation", "1234", "2022-03-01 14:00", "signal.sl_distance", "bars <= t"]:
            assert needle in text

    def test_repair_json_prompt_contains_validation_error(self):
        from llm.prompts import render_repair_json_prompt
        msg = render_repair_json_prompt("spec.hypothesis: field required")
        assert "spec.hypothesis: field required" in msg["content"]

    def test_ideation_shows_last_50_ideas_and_avoid_list(self):
        from llm.prompts import render_ideation_prompt
        accepted = [{"name": f"Acc{i:02d}", "concept_family": "trend", "logic_summary": f"logic {i}"} for i in range(60)]
        rejected = [{"name": f"Rej{i:02d}", "concept_family": "breakout", "category": "parameter_fragility",
                     "reason": "neighbour sharpe 0.4321"} for i in range(60)]
        text = render_ideation_prompt("1h", past_accepted=accepted, past_rejected=rejected,
                                      max_chars=200_000)[-1]["content"]
        assert "Acc59" in text and "Acc10" in text and "Acc09" not in text
        assert "Rej59" in text and "Rej09" not in text
        assert "avoid these logics" in text.lower()
        assert "Acc59 | trend | logic 59" in text  # name | concept | one-line logic
        assert "0.4321" not in text  # rejection reason is category only

    def test_ideation_prompt_respects_budget_truncating_low_priority_first(self):
        from llm.prompts import render_ideation_prompt
        long = "x" * 300
        accepted = [{"name": f"A{i}", "concept_family": "trend", "logic_summary": long} for i in range(50)]
        rejected = [{"name": f"R{i}", "concept_family": "trend", "category": "low_sharpe"} for i in range(50)]
        text = render_ideation_prompt("4h", past_accepted=accepted, past_rejected=rejected,
                                      max_chars=8000)[-1]["content"]
        assert len(text) <= 8000
        assert "OUTPUT FORMAT" in text and "supertrend" in text  # contract + helper list survive
        assert "A49" in text  # newest ideas kept, oldest dropped first

