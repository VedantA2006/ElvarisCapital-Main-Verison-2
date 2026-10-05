"""
Phase 5 tests: LLM client, prompts, orchestrator, leaderboard.

These tests use MOCKED LLM responses so they run fast and deterministically.

Run:  python -m pytest tests/test_phase5_orchestrator.py -v
"""

from __future__ import annotations

import json
import textwrap
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    from core.data_loader import add_session_labels
    from core.config import load_config
    cfg = load_config()
    df = make_bars("2022-01-03", "2023-06-01", "60min", seed=seed)
    df = df.head(n).reset_index(drop=True)
    df = add_session_labels(df, cfg)
    df.attrs = {"timeframe": "1h", "split": "train", "file_hash": "test", "slice_hash": "test"}
    return df


MOCK_IDEA = {
    "name": "SMA Crossover Momentum",
    "rationale": "Fast MA crossing slow MA captures momentum shifts in gold",
    "mechanism": "Buy when 10-bar SMA crosses above 30-bar SMA, sell on reverse",
    "timeframe": "1h",
    "expected_trades_per_year": 50,
    "parameters": {"fast_period": 10, "slow_period": 30, "sl_pct": 0.02, "tp_pct": 0.04},
    "risk_management": "SL at 2% below entry, TP at 4% above"
}

MOCK_CODE = textwrap.dedent("""
import numpy as np

class Strategy:
    def __init__(self):
        self.fast = 10
        self.slow = 30

    def on_bar(self, bars):
        if len(bars) < self.slow + 1:
            return None
        closes = bars['close'].to_numpy()
        fast_ma = float(np.mean(closes[-self.fast:]))
        slow_ma = float(np.mean(closes[-self.slow:]))
        prev_fast = float(np.mean(closes[-self.fast-1:-1]))
        prev_slow = float(np.mean(closes[-self.slow-1:-1]))
        price = float(closes[-1])

        if prev_fast <= prev_slow and fast_ma > slow_ma:
            return Signal(direction=Direction.LONG,
                          stop_loss=price * 0.98,
                          take_profit=price * 1.04)
        elif prev_fast >= prev_slow and fast_ma < slow_ma:
            return Signal(direction=Direction.SHORT,
                          stop_loss=price * 1.02,
                          take_profit=price * 0.96)
        return None
""").strip()


# ═══════════════════════════════════════════════════════════════════════════
# LLM Client
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMClient:
    def test_extract_json_direct(self, base_cfg):
        from llm.client import LLMClient
        # Mock env
        with patch.dict("os.environ", {"LLM_BASE_URL": "http://test",
                                        "LLM_MODEL": "test",
                                        "LLM_API_KEY_1": "key1"}):
            client = LLMClient(base_cfg)

        raw = json.dumps({"name": "test", "value": 42})
        assert client.extract_json(raw) == {"name": "test", "value": 42}

    def test_extract_json_from_fence(self, base_cfg):
        from llm.client import LLMClient
        with patch.dict("os.environ", {"LLM_BASE_URL": "http://test",
                                        "LLM_MODEL": "test",
                                        "LLM_API_KEY_1": "key1"}):
            client = LLMClient(base_cfg)

        text = 'Here is the result:\n```json\n{"name": "test"}\n```'
        assert client.extract_json(text) == {"name": "test"}

    def test_extract_json_trailing_comma(self, base_cfg):
        from llm.client import LLMClient
        with patch.dict("os.environ", {"LLM_BASE_URL": "http://test",
                                        "LLM_MODEL": "test",
                                        "LLM_API_KEY_1": "key1"}):
            client = LLMClient(base_cfg)

        text = '```json\n{"a": 1, "b": 2,}\n```'
        assert client.extract_json(text) == {"a": 1, "b": 2}

    def test_extract_code_from_fence(self, base_cfg):
        from llm.client import LLMClient
        with patch.dict("os.environ", {"LLM_BASE_URL": "http://test",
                                        "LLM_MODEL": "test",
                                        "LLM_API_KEY_1": "key1"}):
            client = LLMClient(base_cfg)

        text = 'Here:\n```python\nclass Foo:\n    pass\n```\nDone.'
        code = client.extract_code(text)
        assert "class Foo:" in code
        assert "Done" not in code

    def test_no_api_key_raises(self, base_cfg):
        from llm.client import LLMClient, LLMError
        with patch.dict("os.environ", {"LLM_BASE_URL": "http://test",
                                        "LLM_MODEL": "test"}, clear=True):
            with pytest.raises(LLMError, match="No LLM_API_KEY"):
                LLMClient(base_cfg)

    def test_budget_limit(self, base_cfg):
        import copy
        from llm.client import LLMClient, LLMRateLimitError
        cfg = copy.deepcopy(base_cfg)
        cfg["llm"]["max_calls_per_day"] = 2

        with patch.dict("os.environ", {"LLM_BASE_URL": "http://test",
                                        "LLM_MODEL": "test",
                                        "LLM_API_KEY_1": "key1"}):
            client = LLMClient(cfg)
            client._calls_today = 2
            with pytest.raises(LLMRateLimitError):
                client.chat([{"role": "user", "content": "test"}])


# ═══════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════

class TestPrompts:
    def test_ideation_prompt_structure(self):
        from llm.prompts import ideation_prompt
        msgs = ideation_prompt("1h", past_ideas=["SMA Cross"], past_failures=["RSI failed"])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "1h" in msgs[1]["content"]
        assert "SMA Cross" in msgs[1]["content"]

    def test_code_gen_prompt_structure(self):
        from llm.prompts import code_generation_prompt
        msgs = code_generation_prompt(MOCK_IDEA, "1h")
        assert len(msgs) == 2
        assert "SMA Crossover" in msgs[1]["content"]
        assert "shift(-1)" in msgs[1]["content"]  # negative example in rules

    def test_fix_prompt_includes_error(self):
        from llm.prompts import code_fix_prompt
        msgs = code_fix_prompt("class Foo: pass", "NameError: bar")
        assert "NameError: bar" in msgs[1]["content"]

    def test_lookahead_fix_prompt_includes_violations(self):
        from llm.prompts import lookahead_fix_prompt
        msgs = lookahead_fix_prompt("code", [{"line": 5, "detail": "shift(-1)"}])
        assert "shift(-1)" in msgs[1]["content"]


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════

class TestOrchestrator:
    @pytest.fixture(autouse=True)
    def _mock_datastore(self, monkeypatch):
        from core.splits import DataStore
        mock_train = _make_df(500)
        mock_train.attrs["split"] = "train"
        mock_val = _make_df(200)
        mock_val.attrs["split"] = "validation"

        def fake_get_data(self_ds, timeframe, split):
            return mock_train if split == "train" else mock_val

        monkeypatch.setattr(DataStore, "get_data", fake_get_data)

    def _make_mock_orchestrator(self, base_cfg):
        """Create an Orchestrator with mocked LLM."""
        from llm.orchestrator import Orchestrator
        from llm.client import LLMResponse

        with patch.dict("os.environ", {"LLM_BASE_URL": "http://test",
                                        "LLM_MODEL": "test",
                                        "LLM_API_KEY_1": "key1"}):
            orch = Orchestrator(base_cfg)

        # Mock the LLM client
        mock_llm = MagicMock()

        def mock_chat(messages, temperature=0.7, max_tokens=4096):
            content = messages[-1]["content"]
            if "Invent a NEW" in content:
                return LLMResponse(content=json.dumps(MOCK_IDEA), model="test")
            elif "Write Python code" in content:
                return LLMResponse(content=f"```python\n{MOCK_CODE}\n```", model="test")
            elif "Fix" in content:
                return LLMResponse(content=f"```python\n{MOCK_CODE}\n```", model="test")
            return LLMResponse(content="{}", model="test")

        mock_llm.chat = mock_chat
        mock_llm.extract_json = orch._llm.extract_json
        mock_llm.extract_code = orch._llm.extract_code
        orch._llm = mock_llm

        return orch

    def test_single_trial_runs_to_completion(self, base_cfg):
        orch = self._make_mock_orchestrator(base_cfg)
        record = orch.run_trial("1h")
        assert record.trial_id.startswith("trial-")
        assert record.strategy_name == "SMA Crossover Momentum"
        assert record.status in ("candidate", "candidate (unproven)", "survived", "rejected", "error")
        assert record.wall_seconds > 0

    def test_trial_records_train_result(self, base_cfg):
        orch = self._make_mock_orchestrator(base_cfg)
        record = orch.run_trial("1h")
        if record.status != "error":
            assert record.train_result is not None
            assert record.train_gates is not None

    def test_rejected_trial_has_rejection_point(self, base_cfg):
        orch = self._make_mock_orchestrator(base_cfg)
        record = orch.run_trial("1h")
        if record.status == "rejected":
            assert record.rejected_at is not None

    def test_past_ideas_tracked(self, base_cfg):
        orch = self._make_mock_orchestrator(base_cfg)
        orch.run_trial("1h")
        assert len(orch._past_ideas) > 0

    def test_trial_record_serialisable(self, base_cfg):
        orch = self._make_mock_orchestrator(base_cfg)
        record = orch.run_trial("1h")
        doc = record.to_doc()
        json.dumps(doc, default=str)  # must not raise


# ═══════════════════════════════════════════════════════════════════════════
# Leaderboard / Robustness Score
# ═══════════════════════════════════════════════════════════════════════════

class TestLeaderboard:
    def test_robustness_score_bounded(self, base_cfg):
        from llm.orchestrator import compute_robustness_score, TrialRecord
        record = TrialRecord(
            trial_id="test-1", strategy_name="test",
            timeframe="1h", idea={}, source_code="", source_hash="abc",
            status="survived",
            train_result={"metrics": {"cost_gross_ratio": 0.2}},
            train_gates={"gates": [
                {"gate": "dsr", "passed": True, "data": {"probability": 0.98}},
                {"gate": "monte_carlo", "passed": True, "data": {}},
                {"gate": "regime_concentration", "passed": True, "data": {"max_concentration": 0.3}},
            ]},
            delay_test={"sharpe_drop_pct": 20},
        )
        score = compute_robustness_score(record, base_cfg)
        assert 0.0 <= score <= 100.0

    def test_better_metrics_higher_score(self, base_cfg):
        from llm.orchestrator import compute_robustness_score, TrialRecord

        good = TrialRecord(
            trial_id="good", strategy_name="good",
            timeframe="1h", idea={}, source_code="", source_hash="abc",
            status="survived",
            train_result={"metrics": {"cost_gross_ratio": 0.1}},
            train_gates={"gates": [
                {"gate": "dsr", "passed": True, "data": {"probability": 0.99}},
                {"gate": "monte_carlo", "passed": True, "data": {}},
                {"gate": "regime_concentration", "passed": True, "data": {"max_concentration": 0.2}},
            ]},
            delay_test={"sharpe_drop_pct": 5},
        )
        mediocre = TrialRecord(
            trial_id="med", strategy_name="med",
            timeframe="1h", idea={}, source_code="", source_hash="abc",
            status="survived",
            train_result={"metrics": {"cost_gross_ratio": 0.35}},
            train_gates={"gates": [
                {"gate": "dsr", "passed": True, "data": {"probability": 0.96}},
            ]},
            delay_test={"sharpe_drop_pct": 50},
        )
        assert compute_robustness_score(good, base_cfg) > compute_robustness_score(mediocre, base_cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Code Fix Loop (_eval_with_code_fix & _is_runtime_error)
# ═══════════════════════════════════════════════════════════════════════════

class TestEvalWithCodeFix:
    def test_code_fix_result_unpacking(self):
        from llm.orchestrator import CodeFixResult
        cfr = CodeFixResult("mock_res", "mock_code", 2, True)
        assert cfr.result == "mock_res"
        assert cfr.code == "mock_code"
        assert cfr.fix_attempts == 2
        assert cfr.success is True
        # Unpack 4-tuple
        res, code, attempts, succ = cfr
        assert (res, code, attempts, succ) == ("mock_res", "mock_code", 2, True)
        # Unpack 2-tuple
        res2, code2 = cfr
        assert (res2, code2) == ("mock_res", "mock_code")

    def test_is_runtime_error_detection(self, base_cfg):
        from llm.orchestrator import Orchestrator
        orch = Orchestrator(base_cfg)

        # None is error
        is_err, msg = orch._is_runtime_error(None)
        assert is_err is True

        # Successful result
        from unittest.mock import MagicMock
        passed_res = MagicMock()
        passed_res.all_passed = True
        assert orch._is_runtime_error(passed_res)[0] is False

        # Gate failure that is NOT runtime error (e.g. basic_quality)
        quality_res = MagicMock()
        quality_res.all_passed = False
        quality_res.stopped_at = "basic_quality"
        gate_res = MagicMock()
        gate_res.passed = False
        gate_res.name = "basic_quality"
        gate_res.category = "economic"
        gate_res.details = {"sharpe": 0.2}
        quality_res.results = [gate_res]
        is_err, _ = orch._is_runtime_error(quality_res)
        assert is_err is False

        # Determinism failure with Traceback in detail
        crash_res = MagicMock()
        crash_res.all_passed = False
        crash_res.stopped_at = "determinism"
        gate_crash = MagicMock()
        gate_crash.passed = False
        gate_crash.name = "determinism"
        gate_crash.category = "execution"
        gate_crash.details = {"detail": "Failed with error: Traceback: IndexError at line 42"}
        crash_res.results = [gate_crash]
        is_err, msg = orch._is_runtime_error(crash_res)
        assert is_err is True
        assert "IndexError" in msg

    def test_eval_with_code_fix_successful_fix(self, base_cfg):
        from llm.orchestrator import Orchestrator, TrialRecord
        from llm.client import LLMResponse
        from unittest.mock import MagicMock

        orch = Orchestrator(base_cfg)
        mock_llm = MagicMock()
        fixed_code = "class Strategy:\n    def on_bar(self, bars):\n        return None"
        mock_llm.chat.return_value = LLMResponse(content=f"```python\n{fixed_code}\n```", model="test")
        mock_llm.extract_code.return_value = fixed_code
        orch._llm = mock_llm

        def mock_eval(c):
            # Once fixed, evaluation passes
            res = MagicMock()
            res.all_passed = True
            return res

        rec = TrialRecord(trial_id="t1", strategy_name="test", timeframe="1h", idea={}, source_code="bad", source_hash="1")
        fix_res = orch._eval_with_code_fix(
            code="bad_code",
            traceback_str="KeyError: 'close' on bar 59",
            spec={"name": "test"},
            record=rec,
            evaluate_fn=mock_eval,
        )

        assert fix_res.success is True
        assert fix_res.fix_attempts == 1
        assert fix_res.code == fixed_code
        assert rec.source_code == fixed_code
        assert rec.llm_calls == 1

    def test_eval_with_code_fix_no_initial_error(self, base_cfg):
        from llm.orchestrator import Orchestrator, TrialRecord
        from llm.client import LLMResponse
        from unittest.mock import MagicMock

        orch = Orchestrator(base_cfg)
        mock_llm = MagicMock()
        fixed_code = "class Strategy:\n    def on_bar(self, bars):\n        return None"
        mock_llm.chat.return_value = LLMResponse(content=f"```python\n{fixed_code}\n```", model="test")
        mock_llm.extract_code.return_value = fixed_code
        orch._llm = mock_llm

        calls = []
        def mock_eval(c):
            calls.append(c)
            if len(calls) == 1:
                # First run (no initial traceback passed) fails
                raise RuntimeError("ZeroDivisionError: division by zero")
            res = MagicMock()
            res.all_passed = True
            return res

        fix_res = orch._eval_with_code_fix(
            code="bad_code",
            spec={"name": "test"},
            evaluate_fn=mock_eval,
        )

        assert fix_res.success is True
        assert fix_res.fix_attempts == 1
        assert len(calls) == 2


    def test_eval_with_code_fix_exhausted(self, base_cfg):
        from llm.orchestrator import Orchestrator
        from llm.client import LLMResponse
        from unittest.mock import MagicMock

        orch = Orchestrator(base_cfg)
        mock_llm = MagicMock()
        broken_code = "class Strategy:\n    pass"
        mock_llm.chat.return_value = LLMResponse(content=f"```python\n{broken_code}\n```", model="test")
        mock_llm.extract_code.return_value = broken_code
        orch._llm = mock_llm

        def always_fail(c):
            raise ValueError("Persistent bug")

        fix_res = orch._eval_with_code_fix(
            code="broken_code",
            traceback_str="ValueError: Persistent bug",
            spec={"name": "test"},
            evaluate_fn=always_fail,
        )

        assert fix_res.success is False
        assert fix_res.fix_attempts == base_cfg.get("llm", {}).get("max_code_fix_attempts", 3)

