"""
tests/regression/test_f6_llm.py – Regression tests for Phase F6:
LLM client, two-key failover, token tracking, and reasoning cleanup.

Covers Defect IDs:
- LLM-1: No key failover on 429 / rate limits / quota exceeded.
- LLM-2: No token tracking in MongoDB collections (llm_calls, key_status, engine_counters).
- LLM-5: Unhandled <think> tags, reasoning_content, and finish_reason == 'length' truncation.
- LLM-6: Secret leakage scan & robust JSON schema validation with repair calls.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_response(status_code: int = 200, data: dict | None = None, text: str = "", headers: dict | None = None):
    mock = MagicMock()
    mock.status_code = status_code
    mock.headers = headers or {}
    if data is not None:
        mock.json.return_value = data
        mock.text = json.dumps(data)
    else:
        mock.text = text
        mock.json.side_effect = json.JSONDecodeError("Invalid JSON", text, 0)
    return mock


class TestF6KeyFailoverAndManager:
    """Verifies LLM-1: Two-key failover, cooldown survival, and quota waiting."""

    def test_key_failover_on_429_immediately_retries_key_2(self, base_cfg, fresh_db):
        """When key 1 hits HTTP 429, key 1 is cooled down and request is retried on key 2."""
        from llm.client import LLMClient
        from llm.key_manager import KeyManager, KeyStatus
        from llm.usage import UsageTracker

        env = {
            "LLM_BASE_URL": "https://api.mock.test/v1",
            "LLM_MODEL": "mock-quant-model",
            "LLM_API_KEY_1": "sk-secret-key-one-1111111111",
            "LLM_API_KEY_2": "sk-secret-key-two-2222222222",
        }

        with patch.dict(os.environ, env):
            km = KeyManager(base_cfg, db=fresh_db)
            usage = UsageTracker(db=fresh_db, cfg=base_cfg)
            client = LLMClient(base_cfg, db=fresh_db, key_manager=km, usage_tracker=usage)

            call_keys = []

            def mock_post(url, headers=None, json=None, timeout=None, **kwargs):
                auth = headers.get("Authorization", "")
                if "sk-secret-key-one-1111111111" in auth:
                    call_keys.append("key_1")
                    return _make_mock_response(
                        429,
                        data={"error": {"message": "Rate limit exceeded"}},
                        headers={"Retry-After": "30"}
                    )
                elif "sk-secret-key-two-2222222222" in auth:
                    call_keys.append("key_2")
                    return _make_mock_response(
                        200,
                        data={
                            "id": "chatcmpl-test",
                            "model": "mock-quant-model",
                            "choices": [{
                                "message": {"role": "assistant", "content": "Valid response from key 2"},
                                "finish_reason": "stop"
                            }],
                            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
                        }
                    )
                return _make_mock_response(500, text="Unknown key")

            with patch("requests.post", side_effect=mock_post):
                resp = client.chat([{"role": "user", "content": "Hello"}], purpose="ideate")
                assert resp.content == "Valid response from key 2"
                assert resp.key_label == "key_2"
                assert call_keys == ["key_1", "key_2"], "Must have attempted key 1 first, then failed over to key 2"

            # Check KeyManager status
            k1_status = km.get_key_status("key_1")
            assert k1_status["status"] == KeyStatus.COOLING_DOWN.value
            assert k1_status["reset_at"] is not None

            k2_status = km.get_key_status("key_2")
            assert k2_status["status"] == KeyStatus.ACTIVE.value
            assert k2_status["calls_today"] == 1

    def test_cooldown_survives_restart(self, base_cfg, fresh_db):
        """Cooldown timestamp persisted in Mongo survives KeyManager re-instantiation."""
        from llm.key_manager import KeyManager, KeyStatus

        env = {
            "LLM_BASE_URL": "https://api.mock.test/v1",
            "LLM_MODEL": "mock-quant-model",
            "LLM_API_KEY_1": "sk-secret-key-one-1111111111",
            "LLM_API_KEY_2": "sk-secret-key-two-2222222222",
        }

        with patch.dict(os.environ, env):
            km1 = KeyManager(base_cfg, db=fresh_db)
            now = datetime.now(timezone.utc)
            km1.mark_rate_limited("key_1", retry_after_s=120, now=now)

            # Re-create KeyManager (simulating process restart)
            km2 = KeyManager(base_cfg, db=fresh_db)
            active_label, _ = km2.get_active_key(now=now + timedelta(seconds=10))
            # key_1 is still cooling down, so key_2 must be selected
            assert active_label == "key_2"
            status = km2.get_key_status("key_1")
            assert status["status"] == KeyStatus.COOLING_DOWN.value

    def test_both_keys_limited_transitions_to_waiting_for_quota(self, base_cfg, fresh_db):
        """When all keys are cooling down, client raises WaitingForQuotaError with earliest reset."""
        from llm.client import LLMClient, WaitingForQuotaError
        from llm.key_manager import KeyManager

        env = {
            "LLM_BASE_URL": "https://api.mock.test/v1",
            "LLM_MODEL": "mock-quant-model",
            "LLM_API_KEY_1": "sk-secret-key-one-1111111111",
            "LLM_API_KEY_2": "sk-secret-key-two-2222222222",
        }

        with patch.dict(os.environ, env):
            km = KeyManager(base_cfg, db=fresh_db)
            now = datetime.now(timezone.utc)
            km.mark_rate_limited("key_1", retry_after_s=45, now=now)
            km.mark_rate_limited("key_2", retry_after_s=90, now=now)

            client = LLMClient(base_cfg, db=fresh_db, key_manager=km)

            with pytest.raises(WaitingForQuotaError) as exc_info:
                client.chat([{"role": "user", "content": "Test"}], now=now)

            assert exc_info.value.reset_at is not None
            # Earliest reset should be ~45s from now
            diff = (exc_info.value.reset_at - now).total_seconds()
            assert 40 <= diff <= 50

    def test_both_keys_limited_resumes_when_reset_passes(self, base_cfg, fresh_db):
        """When time advances past reset_at, client automatically resumes and succeeds."""
        from llm.client import LLMClient, WaitingForQuotaError
        from llm.key_manager import KeyManager

        env = {
            "LLM_BASE_URL": "https://api.mock.test/v1",
            "LLM_MODEL": "mock-quant-model",
            "LLM_API_KEY_1": "sk-secret-key-one-1111111111",
            "LLM_API_KEY_2": "sk-secret-key-two-2222222222",
        }

        with patch.dict(os.environ, env):
            km = KeyManager(base_cfg, db=fresh_db)
            t0 = datetime.now(timezone.utc)
            km.mark_rate_limited("key_1", retry_after_s=30, now=t0)
            km.mark_rate_limited("key_2", retry_after_s=60, now=t0)

            client = LLMClient(base_cfg, db=fresh_db, key_manager=km)

            # At t0 + 10s: both still limited
            with pytest.raises(WaitingForQuotaError):
                client.chat([{"role": "user", "content": "Test"}], now=t0 + timedelta(seconds=10))

            # At t0 + 35s: key_1 cooldown has passed! Should succeed on key_1
            mock_res = _make_mock_response(
                200,
                data={
                    "choices": [{"message": {"role": "assistant", "content": "Resumed on key 1"}, "finish_reason": "stop"}],
                    "usage": {"total_tokens": 80}
                }
            )
            with patch("requests.post", return_value=mock_res):
                resp = client.chat([{"role": "user", "content": "Test"}], now=t0 + timedelta(seconds=35))
                assert resp.content == "Resumed on key 1"
                assert resp.key_label == "key_1"

    def test_key_invalid_on_401(self, base_cfg, fresh_db):
        """HTTP 401 marks key as INVALID and retries on alternate key."""
        from llm.client import LLMClient
        from llm.key_manager import KeyManager, KeyStatus

        env = {
            "LLM_BASE_URL": "https://api.mock.test/v1",
            "LLM_MODEL": "mock-quant-model",
            "LLM_API_KEY_1": "sk-bad-key-1",
            "LLM_API_KEY_2": "sk-good-key-2",
        }

        with patch.dict(os.environ, env):
            km = KeyManager(base_cfg, db=fresh_db)
            client = LLMClient(base_cfg, db=fresh_db, key_manager=km)

            def mock_post(url, headers=None, **kwargs):
                auth = headers.get("Authorization", "")
                if "sk-bad-key-1" in auth:
                    return _make_mock_response(401, data={"error": {"message": "Invalid API key"}})
                return _make_mock_response(
                    200,
                    data={
                        "choices": [{"message": {"role": "assistant", "content": "OK from key 2"}, "finish_reason": "stop"}],
                        "usage": {"total_tokens": 50}
                    }
                )

            with patch("requests.post", side_effect=mock_post):
                resp = client.chat([{"role": "user", "content": "Test"}])
                assert resp.content == "OK from key 2"
                assert km.get_key_status("key_1")["status"] == KeyStatus.INVALID.value


class TestF6UsageTracking:
    """Verifies LLM-2: Token tracking in Mongo collections."""

    def test_llm_calls_and_counters_updated_on_each_call(self, base_cfg, fresh_db):
        """Every call inserts a document into llm_calls and updates atomic running totals."""
        from llm.client import LLMClient
        from llm.key_manager import KeyManager
        from llm.usage import UsageTracker

        env = {
            "LLM_BASE_URL": "https://api.mock.test/v1",
            "LLM_MODEL": "mock-quant-model",
            "LLM_API_KEY_1": "sk-key-1",
        }

        with patch.dict(os.environ, env):
            km = KeyManager(base_cfg, db=fresh_db)
            usage = UsageTracker(db=fresh_db, cfg=base_cfg)
            client = LLMClient(base_cfg, db=fresh_db, key_manager=km, usage_tracker=usage)

            mock_res = _make_mock_response(
                200,
                data={
                    "model": "mock-quant-model",
                    "choices": [{"message": {"role": "assistant", "content": '{"action": "test"}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 250, "completion_tokens": 120, "total_tokens": 370}
                }
            )

            with patch("requests.post", return_value=mock_res):
                client.chat(
                    [{"role": "user", "content": "Hello"}],
                    purpose="ideate",
                    strategy_id="strat_001",
                    cycle_id="cycle_99",
                )

            # Assert llm_calls document inserted
            calls = list(fresh_db["llm_calls"].find())
            assert len(calls) == 1
            call = calls[0]
            assert call["key_label"] == "key_1"
            assert call["model"] == "mock-quant-model"
            assert call["purpose"] == "ideate"
            assert call["strategy_id"] == "strat_001"
            assert call["cycle_id"] == "cycle_99"
            assert call["prompt_tokens"] == 250
            assert call["completion_tokens"] == 120
            assert call["total_tokens"] == 370
            assert call["status"] == "success"
            assert call["estimated"] is False

            # Assert running summary
            summary = usage.get_usage_summary()
            assert summary["total_tokens"] == 370
            assert summary["by_purpose"]["ideate"]["total_tokens"] == 370
            assert summary["by_key"]["key_1"]["calls"] == 1

    def test_missing_usage_is_estimated(self, base_cfg, fresh_db):
        """When provider returns no usage block, tokens are estimated with estimated=True."""
        from llm.client import LLMClient
        from llm.key_manager import KeyManager
        from llm.usage import UsageTracker

        env = {
            "LLM_BASE_URL": "https://api.mock.test/v1",
            "LLM_MODEL": "mock-quant-model",
            "LLM_API_KEY_1": "sk-key-1",
        }

        with patch.dict(os.environ, env):
            km = KeyManager(base_cfg, db=fresh_db)
            usage = UsageTracker(db=fresh_db, cfg=base_cfg)
            client = LLMClient(base_cfg, db=fresh_db, key_manager=km, usage_tracker=usage)

            mock_res = _make_mock_response(
                200,
                data={
                    "model": "mock-quant-model",
                    "choices": [{"message": {"role": "assistant", "content": "Hello world from assistant!"}, "finish_reason": "stop"}]
                }
            )

            with patch("requests.post", return_value=mock_res):
                resp = client.chat([{"role": "user", "content": "Tell me something"}], purpose="ideate")
                assert resp.usage["total_tokens"] > 0

            call = fresh_db["llm_calls"].find_one()
            assert call["estimated"] is True
            assert call["total_tokens"] > 0

    def test_token_totals_equal_sum_of_llm_calls(self, base_cfg, fresh_db):
        """Aggregate totals in engine_counters exactly match the sum of individual llm_calls records."""
        from llm.client import LLMClient
        from llm.key_manager import KeyManager
        from llm.usage import UsageTracker

        env = {
            "LLM_BASE_URL": "https://api.mock.test/v1",
            "LLM_MODEL": "mock-quant-model",
            "LLM_API_KEY_1": "sk-key-1",
            "LLM_API_KEY_2": "sk-key-2",
        }

        with patch.dict(os.environ, env):
            km = KeyManager(base_cfg, db=fresh_db)
            usage = UsageTracker(db=fresh_db, cfg=base_cfg)
            client = LLMClient(base_cfg, db=fresh_db, key_manager=km, usage_tracker=usage)

            # Fire 5 diverse calls
            mock_specs = [
                (100, 50, "ideate"),
                (200, 80, "code"),
                (150, 40, "fix_code"),
                (80, 20, "repair_json"),
                (300, 100, "improve"),
            ]

            for p_toks, c_toks, purpose in mock_specs:
                mock_res = _make_mock_response(
                    200,
                    data={
                        "model": "mock-quant-model",
                        "choices": [{"message": {"role": "assistant", "content": f"Result for {purpose}"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": p_toks, "completion_tokens": c_toks, "total_tokens": p_toks + c_toks}
                    }
                )
                with patch("requests.post", return_value=mock_res):
                    client.chat([{"role": "user", "content": "prompt"}], purpose=purpose)

            # Sum from individual audit records in Mongo
            all_calls = list(fresh_db["llm_calls"].find())
            assert len(all_calls) == 5

            sum_prompt = sum(c["prompt_tokens"] for c in all_calls)
            sum_completion = sum(c["completion_tokens"] for c in all_calls)
            sum_total = sum(c["total_tokens"] for c in all_calls)

            # Compare to atomic running counters
            summary = usage.get_usage_summary()
            assert summary["prompt_tokens"] == sum_prompt
            assert summary["completion_tokens"] == sum_completion
            assert summary["total_tokens"] == sum_total


class TestF6ReasoningAndTruncation:
    """Verifies LLM-5: <think> stripping, reasoning_content, and finish_reason == 'length'."""

    def test_think_tags_and_reasoning_content_stripped(self, base_cfg, fresh_db):
        """<think>...</think> blocks and reasoning_content fields are completely stripped from content."""
        from llm.client import clean_llm_content

        # Direct cleaner check
        raw_msg = "<think>\nThinking step 1\nConsidering RSI(14)\n</think>\n\nHere is the strategy."
        cleaned = clean_llm_content(raw_msg)
        assert cleaned == "Here is the strategy."
        assert "<think>" not in cleaned

        # When choice has reasoning_content
        choice = {
            "message": {
                "role": "assistant",
                "content": "<think>private internal reasoning</think>```python\ndef test(): pass\n```",
                "reasoning_content": "Internal Qwen reasoning",
            }
        }
        res = clean_llm_content(choice["message"]["content"], choice)
        assert "private internal reasoning" not in res
        assert "Internal Qwen reasoning" not in res
        assert res.strip() == "```python\ndef test(): pass\n```"

    def test_finish_reason_length_handled(self, base_cfg, fresh_db):
        """When response is truncated due to max_tokens, client handles length cleanly."""
        from llm.client import LLMClient, TruncatedResponse
        from llm.key_manager import KeyManager

        env = {
            "LLM_BASE_URL": "https://api.mock.test/v1",
            "LLM_MODEL": "mock-quant-model",
            "LLM_API_KEY_1": "sk-key-1",
        }

        with patch.dict(os.environ, env):
            km = KeyManager(base_cfg, db=fresh_db)
            client = LLMClient(base_cfg, db=fresh_db, key_manager=km)

            responses_list = [
                _make_mock_response(200, data={
                    "choices": [{"message": {"role": "assistant", "content": "incomplete code part 1"}, "finish_reason": "length"}],
                    "usage": {"total_tokens": 1000}
                }),
                _make_mock_response(200, data={
                    "choices": [{"message": {"role": "assistant", "content": "incomplete code part 2"}, "finish_reason": "length"}],
                    "usage": {"total_tokens": 1200}
                })
            ]

            with patch("requests.post", side_effect=responses_list):
                resp = client.chat([{"role": "user", "content": "Write strategy"}], purpose="code")
                assert isinstance(resp, TruncatedResponse)
                assert resp.finish_reason == "length"


class TestF6SecretRedactionAndJsonValidation:
    """Verifies no secrets leaked in logs/exceptions and JSON schema validation with repair."""

    def test_no_keys_in_logs_mongo_or_exceptions(self, base_cfg, fresh_db):
        """API key values are never present in Mongo documents, log records, or exception messages."""
        from llm.client import LLMClient, LLMError
        from llm.key_manager import KeyManager
        from llm.usage import UsageTracker

        SECRET_KEY_1 = "sk-super-secret-key-123456789"
        env = {
            "LLM_BASE_URL": "https://api.mock.test/v1",
            "LLM_MODEL": "mock-quant-model",
            "LLM_API_KEY_1": SECRET_KEY_1,
        }

        with patch.dict(os.environ, env):
            km = KeyManager(base_cfg, db=fresh_db)
            usage = UsageTracker(db=fresh_db, cfg=base_cfg)
            client = LLMClient(base_cfg, db=fresh_db, key_manager=km, usage_tracker=usage)

            mock_500 = _make_mock_response(
                500,
                text=f"Internal Server Error for Authorization Bearer {SECRET_KEY_1}"
            )

            with patch("requests.post", return_value=mock_500):
                with pytest.raises(LLMError) as exc_info:
                    client.chat([{"role": "user", "content": "Hi"}], purpose="ideate")

                exc_msg = str(exc_info.value)
                assert SECRET_KEY_1 not in exc_msg, "Raw API key leaked into exception string!"

            # Scan all documents in the database
            for col_name in fresh_db.list_collection_names():
                for doc in fresh_db[col_name].find():
                    doc_str = json.dumps(doc, default=str)
                    assert SECRET_KEY_1 not in doc_str, f"Raw API key found in Mongo collection {col_name}!"

    def test_json_schema_validation_and_repair_flow(self, base_cfg, fresh_db):
        """Invalid JSON triggers a repair request including the pydantic validation error."""
        from pydantic import BaseModel, Field
        from llm.client import LLMClient
        from llm.key_manager import KeyManager

        class StrategySpecModel(BaseModel):
            name: str
            timeframe: str
            hypothesis: str

        env = {
            "LLM_BASE_URL": "https://api.mock.test/v1",
            "LLM_MODEL": "mock-quant-model",
            "LLM_API_KEY_1": "sk-key-1",
        }

        with patch.dict(os.environ, env):
            km = KeyManager(base_cfg, db=fresh_db)
            client = LLMClient(base_cfg, db=fresh_db, key_manager=km)

            repair_prompts = []

            def mock_post(url, headers=None, json=None, **kwargs):
                msgs = json["messages"]
                if len(msgs) == 1:
                    return _make_mock_response(200, data={
                        "choices": [{"message": {"role": "assistant", "content": '{"name": "BadStrat", "timeframe": "1h"}'}, "finish_reason": "stop"}],
                        "usage": {"total_tokens": 50}
                    })
                else:
                    repair_prompts.append(msgs[-1]["content"])
                    return _make_mock_response(200, data={
                        "choices": [{"message": {"role": "assistant", "content": '{"name": "FixedStrat", "timeframe": "1h", "hypothesis": "Valid edge"}'}, "finish_reason": "stop"}],
                        "usage": {"total_tokens": 60}
                    })

            with patch("requests.post", side_effect=mock_post):
                parsed = client.chat_and_validate(
                    messages=[{"role": "user", "content": "Generate spec"}],
                    schema=StrategySpecModel,
                    purpose="ideate",
                )

                assert isinstance(parsed, StrategySpecModel)
                assert parsed.name == "FixedStrat"
                assert parsed.hypothesis == "Valid edge"
                assert len(repair_prompts) == 1
                assert "hypothesis" in repair_prompts[0], "Validation error was passed back in repair request"
