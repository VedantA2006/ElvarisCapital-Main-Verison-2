"""
llm/client.py – OpenAI-compatible chat completions client with two-key failover,
reasoning cleanup, token tracking, and structured validation.

Handles:
- Two-key failover on 429/quota/401 with persistent cooldown in KeyManager
- Token estimation & audit telemetry in UsageTracker
- Stripping <think>...</think> and reasoning_content from Qwen/DeepSeek models
- finish_reason == 'length' retry and TruncatedResponse handling
- Redaction of secret keys from all logs, exceptions, and payloads
- Pydantic schema validation with automatic JSON repair loop
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from llm.key_manager import AllKeysUnavailableError, KeyManager
from llm.usage import UsageTracker, estimate_tokens

_log = logging.getLogger("quantforge.llm.client")

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
    key_label: str = "key_1"


@dataclass
class TruncatedResponse:
    content: str
    model: str
    finish_reason: str = "length"
    key_label: str = "key_1"
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0


class LLMError(RuntimeError):
    pass


class LLMRateLimitError(LLMError):
    pass


class WaitingForQuotaError(LLMRateLimitError):
    def __init__(self, message: str, reset_at: datetime | None = None):
        super().__init__(message)
        self.reset_at = reset_at


class LLMValidationError(LLMError):
    pass


def clean_llm_content(content: str, raw_choice: dict[str, Any] | None = None) -> str:
    """Strip <think>...</think> blocks and reasoning_content fields."""
    if not content:
        return ""

    # Remove matched <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    # Also handle unclosed <think> if truncated mid-thought
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Never include separate reasoning_content in the final cleaned output
    return text.strip()


class LLMClient:
    """Resilient LLM client with two-key failover and token accounting."""

    def __init__(
        self,
        cfg: dict,
        db=None,
        key_manager: KeyManager | None = None,
        usage_tracker: UsageTracker | None = None,
    ):
        self._cfg = cfg.get("llm", {})
        self._base_url = (os.environ.get("LLM_BASE_URL", "") or self._cfg.get("base_url", "")).rstrip("/")
        self._model = os.environ.get("LLM_MODEL", "") or self._cfg.get("model", "qwen-plus")
        self._timeout = self._cfg.get("timeout_seconds", 120)
        self._max_retries = self._cfg.get("backoff_max_retries", 3)
        self._supports_json_mode = self._cfg.get("supports_json_mode", False)

        # Token caps per purpose
        self._purpose_max_tokens = {
            "ideate": 3000,
            "code": 6000,
            "improve": 6000,
            "fix_code": 4000,
            "fix_lookahead": 4000,
            "repair_json": 2000,
        }

        try:
            self._km = key_manager if key_manager is not None else KeyManager(cfg, db=db)
        except Exception as e:
            if "No LLM_API_KEY" in str(e):
                raise LLMError("No LLM_API_KEY_* found in environment.") from e
            raise
        self._usage = usage_tracker if usage_tracker is not None else UsageTracker(db=db, cfg=cfg)

    @property
    def _calls_today(self) -> int:
        doc = self._km._col.find_one({"label": "key_1"})
        return doc.get("calls_today", 0) if doc else 0

    @_calls_today.setter
    def _calls_today(self, val: int) -> None:
        self._km._col.update_many({}, {"$set": {"calls_today": val}})
        if val >= self._km._max_calls_per_day:
            for label in self._km._order:
                self._km.mark_budget_exhausted(label)

    def _parse_retry_after(self, headers: dict[str, str]) -> float | None:
        """Parse Retry-After header as seconds if present."""
        val = headers.get("Retry-After") or headers.get("retry-after")
        if not val:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    def chat(
        self,
        messages: list[dict[str, str]],
        purpose: str = "ideate",
        temperature: float | None = None,
        max_tokens: int | None = None,
        strategy_id: str | None = None,
        cycle_id: str | None = None,
        now: datetime | None = None,
    ) -> LLMResponse | TruncatedResponse:
        """Send chat completion with automatic two-key failover on rate limits/errors."""
        temp = temperature if temperature is not None else self._cfg.get("ideation_temperature", 0.7)
        tokens_limit = max_tokens or self._purpose_max_tokens.get(purpose, 4096)

        max_attempts = 10
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            current_time = now or datetime.now(timezone.utc)

            try:
                key_label, key_secret = self._km.get_active_key(now=current_time)
            except AllKeysUnavailableError as e:
                raise WaitingForQuotaError(self._km.redact(str(e)), reset_at=e.reset_at)

            headers = {
                "Authorization": f"Bearer {key_secret}",
                "Content-Type": "application/json",
            }
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": tokens_limit,
            }
            if self._supports_json_mode and purpose in ("ideate", "repair_json"):
                payload["response_format"] = {"type": "json_object"}

            t0 = time.perf_counter()
            resp = None
            try:
                resp = requests.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                latency = int((time.perf_counter() - t0) * 1000)
                err_msg = self._km.redact(str(exc))
                _log.warning("Network/Timeout on %s: %s. Retrying...", key_label, err_msg)
                self._usage.log_call(
                    key_label=key_label,
                    model=self._model,
                    purpose=purpose,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=latency,
                    status="timeout" if isinstance(exc, requests.exceptions.Timeout) else "network_error",
                    error_class=type(exc).__name__,
                    strategy_id=strategy_id,
                    cycle_id=cycle_id,
                    now=current_time,
                )
                if attempt >= self._max_retries:
                    raise LLMError(f"LLM network error after {self._max_retries} attempts: {err_msg}")
                continue

            latency = int((time.perf_counter() - t0) * 1000)

            # Handle 429 Rate Limit
            if resp.status_code == 429:
                retry_after = self._parse_retry_after(dict(resp.headers))
                self._km.mark_rate_limited(key_label, retry_after_s=retry_after, now=current_time)
                self._usage.log_call(
                    key_label=key_label,
                    model=self._model,
                    purpose=purpose,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=latency,
                    status="rate_limited",
                    error_class="HTTP429",
                    strategy_id=strategy_id,
                    cycle_id=cycle_id,
                    now=current_time,
                )
                _log.warning("HTTP 429 on %s, failing over immediately...", key_label)
                continue

            # Handle 401 / 403 Invalid API Key
            if resp.status_code in (401, 403):
                self._km.mark_invalid(key_label, reason=f"HTTP {resp.status_code}")
                self._usage.log_call(
                    key_label=key_label,
                    model=self._model,
                    purpose=purpose,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=latency,
                    status="invalid_key",
                    error_class=f"HTTP{resp.status_code}",
                    strategy_id=strategy_id,
                    cycle_id=cycle_id,
                    now=current_time,
                )
                _log.error("HTTP %s on %s, marked invalid, failing over...", resp.status_code, key_label)
                continue

            # Handle 5xx Server Errors
            if resp.status_code >= 500:
                err_text = self._km.redact(resp.text[:300])
                self._usage.log_call(
                    key_label=key_label,
                    model=self._model,
                    purpose=purpose,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=latency,
                    status="server_error",
                    error_class=f"HTTP{resp.status_code}",
                    strategy_id=strategy_id,
                    cycle_id=cycle_id,
                    now=current_time,
                )
                if attempt >= self._max_retries:
                    raise LLMError(f"LLM API error {resp.status_code}: {err_text}")
                continue

            # Other non-200 errors
            if resp.status_code != 200:
                redacted_body = self._km.redact(resp.text[:500])
                self._usage.log_call(
                    key_label=key_label,
                    model=self._model,
                    purpose=purpose,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=latency,
                    status="api_error",
                    error_class=f"HTTP{resp.status_code}",
                    strategy_id=strategy_id,
                    cycle_id=cycle_id,
                    now=current_time,
                )
                raise LLMError(f"LLM API error {resp.status_code}: {redacted_body}")

            # Process 200 OK Response
            try:
                data = resp.json()
            except json.JSONDecodeError as exc:
                raise LLMError(f"Provider returned invalid JSON response: {exc}")

            choices = data.get("choices", [])
            if not choices:
                raise LLMError("Provider returned no choices in response")

            first_choice = choices[0]
            finish_reason = first_choice.get("finish_reason", "stop")
            raw_content = first_choice.get("message", {}).get("content", "") or ""

            # Check usage block or estimate tokens
            usage_data = data.get("usage")
            estimated = False
            if not usage_data or not isinstance(usage_data, dict) or not usage_data.get("total_tokens"):
                p_toks = sum(estimate_tokens(m.get("content", "")) for m in messages)
                c_toks = estimate_tokens(raw_content)
                usage_data = {
                    "prompt_tokens": p_toks,
                    "completion_tokens": c_toks,
                    "total_tokens": p_toks + c_toks,
                }
                estimated = True
            else:
                p_toks = int(usage_data.get("prompt_tokens", 0))
                c_toks = int(usage_data.get("completion_tokens", 0))

            # Record success in KeyManager & UsageTracker
            self._km.record_success(key_label, usage_data.get("total_tokens", 0), now=current_time)
            self._usage.log_call(
                key_label=key_label,
                model=data.get("model", self._model),
                purpose=purpose,
                prompt_tokens=p_toks,
                completion_tokens=c_toks,
                latency_ms=latency,
                status="truncated" if finish_reason == "length" else "success",
                strategy_id=strategy_id,
                cycle_id=cycle_id,
                estimated=estimated,
                now=current_time,
            )

            # Handle finish_reason == "length"
            if finish_reason == "length":
                _log.warning("Response truncated by max_tokens limit (%d) on %s", tokens_limit, key_label)
                cleaned = clean_llm_content(raw_content, first_choice)
                return TruncatedResponse(
                    content=cleaned,
                    model=data.get("model", self._model),
                    finish_reason="length",
                    key_label=key_label,
                    usage=usage_data,
                    latency_ms=latency,
                )

            cleaned_content = clean_llm_content(raw_content, first_choice)
            return LLMResponse(
                content=cleaned_content,
                model=data.get("model", self._model),
                usage=usage_data,
                latency_ms=latency,
                raw=data,
                key_label=key_label,
            )

        raise LLMError("Exceeded maximum LLM request attempts across all keys.")

    def extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from raw LLM text (handles fences, trailing commas)."""
        # 1. Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            _log.debug("direct JSON parse failed (%s); trying markdown fences", exc)

        # 2. Extract from markdown code fences or brackets
        patterns = [
            r"```json\s*\n(.*?)\n```",
            r"```\s*\n(.*?)\n```",
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.DOTALL)
            if match:
                candidate = match.group(1) if match.lastindex else match.group(0)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Repair trailing commas
                    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        continue

        raise LLMError(f"Could not extract JSON from LLM response: {text[:200]}...")

    def chat_and_validate(
        self,
        messages: list[dict[str, str]],
        schema: Type[T],
        purpose: str = "ideate",
        max_repair: int | None = None,
        strategy_id: str | None = None,
        cycle_id: str | None = None,
    ) -> T:
        """Call LLM and validate response against Pydantic schema with repair retry loop."""
        repair_budget = max_repair if max_repair is not None else self._cfg.get("max_json_repair_attempts", 2)
        current_messages = list(messages)
        attempt = 0

        while attempt <= repair_budget:
            attempt += 1
            resp = self.chat(
                current_messages,
                purpose=purpose if attempt == 1 else "repair_json",
                strategy_id=strategy_id,
                cycle_id=cycle_id,
            )

            if isinstance(resp, TruncatedResponse):
                raise LLMError(f"Response truncated with finish_reason={resp.finish_reason}")

            try:
                json_dict = self.extract_json(resp.content)
                parsed = schema.model_validate(json_dict)
                return parsed
            except (json.JSONDecodeError, LLMError, ValidationError) as exc:
                err_msg = str(exc)
                _log.warning("Validation failed (attempt %d/%d): %s", attempt, repair_budget + 1, err_msg[:200])

                if attempt > repair_budget:
                    raise LLMValidationError(f"Failed to obtain valid {schema.__name__} after {repair_budget} repairs: {err_msg}")

                # Append repair instructions
                current_messages.append({"role": "assistant", "content": resp.content})
                current_messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous response produced a validation error:\n{err_msg}\n\n"
                        "Please correct the error and output ONLY the complete, valid JSON object matching the required schema."
                    ),
                })

        raise LLMValidationError("JSON repair attempts exhausted.")

    def extract_code(self, text: str) -> str:
        """Extract Python code from LLM response."""
        match = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
