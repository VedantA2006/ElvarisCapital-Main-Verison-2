"""
llm/client.py – LLM API client for strategy ideation and code generation.

Uses an OpenAI-compatible endpoint (configured via .env).
Handles:
- API key rotation (LLM_API_KEY_1, LLM_API_KEY_2, ...)
- Rate limiting / cooldown
- JSON response extraction and repair
- Structured retry with error feedback
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMClient:
    """OpenAI-compatible chat completions client with key rotation and budget."""

    def __init__(self, cfg: dict):
        self._cfg = cfg["llm"]
        self._base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self._model = os.environ.get("LLM_MODEL", "")
        self._timeout = self._cfg.get("timeout_seconds", 120)
        self._cooldown = self._cfg.get("cooldown_default_seconds", 60)

        # Collect API keys
        self._keys: list[str] = []
        for i in range(1, 10):
            k = os.environ.get(f"LLM_API_KEY_{i}", "")
            if k:
                self._keys.append(k)
        if not self._keys:
            raise LLMError("No LLM_API_KEY_* found in environment.")

        self._key_idx = 0
        self._calls_today = 0
        self._tokens_today = 0
        self._day_start = datetime.now(timezone.utc).date()

    def _rotate_key(self) -> str:
        key = self._keys[self._key_idx % len(self._keys)]
        if self._cfg.get("alternate_keys", False):
            self._key_idx += 1
        return key

    def _check_budget(self):
        today = datetime.now(timezone.utc).date()
        if today != self._day_start:
            self._calls_today = 0
            self._tokens_today = 0
            self._day_start = today

        if self._calls_today >= self._cfg.get("max_calls_per_day", 500):
            raise LLMRateLimitError(f"Daily call limit reached: {self._calls_today}")
        if self._tokens_today >= self._cfg.get("max_tokens_per_day", 2000000):
            raise LLMRateLimitError(f"Daily token limit reached: {self._tokens_today}")

    def chat(self, messages: list[dict[str, str]],
             temperature: float = 0.7,
             max_tokens: int = 4096) -> LLMResponse:
        """Send a chat completion request."""
        self._check_budget()
        key = self._rotate_key()

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout:
            raise LLMError(f"LLM request timed out after {self._timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise LLMError(f"LLM connection error: {e}")

        latency = int((time.perf_counter() - t0) * 1000)

        if resp.status_code == 429:
            raise LLMRateLimitError(f"Rate limited (429). Retry after cooldown.")
        if resp.status_code != 200:
            raise LLMError(f"LLM API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        self._calls_today += 1
        self._tokens_today += usage.get("total_tokens", 0)

        return LLMResponse(
            content=content,
            model=data.get("model", self._model),
            usage=usage,
            latency_ms=latency,
            raw=data,
        )

    def extract_json(self, text: str, max_repair: int | None = None) -> dict[str, Any]:
        """Extract JSON from LLM response (handles markdown fences, trailing commas)."""
        if max_repair is None:
            max_repair = self._cfg.get("max_json_repair_attempts", 2)

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Extract from markdown code fence
        patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        ]
        for pat in patterns:
            match = re.search(pat, text, re.DOTALL)
            if match:
                candidate = match.group(1) if match.lastindex else match.group(0)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Try repair: remove trailing commas
                    repaired = re.sub(r',\s*([}\]])', r'\1', candidate)
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        continue

        raise LLMError(f"Could not extract JSON from LLM response: {text[:200]}...")

    def extract_code(self, text: str) -> str:
        """Extract Python code from LLM response."""
        # Try markdown code fence
        match = re.search(r'```python\s*\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r'```\s*\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # If no fence, return the whole thing (might be raw code)
        return text.strip()
