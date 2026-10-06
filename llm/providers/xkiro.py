"""
llm/providers/xkiro.py – XKiro Provider Integration for Qwen 3.8 Max.

Guarantees (Sections 16, 61, 62):
1. Clean provider abstraction with generic `generate(...)` and `chat(...)` interface.
2. Configured strictly via environment variables (XKIRO_API_KEY, XKIRO_BASE_URL, XKIRO_MODEL).
3. Resilient retry engine with exponential backoff for 429, 500, 502, 503, and timeouts.
4. Cleans reasoning blocks (<think>...</think>) from Qwen model output.
5. Strict token accounting and latency recording.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import requests

_log = logging.getLogger("quantforge.llm.xkiro")


@dataclass
class XKiroResponse:
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = "stop"


class XKiroAPIError(RuntimeError):
    """Raised when XKiro API call fails permanently."""
    pass


class XKiroRateLimitError(XKiroAPIError):
    """Raised when quota or rate limits are exhausted."""
    pass


def clean_qwen_output(text: str) -> str:
    """Strip Qwen reasoning thinking blocks (<think>...</think>)."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


class XKiroProvider:
    """Autonomous Provider Client for Qwen 3.8 Max via XKiro API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: int = 60,
        max_retries: int = 4,
    ):
        self.api_key = (
            api_key
            or os.environ.get("XKIRO_API_KEY")
            or os.environ.get("LLM_API_KEY_1")
            or ""
        )
        raw_url = (
            base_url
            or os.environ.get("XKIRO_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or "https://api.xkiro.com/v1"
        ).rstrip("/")

        if not raw_url.endswith("/chat/completions"):
            self.endpoint = f"{raw_url}/chat/completions"
        else:
            self.endpoint = raw_url

        self.model = (
            model
            or os.environ.get("XKIRO_MODEL")
            or os.environ.get("LLM_MODEL")
            or "qwen-3.8-max"
        )
        self.timeout = timeout_seconds
        self.max_retries = max_retries

    def strip_reasoning(self, text: str) -> str:
        """Strip internal reasoning thinking blocks (<think>...)."""
        return clean_qwen_output(text)

        # Telemetry
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency_ms = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> XKiroResponse:
        """Execute chat completion request with exponential backoff and error recovery."""
        if not self.api_key:
            raise XKiroAPIError(
                "XKIRO_API_KEY (or LLM_API_KEY_1) is not configured in environment."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Elvaris-QuantEngine-V3/XKiroClient",
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        payload.update(kwargs)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise XKiroAPIError("XKiro API returned empty choices list.")

                    choice = choices[0]
                    message = choice.get("message", {})
                    raw_content = message.get("content", "") or ""
                    cleaned = clean_qwen_output(raw_content)

                    usage = data.get("usage", {})
                    p_tok = int(usage.get("prompt_tokens", 0))
                    c_tok = int(usage.get("completion_tokens", 0))

                    # Track telemetry
                    self.total_calls += 1
                    self.total_prompt_tokens += p_tok
                    self.total_completion_tokens += c_tok
                    self.total_latency_ms += latency_ms

                    return XKiroResponse(
                        content=cleaned,
                        model=data.get("model", self.model),
                        usage={
                            "prompt_tokens": p_tok,
                            "completion_tokens": c_tok,
                            "total_tokens": p_tok + c_tok,
                        },
                        latency_ms=latency_ms,
                        raw=data,
                        finish_reason=choice.get("finish_reason", "stop"),
                    )

                elif resp.status_code == 429:
                    retry_wait = 2 ** attempt
                    _log.warning(
                        "XKiro API 429 rate limit hit. Backing off %d seconds (attempt %d/%d)",
                        retry_wait, attempt, self.max_retries
                    )
                    time.sleep(retry_wait)
                    last_error = XKiroRateLimitError(f"XKiro Rate limit exceeded: {resp.text}")

                elif resp.status_code in (500, 502, 503, 504):
                    retry_wait = 2 ** attempt
                    _log.warning(
                        "XKiro API %d server error. Backing off %d seconds (attempt %d/%d)",
                        resp.status_code, retry_wait, attempt, self.max_retries
                    )
                    time.sleep(retry_wait)
                    last_error = XKiroAPIError(f"XKiro Server error {resp.status_code}: {resp.text}")

                elif resp.status_code in (401, 403):
                    raise XKiroAPIError(f"XKiro Authentication failed ({resp.status_code}). Check API key.")

                else:
                    raise XKiroAPIError(f"XKiro API error {resp.status_code}: {resp.text}")

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                retry_wait = 2 ** attempt
                _log.warning(
                    "XKiro network exception: %s. Backing off %d seconds (attempt %d/%d)",
                    exc, retry_wait, attempt, self.max_retries
                )
                time.sleep(retry_wait)
                last_error = XKiroAPIError(f"Network error communicating with XKiro: {exc}")

        raise last_error or XKiroAPIError("XKiro API request exhausted maximum retry attempts.")

    def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> XKiroResponse:
        """Convenience method to generate text from a single prompt string."""
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


def get_xkiro_provider(cfg: Optional[dict] = None) -> XKiroProvider:
    """Factory function to build XKiroProvider with environment/config fallback."""
    cfg_llm = (cfg or {}).get("llm", {})
    return XKiroProvider(
        api_key=os.environ.get("XKIRO_API_KEY") or os.environ.get("LLM_API_KEY_1"),
        base_url=os.environ.get("XKIRO_BASE_URL") or cfg_llm.get("base_url"),
        model=os.environ.get("XKIRO_MODEL") or cfg_llm.get("model", "qwen-3.8-max"),
    )
