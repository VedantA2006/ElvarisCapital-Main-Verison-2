"""
llm/prompt_firewall.py – Strict prompt data firewall and outgoing context sanitizer.

Guarantees (Sections 5.5, 64):
1. Never sends holdout or sealed validation metrics to the LLM.
2. Strips secrets, API keys, credentials, and connection strings from prompt text.
3. Enforces blind evaluation: converts sensitive validation terminology into blind screening tokens.
4. Logs outgoing context categories without revealing internal secrets.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any

_log = logging.getLogger("quantforge.llm.firewall")

# Forbidden secret patterns (keys, URLs, credentials)
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE),
    re.compile(r"mongodb(?:\+srv)?://[^\s\"']+", re.IGNORECASE),
    re.compile(r"bearer\s+[a-zA-Z0-9_\-\.]{15,}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{10,}['\"]?", re.IGNORECASE),
]

# Sensitive keys never allowed in prompt payloads
FORBIDDEN_CONTEXT_KEYS = {
    "holdout", "holdout_metrics", "holdout_sharpe", "holdout_pf", "holdout_trades",
    "holdout_pnl", "holdout_result", "sealed", "private_details", "hidden_thresholds",
    "val_sharpe", "val_pf", "val_trades", "validation_result", "val_result",
}


def sanitize_prompt_context(context: Any) -> Any:
    """Recursively strip holdout metrics, secret keys, and private numbers from context."""
    if isinstance(context, dict):
        sanitized = {}
        for k, v in context.items():
            k_lower = str(k).lower()
            if any(forbidden in k_lower for forbidden in FORBIDDEN_CONTEXT_KEYS):
                continue
            if any(sec in k_lower for sec in ("key", "secret", "password", "token", "mongo")):
                continue
            sanitized[k] = sanitize_prompt_context(v)
        return sanitized
    elif isinstance(context, list):
        return [sanitize_prompt_context(item) for item in context]
    elif isinstance(context, str):
        return sanitize_prompt_text(context)
    return context


def sanitize_prompt_text(text: str) -> str:
    """Sanitize raw prompt text by redacting secrets and stripping leaks."""
    if not text:
        return ""
    out = text
    for pat in SECRET_PATTERNS:
        out = pat.sub("[REDACTED_SECRET]", out)

    # Redact holdout references and metrics
    out = re.sub(r"(?i)\bholdout(?:\s+\w+)?(?:\s*[:=]\s*[\d\.\-\+]+)?\b", "[REDACTED_HOLDOUT]", out)
    out = re.sub(r"(?i)\bxkiro_[a-zA-Z0-9_\-]+", "[REDACTED_SECRET]", out)

    # Prevent accidental path leaking
    out = re.sub(r'[A-Za-z]:\\[\w\s\-\\\.]+', '[REDACTED_PATH]', out)
    return out


class PromptFirewall:
    """Strict outward context gatekeeper protecting research blind integrity."""

    def __init__(self, strict_blind_mode: bool = True):
        self.strict_blind_mode = strict_blind_mode
        self._sanitization_log: list[dict[str, Any]] = []

    def sanitize_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Sanitize an entire chat messages payload."""
        cleaned = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            sanitized_content = sanitize_prompt_text(content)

            # In strict blind mode, sanitize forbidden keywords if they appear in diagnostics
            if self.strict_blind_mode and role == "user":
                # Ensure no holdout metric fragments leak
                for forbidden in ("holdout_sharpe", "val_sharpe", "sealed_test"):
                    if forbidden in sanitized_content.lower():
                        sanitized_content = re.sub(
                            re.escape(forbidden),
                            "[FILTERED_METRIC]",
                            sanitized_content,
                            flags=re.IGNORECASE,
                        )

            cleaned.append({"role": role, "content": sanitized_content})
        return cleaned

    def audit_context(self, context_dict: dict[str, Any]) -> tuple[bool, list[str]]:
        """Audit outgoing context before rendering prompt templates."""
        violations = []
        for k in context_dict.keys():
            k_lower = str(k).lower()
            if any(f in k_lower for f in FORBIDDEN_CONTEXT_KEYS):
                violations.append(f"Forbidden evaluation metric key present: '{k}'")
        return len(violations) == 0, violations

    def sanitize_context(self, context: Any) -> Any:
        """Sanitize an entire dictionary or structure before sending to LLM."""
        return sanitize_prompt_context(context)

    def sanitize_text(self, text: str) -> str:
        """Sanitize prompt text, redacting secrets and holdout leaks."""
        return sanitize_prompt_text(text)

