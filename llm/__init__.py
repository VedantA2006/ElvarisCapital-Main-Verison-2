"""
quantforge.llm – Resilient LLM strategy ideation, code generation, and repair layer.
"""

from llm.client import (
    LLMClient,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMValidationError,
    TruncatedResponse,
    WaitingForQuotaError,
    clean_llm_content,
)
from llm.diagnostics import build_train_diagnostics
from llm.improve_loop import StrategyImproveLoop
from llm.key_manager import AllKeysUnavailableError, KeyManager, KeyStatus
from llm.prompts import (
    render_code_fix_prompt,
    render_ideation_prompt,
    render_improve_prompt,
)
from llm.schemas import (
    ParameterDef,
    StrategyImproveResponse,
    StrategyResponse,
    StrategySpec,
)
from llm.usage import UsageTracker, estimate_tokens

__all__ = [
    "LLMClient",
    "LLMResponse",
    "TruncatedResponse",
    "LLMError",
    "LLMRateLimitError",
    "WaitingForQuotaError",
    "LLMValidationError",
    "clean_llm_content",
    "KeyManager",
    "KeyStatus",
    "AllKeysUnavailableError",
    "UsageTracker",
    "estimate_tokens",
    "StrategySpec",
    "ParameterDef",
    "StrategyResponse",
    "StrategyImproveResponse",
    "StrategyImproveLoop",
    "build_train_diagnostics",
    "render_ideation_prompt",
    "render_code_fix_prompt",
    "render_improve_prompt",
]
