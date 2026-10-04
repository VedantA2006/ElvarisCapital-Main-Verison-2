"""
llm/prompts.py – Jinja2 prompt templates and builders for autonomous strategy discovery.

Includes:
- Ideation prompt with core.indicators library catalog, past logic summaries, and coverage targeting.
- Code generation prompt enforcing Strategy contract (PARAMS, __init__(self, params), on_bar).
- Sanitized code fix prompt with line-numbered source and truncated tracebacks (BT-12).
- Lookahead fix prompt with diverging bar/field diagnostics.
- Improve loop prompt with train-only diagnostics for targeted structural refinement (LLM-3).
- JSON repair prompt with Pydantic validation error feedback.
"""

from __future__ import annotations

import re
from typing import Any
from jinja2 import Template

SYSTEM_PROMPT = """You are a senior quantitative researcher specializing in XAUUSD (gold) automated strategy design.
Your task is to invent robust, institutional-grade strategies grounded in market microstructure and statistical edges.

CORE ARCHITECTURE & CONTRACT RULES:
1. Every strategy is executed in a process-isolated sandbox.
2. The strategy receives closed OHLCV bars incrementally via `on_bar(self, bars: pd.DataFrame)`.
3. Return `Signal.enter_long(sl_distance=..., tp_distance=...)`, `Signal.enter_short(...)`, `Signal.close()`, or `None`.
4. DISTANCES: `sl_distance` and `tp_distance` must be POSITIVE USD point distances (e.g. 5.0 for $5.00/oz), never absolute prices.
5. Signal and Direction are pre-imported in your namespace. Do not re-import them.
6. TUNABLE PARAMETERS: Hard cap of at most 6 parameters. Declare `PARAMS = {name: {"default": ..., "min": ..., "max": ...}}`.
7. `__init__(self, params)`: Strategies must accept the `params` dictionary in `__init__`.
8. REUSABLE HELPERS: Use tested leak-safe indicators from `core.indicators` (e.g. `sma`, `ema`, `atr`, `supertrend`, `session_range`, `htf`, `swing_high`, `fvg`, `bos`, etc.).
9. NO LOOKAHEAD: Never use negative shifts, forward indexing, or unconfirmed pivots. All swings require right-side confirmation.
"""

INDICATOR_LIBRARY_CATALOG = """
Available Verified Leak-Free Helpers (`from core.indicators import ...`):
- Trend/Momentum: sma(series, n), ema(series, n), wma(series, n), macd(series, fast, slow, signal), adx(h, l, c, n), supertrend(h, l, c, n, mult), donchian(h, l, n), keltner(h, l, c, ema_n, atr_n, mult), roc(series, n)
- Volatility/Reversion: rsi(series, n), stoch(h, l, c, k, d, slowing), bollinger(series, n, std), zscore(series, n), atr(h, l, c, n), true_range(h, l, c), realized_vol(c, n), vol_percentile(h, l, c, atr_n, lookback), squeeze(h, l, c, bb_n, bb_std, kc_n, kc_mult)
- Volume/Location: session_vwap(df, start_utc), anchored_vwap(df, mask), previous_day_hl(df), previous_week_hl(df), session_range(df, name), opening_range(df, duration_bars)
- Time Features: hour_utc(df), dow(df), is_session(df, name), minutes_since_session_open(df, name), month_end_flag(df, days_before)
- Closed-Bar HTF: htf(df, "4h", func) - resamples lower timeframe using strictly closed 4h candles
- Confirmed Swings: swing_high(h, n), swing_low(l, n) - known strictly n bars after occurrence
- Market Structure: bos(h, l, swing_n), choch(h, l, swing_n), order_block(df, swing_n), fvg(df, min_gap_usd), liquidity_sweep(df, swing_n), premium_discount(df, swing_n)
- Regimes: trend_range_regime(h, l, c, adx_n, adx_thresh), vol_regime(h, l, c, atr_n, lookback), session_regime(df)
"""

_IDEATE_USER_TEMPLATE = Template("""
Invent a NEW, institutional-grade trading strategy for XAUUSD on the {{ timeframe }} timeframe.

{% if target_cell %}
TARGET RESEARCH FOCUS (Coverage Map Target):
- Concept Family: {{ target_cell.get('concept_family', 'any') }}
- Target Session: {{ target_cell.get('session', 'any') }}
- Regime Bias: {{ target_cell.get('regime_bias', 'any') }}
{% endif %}

{{ indicator_catalog }}

PAST ACCEPTED IDEAS (Build on diverse concepts, do not duplicate):
{% if past_accepted %}
{% for idea in past_accepted[:30] %}
- {{ idea.get('name', 'Strat') }} ({{ idea.get('concept_family', 'trend') }}): {{ idea.get('logic_summary', idea.get('hypothesis', '')) }}
{% endfor %}
{% else %}
- None yet. You are discovering the initial candidate strategies.
{% endif %}

PAST REJECTED IDEAS & FAILURE MODES (Avoid these mechanisms):
{% if past_rejected %}
{% for fail in past_rejected[:30] %}
- {{ fail.get('name', 'Strat') }} ({{ fail.get('concept_family', 'trend') }}): {{ fail.get('reason', fail.get('logic_summary', 'Failed robustness gates')) }}
{% endfor %}
{% else %}
- None recorded yet.
{% endif %}

OUTPUT FORMAT:
Return a single JSON object with `spec` and `code`:
{
  "spec": {
    "name": "DescriptiveName",
    "timeframe": "{{ timeframe }}",
    "hypothesis": "Detailed explanation of WHY this edge exists in gold and WHO is on the losing side of this trade.",
    "concept_family": "{{ target_cell.get('concept_family', 'trend') if target_cell else 'trend' }}",
    "indicators_used": ["helper1", "helper2"],
    "entry_logic": "Precise rule for long/short entry.",
    "exit_logic": "Stop loss distance and take profit / trailing distance.",
    "filters": ["session or regime condition"],
    "session_filter": "london_ny",
    "direction": "both",
    "parameters": {
      "param1": {"default": 14, "min": 5, "max": 50}
    },
    "expected_trades_per_year": 60,
    "expected_failure_conditions": "Protracted low-volatility summer chop"
  },
  "code": "class Strategy:\\n    PARAMS = {...}\\n    def __init__(self, params):\\n        ...\\n    def on_bar(self, bars):\\n        ..."
}
""")

_CODE_FIX_TEMPLATE = Template("""
The following strategy encountered an execution error. Fix the code so it runs reliably.

ERROR TRACEBACK:
{{ sanitized_error }}

SOURCE CODE (with line numbers):
```python
{{ numbered_code }}
```

FIX REQUIREMENTS:
1. Fix the error while maintaining the same economic hypothesis and strategy concept.
2. Return a class `Strategy` with `PARAMS`, `__init__(self, params)`, and `on_bar(self, bars)`.
3. Use only numpy, pandas, math, and core.indicators helpers.
4. Keep the parameter cap <= 6.

Return the complete revised strategy JSON:
{
  "spec": {{ spec_json }},
  "code": "class Strategy:\\n    ..."
}
""")

_IMPROVE_TEMPLATE = Template("""
Analyze the in-sample (train) backtest performance of strategy '{{ spec.name }}' and propose ONE targeted structural improvement.

CURRENT STRATEGY SPECIFICATION:
- Hypothesis: {{ spec.hypothesis }}
- Concept Family: {{ spec.concept_family }}
- Current Parameters: {{ spec.parameters }}
- Entry Logic: {{ spec.entry_logic }}
- Exit Logic: {{ spec.exit_logic }}

TRAIN-ONLY PERFORMANCE DIAGNOSTICS:
{{ diagnostics }}

REFINEMENT RULES:
1. Propose ONE targeted STRUCTURAL change (e.g. add a session filter, volatility regime filter, trailing stop, or confirmation rule).
2. DO NOT perform blanket parameter tuning; keep existing parameters or adjust only with structural rationale.
3. Max 6 tunable parameters hard cap.
4. State the exact reason for this refinement.

OUTPUT FORMAT:
Return a JSON object:
{
  "spec": { ... updated spec ... },
  "code": "class Strategy:\\n    ...",
  "refinement_reason": "Detailed explanation of why this structural improvement directly addresses the diagnosed weakness."
}
""")


def _sanitize_traceback(tb: str) -> str:
    """Strip local filesystem paths and keep only line numbers and exception details."""
    if not tb:
        return ""
    # Strip paths like File "c:\...\file.py", line X -> File "strategy.py", line X
    sanitized = re.sub(r'File ".*[\\/]([^\\/]+)",', r'File "\1",', tb)
    return sanitized.strip()


def _add_line_numbers(code: str) -> str:
    """Format code with 1-indexed line numbers."""
    lines = code.splitlines()
    return "\n".join(f"{i + 1:3d}: {line}" for i, line in enumerate(lines))


def render_ideation_prompt(
    timeframe: str,
    target_cell: dict[str, Any] | None = None,
    past_accepted: list[dict[str, Any]] | None = None,
    past_rejected: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Render full Jinja2 prompt for strategy ideation."""
    user_content = _IDEATE_USER_TEMPLATE.render(
        timeframe=timeframe,
        target_cell=target_cell or {},
        indicator_catalog=INDICATOR_LIBRARY_CATALOG,
        past_accepted=past_accepted or [],
        past_rejected=past_rejected or [],
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content.strip()},
    ]


def render_code_fix_prompt(
    code: str,
    error: str,
    spec: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Render prompt to repair code exception with clean traceback."""
    sanitized = _sanitize_traceback(error)
    numbered = _add_line_numbers(code)
    import json
    spec_str = json.dumps(spec or {}, indent=2)

    user_content = _CODE_FIX_TEMPLATE.render(
        sanitized_error=sanitized,
        numbered_code=numbered,
        spec_json=spec_str,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content.strip()},
    ]


def render_improve_prompt(
    spec: Any,
    code: str,
    diagnostics: str,
) -> list[dict[str, str]]:
    """Render prompt to structurally improve a candidate strategy using train diagnostics."""
    user_content = _IMPROVE_TEMPLATE.render(
        spec=spec,
        code=code,
        diagnostics=diagnostics,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content.strip()},
    ]


# ─── Backward-Compatible Wrappers for Legacy Orchestrator Tests ───────────────

def ideation_prompt(timeframe: str, past_ideas: list[str] | None = None,
                    past_failures: list[str] | None = None) -> list[dict[str, str]]:
    """Legacy compatibility wrapper."""
    accepted = [{"name": p, "concept_family": "trend", "logic_summary": p} for p in (past_ideas or [])]
    rejected = [{"name": f, "concept_family": "other", "reason": f} for f in (past_failures or [])]
    return render_ideation_prompt(timeframe=timeframe, past_accepted=accepted, past_rejected=rejected)


def code_generation_prompt(idea: dict, timeframe: str) -> list[dict[str, str]]:
    """Legacy compatibility wrapper for code generation."""
    user = f"""Write Python code implementing this trading strategy:
Strategy: {idea.get('name', 'unnamed')}
Rationale: {idea.get('rationale', '')}
Mechanism: {idea.get('mechanism', '')}
Timeframe: {timeframe}
Parameters: {idea.get('parameters', {})}

STRICT RULES:
1. Define class Strategy with __init__(self, params) and on_bar(self, bars).
2. NEVER use shift(-1), iloc[i+1], or any forward-looking operations.
3. Return ONLY valid Python code inside a ```python``` block."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def code_fix_prompt(code: str, error: str) -> list[dict[str, str]]:
    """Legacy compatibility wrapper for code fix."""
    return render_code_fix_prompt(code=code, error=error)


def lookahead_fix_prompt(code: str, violations: list[str]) -> list[dict[str, str]]:
    """Legacy compatibility wrapper for lookahead fix."""
    user = f"""The following strategy code violated lookahead rules:
Violations: {violations}
Code:
```python
{code}
```
Fix the code so that it never peeks into future data. Return the fixed code inside a ```python``` block."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
