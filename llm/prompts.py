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
10. WARMUP & STOP LOSS SAFETY:
    - At start of `on_bar(self, bars)`: ALWAYS include `if len(bars) < 60: return None`.
    - Indicators (ATR, SMA, etc.) will have NaN values during early bars. Before taking a trade, ALWAYS check: `if np.isnan(atr_val) or atr_val <= 0: return None`.
    - `sl_distance` must be a POSITIVE point distance (e.g., `sl_dist = max(5.0, float(atr_val * self.params["sl_mult"]))`). Never pass NaN or <= 0!
    - If `tp_distance` is used, ensure it is >= 1.50 USD points (or leave as None).
11. TRADE FREQUENCY & SAMPLE SIZE:
    - Institutional validation strictly enforces Gate 7 (Minimum Sample): total trades >= 150 (at least 30-50 trades per year).
    - Ensure your entry criteria are not excessively restrictive. The strategy must actively participate during valid market conditions to generate sufficient trade observations for Monte Carlo and Walk-Forward statistical significance.
"""

INDICATOR_LIBRARY_CATALOG = """
Available Verified Leak-Free Helpers (`from core.indicators import ...`):
All indicators accept `bars` (the DataFrame) as their FIRST argument (positional).
Named parameters use 'period' or 'n' interchangeably. Return types are shown precisely.

CRITICAL API RULES:
1. Pass `bars` (the full DataFrame) as the first argument to ALL indicators.
2. Use the EXACT parameter names shown below. Do NOT guess parameter names.
3. Named tuple / dataclass results must be accessed via ATTRIBUTES (e.g., `result.upper`), NOT index unpacking.
4. Tuple results (bos, choch, fvg, liquidity_sweep) use standard unpacking: `bull, bear = bos(bars)`.

TREND/MOMENTUM:
  sma(bars, period=20) -> pd.Series
  ema(bars, period=20) -> pd.Series
  wma(bars, period=20) -> pd.Series
  roc(bars, period=10) -> pd.Series
  macd(bars, fast_period=12, slow_period=26, signal_period=9) -> MacdResult(macd_line, signal_line, hist)
  adx(bars, period=14) -> AdxResult(adx_line, plus_di, minus_di)
  supertrend(bars, period=10, multiplier=3.0) -> SupertrendResult(st, direction, upper, lower)
  donchian(bars, period=20) -> BandResult(upper, middle, lower)
  keltner(bars, ema_period=20, atr_period=10, multiplier=2.0) -> BandResult(upper, middle, lower)

VOLATILITY/REVERSION:
  rsi(bars, period=14) -> pd.Series
  stoch(bars, k_period=14, d_period=3, slowing=3) -> StochResult(slow_k, slow_d)
  bollinger(bars, period=20, num_std=2.0) -> BollingerResult(upper, middle, lower, bandwidth, pct_b)
  zscore(bars, period=20) -> pd.Series
  atr(bars, period=14) -> pd.Series
  true_range(bars) -> pd.Series
  realized_vol(bars, period=20) -> pd.Series
  vol_percentile(bars, lookback=100) -> pd.Series  (0-100 percentile of ATR)
  squeeze(bars, bb_period=20, bb_std=2.0, kc_period=20, kc_mult=1.5) -> pd.Series  (1=squeeze on, 0=squeeze off)

VOLUME/LOCATION:
  session_vwap(bars) -> pd.Series
  anchored_vwap(bars, anchor_mask) -> pd.Series
  previous_day_hl(bars) -> tuple[pd.Series, pd.Series]  (prev_high, prev_low)
  previous_week_hl(bars) -> tuple[pd.Series, pd.Series]  (prev_high, prev_low)
  session_range(bars, session_name) -> SessionRangeResult(high: pd.Series, low: pd.Series)
  opening_range(bars, duration_bars=1) -> tuple[pd.Series, pd.Series]  (or_high, or_low)

TIME FEATURES:
  hour_utc(bars) -> pd.Series
  dow(bars) -> pd.Series
  is_session(bars, "london"|"ny"|"asia") -> pd.Series[bool]
  minutes_since_session_open(bars) -> pd.Series
  month_end_flag(bars, days_before=2) -> pd.Series[bool]

CLOSED-BAR HTF:
  htf(bars, target_tf="4h", func=lambda df: df["close"]) -> pd.Series

CONFIRMED SWINGS:
  swing_high(bars, n=2) -> SwingResult  (access .pivot_value: pd.Series)
  swing_low(bars, n=2) -> SwingResult  (access .pivot_value: pd.Series)

MARKET STRUCTURE (return plain tuples – use tuple unpacking):
  bos(bars, swing_n=2) -> tuple[pd.Series, pd.Series]  # (bull_bos: bool, bear_bos: bool)
  choch(bars, swing_n=2) -> tuple[pd.Series, pd.Series]  # (bull_choch: bool, bear_choch: bool)
  fvg(bars, min_gap_usd=0.0) -> tuple[pd.Series, pd.Series]  # (bull_fvg: bool, bear_fvg: bool)
  liquidity_sweep(bars, swing_n=2) -> tuple[pd.Series, pd.Series]  # (high_sweep: bool, low_sweep: bool)
  premium_discount(bars, swing_n=5) -> pd.Series[float]  # 0.0-1.0 (<0.5 = discount, >0.5 = premium)
  order_block(bars, swing_n=2) -> tuple[pd.Series, pd.Series]  # (bull_ob, bear_ob)

REGIMES:
  trend_range_regime(high, low, close, adx_period=14, er_period=10) -> pd.Series  # "trend" or "range"
  vol_regime(high, low, close, atr_period=14, lookback=100) -> pd.Series  # "high_vol" or "low_vol"
  session_regime(bars) -> pd.Series

WORKING EXAMPLE (follow this pattern exactly):
```python
class Strategy:
    PARAMS = {
        "ema_fast": {"default": 21, "min": 10, "max": 50},
        "ema_slow": {"default": 55, "min": 30, "max": 100},
        "atr_period": {"default": 14, "min": 7, "max": 21},
        "sl_mult": {"default": 1.5, "min": 1.0, "max": 3.0},
        "tp_mult": {"default": 2.5, "min": 1.5, "max": 4.0},
    }
    def __init__(self, params):
        self.params = params
    def on_bar(self, bars):
        if len(bars) < 60:
            return None
        from core.indicators import ema, atr, rsi, is_session
        fast = ema(bars, period=self.params["ema_fast"])
        slow = ema(bars, period=self.params["ema_slow"])
        atr_val = float(atr(bars, period=self.params["atr_period"]).iloc[-1])
        rsi_val = float(rsi(bars, period=14).iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            return None
        in_session = bool(is_session(bars, "london").iloc[-1] or is_session(bars, "ny").iloc[-1])
        if not in_session:
            return None
        sl_dist = max(5.0, atr_val * self.params["sl_mult"])
        tp_dist = max(7.5, atr_val * self.params["tp_mult"])
        if float(fast.iloc[-1]) > float(slow.iloc[-1]) and float(fast.iloc[-2]) <= float(slow.iloc[-2]) and rsi_val < 70:
            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)
        if float(fast.iloc[-1]) < float(slow.iloc[-1]) and float(fast.iloc[-2]) >= float(slow.iloc[-2]) and rsi_val > 30:
            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)
        return None
```
"""


_IDEATE_USER_TEMPLATE = Template("""
Invent a NEW, institutional-grade trading strategy for XAUUSD on the {{ timeframe }} timeframe.

{% if target_cell %}
TARGET RESEARCH FOCUS (Coverage Map Target):
- Concept Family: {{ target_cell.get('concept_family', 'any') }}
- Target Session: {{ target_cell.get('session', 'any') }}
- Regime Bias: {{ target_cell.get('regime_bias', 'any') }}
{% endif %}

CRITICAL VALIDATION CRITERIA (GATE 7 - MINIMUM SAMPLE SIZE):
- The strategy MUST generate at least 40-80 trades per year (total trades >= 150-200 across the 3-year train dataset).
- Avoid ultra-rare confluence setups (like requiring 4 simultaneous indicators or strict 1-hour windows) that generate fewer than 30 trades per year.
- Use clean, frequent entry triggers (e.g., EMA crossovers, Donchian channel breakouts, RSI momentum pullbacks, or Session range breakouts) with reasonable cooldown (e.g., 2 to 6 bars).

{{ indicator_catalog }}

PAST ACCEPTED IDEAS (name | concept | logic). Build on DIFFERENT concepts, do not duplicate:
{% if past_accepted %}
{% for line in past_accepted %}
- {{ line }}
{% endfor %}
{% else %}
- None yet. You are discovering the initial candidate strategies.
{% endif %}

PAST REJECTED IDEAS (name | concept | rejection category):
{% if past_rejected %}
{% for line in past_rejected %}
- {{ line }}
{% endfor %}
{% else %}
- None recorded yet.
{% endif %}

AVOID THESE LOGICS: do not resubmit any logic listed above (accepted or rejected), even under a new name or with different constants.

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

{{ indicator_catalog }}

FIX REQUIREMENTS:
1. Fix the error while maintaining the same economic hypothesis and strategy concept.
2. Return a class `Strategy` with `PARAMS`, `__init__(self, params)`, and `on_bar(self, bars)`.
3. Use only numpy, pandas, math, and core.indicators helpers.
4. Keep the parameter cap <= 6.
5. Use the EXACT parameter names from the indicator catalog above. Do NOT guess.
6. Named tuple/dataclass results (AdxResult, MacdResult, etc.) must be accessed via attributes (e.g., `result.adx_line`), NOT tuple unpacking.
7. Tuple results (bos, choch, fvg, liquidity_sweep) use standard unpacking: `bull, bear = bos(bars)`.

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

CURRENT SOURCE (line-numbered):
```python
{{ numbered_code }}
```

TRAIN-ONLY PERFORMANCE DIAGNOSTICS:
{{ diagnostics }}

REFINEMENT RULES:
1. Propose ONE targeted STRUCTURAL change (e.g. add a session filter, volatility regime filter, trailing stop, or confirmation rule) tied to a weakness shown above.
2. DO NOT perform blanket parameter tuning; changing constants alone is not an improvement.
3. Never exceed the hard cap of 6 tunable parameters.
4. State the exact reason for this refinement, citing the diagnostic that motivates it.

OUTPUT FORMAT:
Return a JSON object:
{
  "spec": { ... updated spec ... },
  "code": "class Strategy:\\n    ...",
  "refinement_reason": "Detailed explanation of why this structural improvement directly addresses the diagnosed weakness."
}
""")

_RETHINK_TEMPLATE = Template("""
Strategy '{{ spec.name }}' was rejected: {{ public_summary }}.

This is NOT an optimisation target and no numbers are provided. You may attempt ONE structural rethink
that makes the edge more robust (simpler logic, fewer parameters, a mechanism less sensitive to exact
thresholds). Do not tune constants. Never exceed 6 parameters.

CURRENT STRATEGY SPECIFICATION:
- Hypothesis: {{ spec.hypothesis }}
- Entry Logic: {{ spec.entry_logic }}
- Exit Logic: {{ spec.exit_logic }}
- Parameters: {{ spec.parameters }}

CURRENT SOURCE (line-numbered):
```python
{{ numbered_code }}
```

Return a JSON object with "spec", "code" and "refinement_reason".
""")

_LOOKAHEAD_FIX_TEMPLATE = Template("""
The strategy FAILED the lookahead check '{{ check }}'.

- Diverging bar index: {{ bar_index }}{% if timestamp %} ({{ timestamp }}){% endif %}
- Diverging field: {{ field }}
- Rule violated: {{ rule }}

The output for that bar changed when future bars were removed, so the code reads data it could not
have had at decision time. Fix the cause (negative shifts, centred windows, unconfirmed pivots, full-series
normalisation, resampling unfinished higher-timeframe candles) without changing the strategy idea.

SOURCE (line-numbered):
```python
{{ numbered_code }}
```

Return the complete revised strategy JSON with "spec" and "code".
""")

_REPAIR_JSON_TEMPLATE = Template("""
Your previous response failed schema validation:
{{ error }}

Correct the error and output ONLY the complete, valid JSON object matching the required schema.
""")

DEFAULT_MAX_PROMPT_CHARS = 24000
MAX_PAST_IDEAS = 50
_TRACEBACK_FRAMES_KEPT = 3
_TRACEBACK_MAX_CHARS = 2500


def _sanitize_traceback(tb: str) -> str:
    """Keep only the last few frames, strip filesystem paths, cap the length."""
    if not tb:
        return ""
    # File "c:\...\file.py", line X -> File "file.py", line X
    sanitized = re.sub(r'File ".*[\\/]([^\\/]+)",', r'File "\1",', tb)
    # Strip any remaining absolute paths (Windows drive or POSIX root) from messages.
    sanitized = re.sub(r'(?:[A-Za-z]:)?(?:[\\/][\w.\- ]+){2,}[\\/]([\w.\-]+)', r'\1', sanitized)
    lines = sanitized.strip().splitlines()
    frame_starts = [i for i, ln in enumerate(lines) if ln.lstrip().startswith('File "')]
    if len(frame_starts) > _TRACEBACK_FRAMES_KEPT:
        cut = frame_starts[-_TRACEBACK_FRAMES_KEPT]
        head = [lines[0]] if lines and lines[0].startswith("Traceback") else []
        lines = head + ["  ... (earlier frames omitted)"] + lines[cut:]
    out = "\n".join(lines)
    if len(out) > _TRACEBACK_MAX_CHARS:
        out = "... " + out[-_TRACEBACK_MAX_CHARS:]
    return out


sanitize_traceback = _sanitize_traceback


def _add_line_numbers(code: str) -> str:
    """Format code with 1-indexed line numbers."""
    lines = code.splitlines()
    return "\n".join(f"{i + 1:3d}: {line}" for i, line in enumerate(lines))


def _one_line(text: Any, limit: int = 160) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _accepted_line(idea: dict[str, Any]) -> str:
    logic = idea.get("logic_summary") or idea.get("entry_logic") or idea.get("hypothesis", "")
    return f"{idea.get('name', 'Strat')} | {idea.get('concept_family', 'other')} | {_one_line(logic)}"


def _rejected_line(idea: dict[str, Any]) -> str:
    # Category only: never forward gate numbers to the LLM.
    cat = idea.get("category") or _one_line(idea.get("reason", "rejected"), 60)
    return f"{idea.get('name', 'Strat')} | {idea.get('concept_family', 'other')} | {cat}"


def render_ideation_prompt(
    timeframe: str,
    target_cell: dict[str, Any] | None = None,
    past_accepted: list[dict[str, Any]] | None = None,
    past_rejected: list[dict[str, Any]] | None = None,
    max_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> list[dict[str, str]]:
    """Render the ideation prompt within a character budget.

    Shows the most recent MAX_PAST_IDEAS accepted and rejected ideas. When over budget,
    the lowest-priority content is dropped first: oldest accepted ideas, then oldest
    rejected ideas. The contract, rules and helper list are never truncated.
    """
    accepted = [_accepted_line(i) for i in (past_accepted or [])[-MAX_PAST_IDEAS:]]
    rejected = [_rejected_line(i) for i in (past_rejected or [])[-MAX_PAST_IDEAS:]]

    def _render() -> str:
        return _IDEATE_USER_TEMPLATE.render(
            timeframe=timeframe,
            target_cell=target_cell or {},
            indicator_catalog=INDICATOR_LIBRARY_CATALOG,
            past_accepted=accepted,
            past_rejected=rejected,
        ).strip()

    user_content = _render()
    while len(user_content) > max_chars and (accepted or rejected):
        if accepted:
            accepted.pop(0)
        else:
            rejected.pop(0)
        user_content = _render()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
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
    spec_str = json.dumps(spec or {}, indent=2, default=str)

    user_content = _CODE_FIX_TEMPLATE.render(
        sanitized_error=sanitized,
        numbered_code=numbered,
        spec_json=spec_str,
        indicator_catalog=INDICATOR_LIBRARY_CATALOG,
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
        numbered_code=_add_line_numbers(code),
        diagnostics=diagnostics,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content.strip()},
    ]


def render_rethink_prompt(spec: Any, code: str, public_summary: str) -> list[dict[str, str]]:
    """One structural rethink after a robustness failure. Receives the category only."""
    user_content = _RETHINK_TEMPLATE.render(
        spec=spec, numbered_code=_add_line_numbers(code), public_summary=public_summary,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content.strip()},
    ]


def render_lookahead_fix_prompt(
    code: str,
    check: str,
    bar_index: int | None,
    field: str,
    rule: str,
    timestamp: str | None = None,
) -> list[dict[str, str]]:
    """Fix prompt naming the failed check, the diverging bar/field and the rule violated."""
    user_content = _LOOKAHEAD_FIX_TEMPLATE.render(
        check=check, bar_index=bar_index, timestamp=timestamp, field=field, rule=rule,
        numbered_code=_add_line_numbers(code),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content.strip()},
    ]


def render_repair_json_prompt(error: str) -> dict[str, str]:
    """User message asking the model to repair a schema-invalid JSON response."""
    return {"role": "user", "content": _REPAIR_JSON_TEMPLATE.render(error=error[:2000]).strip()}


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
