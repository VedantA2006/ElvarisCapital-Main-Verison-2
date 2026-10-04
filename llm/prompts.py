"""
llm/prompts.py – Prompt templates for strategy ideation and code generation.

All prompts enforce the no-lookahead contract and the StrategyProtocol interface.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a senior quantitative researcher specializing in XAUUSD (gold) trading.
You invent trading strategies that are:
- Based on well-understood market microstructure or statistical phenomena
- Robust across different market regimes (trending, ranging, volatile, quiet)
- NOT curve-fitted to specific historical patterns
- Simple enough to explain in one paragraph

CRITICAL RULES:
1. Your strategy code MUST define a class called `Strategy` with an `on_bar(self, bars)` method.
2. `bars` is a pandas DataFrame with columns: timestamp, open, high, low, close, volume, session.
3. `bars` contains ONLY closed bars up to and including the current bar. You CANNOT see future bars.
4. Return `Signal(direction=Direction.LONG, stop_loss=..., take_profit=...)` or `None`.
5. Signal and Direction are pre-imported. Do NOT import them.
6. You may ONLY import: numpy, pandas, math. Nothing else.
7. NEVER use `.shift(-N)` (negative shift), `iloc[i+1]` (forward index), or reversed rolling.
8. All indicators must use `.shift(1)` or positive lookback only.
9. Keep code under 5000 characters. Max 6 tunable parameters.
10. Always include a stop-loss. Strategies without risk management are rejected."""


def ideation_prompt(timeframe: str, past_ideas: list[str] | None = None,
                    past_failures: list[str] | None = None) -> list[dict[str, str]]:
    """Generate a prompt for strategy ideation."""
    user = f"""Invent a NEW trading strategy for XAUUSD on the {timeframe} timeframe.

Requirements:
- Must be fundamentally different from these already-tried ideas: {past_ideas or ['none yet']}
- These approaches FAILED and should be avoided or improved: {past_failures or ['none yet']}
- The strategy should have a clear economic rationale (why does this edge exist?)
- It should work across trending AND ranging markets
- Minimum 30 trades per year expected

Respond with a JSON object:
{{
    "name": "short descriptive name",
    "rationale": "1-2 sentences explaining WHY this edge exists in gold markets",
    "mechanism": "how the strategy generates signals (specific indicators, conditions)",
    "timeframe": "{timeframe}",
    "expected_trades_per_year": <number>,
    "parameters": {{"param_name": default_value, ...}},
    "risk_management": "how SL/TP are set"
}}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def code_generation_prompt(idea: dict, timeframe: str) -> list[dict[str, str]]:
    """Generate a prompt to produce strategy code from an idea."""
    user = f"""Write Python code implementing this trading strategy:

Strategy: {idea.get('name', 'unnamed')}
Rationale: {idea.get('rationale', '')}
Mechanism: {idea.get('mechanism', '')}
Timeframe: {timeframe}
Parameters: {idea.get('parameters', {})}
Risk Management: {idea.get('risk_management', '')}

STRICT REQUIREMENTS:
1. Define a class `Strategy` with `__init__(self)` and `on_bar(self, bars: pd.DataFrame)`.
2. `on_bar` receives a DataFrame of CLOSED bars (columns: timestamp, open, high, low, close, volume, session).
3. Return `Signal(direction=Direction.LONG/SHORT, stop_loss=<price>, take_profit=<price>)` or `None`.
4. Signal, Direction are already available — do NOT import them.
5. Only import numpy, pandas, or math. No other imports.
6. Use `.shift(1)` for previous bar, `.rolling(N).mean()` for moving averages, etc.
7. NEVER use `.shift(-1)`, `iloc[i+1]`, or any forward-looking operation.
8. Keep code under 5000 characters.
9. Include a stop-loss on every signal.

Return ONLY the Python code inside a ```python``` block. No explanation needed."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def code_fix_prompt(code: str, error: str) -> list[dict[str, str]]:
    """Prompt to fix a code error."""
    user = f"""The following strategy code has an error. Fix it.

ERROR: {error}

CODE:
```python
{code}
```

Return the FIXED code inside a ```python``` block. Keep the same strategy logic.
Remember:
- Class must be named `Strategy` with `on_bar(self, bars)` method
- Signal, Direction are pre-available — do NOT import them
- Only numpy, pandas, math imports allowed
- No forward-looking operations (shift(-N), iloc[i+1])"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def lookahead_fix_prompt(code: str, violations: list[dict]) -> list[dict[str, str]]:
    """Prompt to fix lookahead violations."""
    v_str = "\n".join(f"  Line {v['line']}: {v['detail']}" for v in violations[:5])
    user = f"""The following strategy code has LOOKAHEAD BIAS violations:

{v_str}

CODE:
```python
{code}
```

Fix ALL lookahead violations. Rules:
- Use `.shift(1)` instead of `.shift(-1)`
- Use `bars['close'].iloc[-1]` (current bar) not `bars['close'].iloc[-1 + N]`
- All indicators must look BACKWARD only
- Return the fixed code in a ```python``` block."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
