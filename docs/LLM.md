# ELVARIS CAPITAL — LLM INTEGRATION & REASONING SPECIFICATION (V3)

---

## 1. Provider & Model Configuration (Section 16)

QuantForge V3 standardizes on **Qwen 3.8 Max** served through the **XKiro API**:

```bash
# Environment Configuration (.env or shell environment)
XKIRO_API_KEY="your-xkiro-api-key"
XKIRO_BASE_URL="https://api.xkiro.com/v1"
XKIRO_MODEL="qwen-3.8-max"
```

The system strictly resolves credentials from environment variables and key management stores (`llm/key_manager.py`). No API keys are hard-coded in the repository.

---

## 2. Specialized Cognitive Roles (Section 17)

QuantForge does not rely on a monolithic general-purpose prompt. Instead, Qwen is orchestrated across 7 specialized cognitive roles:

1. **Research Director**: Analyzes memory and selects the next strategic family (e.g. Market Structure, Liquidity) and timeframe.
2. **Hypothesis Generator**: Formulates institutional microstructure hypotheses explaining *why* an edge exists.
3. **Strategy Architect**: Translates qualitative hypotheses into declarative `StrategyDSL` specifications.
4. **Failure Analyst**: Interprets `StrategyAutopsy` records to explain the mechanics of strategy breakdown.
5. **Strategy Improver**: Proposes exactly 3 logically distinct structural improvements (avoiding brute-force parameter bumping).
6. **Robustness Analyst**: Evaluates Walk-Forward and Monte Carlo results to identify overfit regimes.
7. **Portfolio Researcher**: Reviews cross-strategy correlation and suggests complimentary diversifiers.

---

## 3. Resilient API Engine (`llm/providers/xkiro.py`)

The XKiro client includes production fault-tolerance:
- **Exponential Backoff**: Jittered retry across transient HTTP errors (429 Rate Limit, 500 Server Error, 502/503 Bad Gateway, timeouts).
- **Thinking Block Stripping**: Automatically removes internal reasoning blocks (`<think>...</think>`), returning clean structured output.
- **Failover Key Rotation**: Rotates through backup API keys upon encountering account exhaustion.
- **Token Telemetry**: Logs prompt tokens, completion tokens, latency (ms), and call purpose in the `llm_calls` collection.

---

## 4. LLM Role Telemetry Counters (Sections 59, 98)

To ensure the engine does not collapse into an ideation-only loop, telemetry is explicitly tracked and displayed across 5 distinct operational categories:

- **Ideas**: Hypotheses and initial DSL generations
- **Improvements**: Guided structural refinements following autopsy feedback
- **Failure Analyses**: Root-cause diagnostic evaluations
- **Reviews**: AST and logic integrity audits
- **Robustness Analyses**: Sensitivity and stress reviews
