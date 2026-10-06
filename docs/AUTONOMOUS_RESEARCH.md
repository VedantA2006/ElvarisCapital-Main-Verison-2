# ELVARIS CAPITAL — AUTONOMOUS RESEARCH SPECIFICATION (V3)

---

## 1. Autonomous Research Lab Loop

The primary operational paradigm of QuantForge V3 is an autonomous scientific cycle rather than a prompt-response chatbot. The laboratory loop operates continuously according to the following cycle:

```
[1] DIRECTOR SELECTS RESEARCH VECTOR (Family, Timeframe, Budget)
    │
[2] HYPOTHESIS FORMULATION (Economic / microstructure rationale)
    │
[3] STRATEGY DSL GENERATION (Qwen 3.8 Max via XKiro)
    │
[4] DSL VALIDATION & COMPILATION (Deterministic AST compilation)
    │
[5] SECURITY & LOOKAHEAD SCAN (Gate 2 Static Policy Check)
    │
[6] DEDUPLICATION CHECK (L1 Exact, L2 Logical, L3 Behavioral)
    │
[7] EVENT-BASED BACKTEST (Full transaction costs & tick simulation)
    │
[8] STRATEGY AUTOPSY (18-class failure taxonomy analysis)
    │
    ├─ If FAIL_FATAL or Non-Repairable: Abandon Lineage, Log Learning
    │
    ├─ If FAIL_REPAIRABLE: Send Sanitized Failure Report to Qwen
    │   └─ Generate 3 Logically Distinct Structural Improvements ─► Retest
    │
    └─ If PASS: Route to Adversarial Robustness Validation Suite
```

---

## 2. Research Director Decision Policy (Section 92)

The `ResearchDirector` (`llm/research_director.py`) is the autonomous top-level governor. At each decision point, it evaluates stored memory and chooses one of eight deterministic actions:

1. `NEW_HYPOTHESIS`: Explores an under-represented concept family (e.g., Market Structure, Liquidity) to maintain the target diversity distribution.
2. `IMPROVE_EXISTING`: If a strategy demonstrates positive gross expectancy but fails on cost drag or slight profit factor shortfall, directs Qwen to refine execution asymmetry.
3. `EXPLORE_NEW_FAMILY`: Shifts research focus when a concept family has been saturated.
4. `EXPLORE_NEW_TIMEFRAME`: Migrates promising concepts across timeframes (e.g., from 1h structure to 15m entry).
5. `EVOLVE_PROMISING_STRATEGY`: Dispatches candidates to the genetic evolution engine for mutation and crossover.
6. `ABANDON_LINEAGE`: Terminates improvement attempts when lineage depth limit (default: 4) or consecutive failures occur.
7. `RUN_ROBUSTNESS`: Dispatches candidates meeting initial thresholds (PF $\ge 1.25$, Sharpe $\ge 0.80$) to Walk-Forward, Monte Carlo, and Cost Stress gates.
8. `FREEZE_CANDIDATE`: Locks an invariant candidate hash and registers it in the Strategy Library for holdout evaluation.

---

## 3. Failure Autopsy & 18-Class Taxonomy (Sections 24, 25)

Every backtest evaluation produces a standardized `StrategyAutopsy` (`llm/failure_analyzer.py`) categorizing root causes:

| Category | Classification | Autonomous Action |
|:---|:---|:---|
| `NO_TRADES` | Fatal | Discard; entry conditions over-constrained |
| `TOO_FEW_TRADES` | Repairable | Relax secondary filter; test lower timeframe |
| `NEGATIVE_EXPECTANCY` | Fatal / Unfavorable | Abandon lineage; hypothesis lacked gross edge |
| `LOW_PROFIT_FACTOR` | Repairable | Widen risk-reward ratio ($\ge 2:1$); refine entry trigger |
| `LOW_SHARPE` | Repairable | Add trend or volatility filter to eliminate choppy periods |
| `HIGH_DRAWDOWN` | Repairable | Tighten ATR stop-loss multiplier; add daily loss limit |
| `COST_SENSITIVE` | Repairable | Widen holding horizon; target $\ge 15$-point price moves |
| `PARAMETER_FRAGILE` | Fatal | Discard; overfit parameter cliff detected |
| `REGIME_FRAGILE` | Repairable | Introduce macro volatility / regime filter |
| `LOOKAHEAD` | Fatal Security | Reject permanently; blacklist AST pattern |
| `REPAINTING` | Fatal Security | Reject permanently; signals altered post-hoc |
| `NON_DETERMINISTIC` | Fatal Engine | Reject permanently; non-reproducible run |
| `SANDBOX_VIOLATION` | Fatal Security | Reject permanently; attempted banned syscall |
| `RUNTIME_ERROR` | Repairable Code | Repair AST or data column alignment |
| `DUPLICATE` | Redundant | Discard; $>90\%$ behavioral or exact match |

---

## 4. Autonomous Control & Crash Recovery (Sections 80, 81)

The autonomous runner (`engine/autonomous_runner.py`) provides full session control:
- **`start`**: Launches continuous research cycles.
- **`pause`**: Gracefully finishes current backtest/autopsy and pauses execution without corrupting state.
- **`resume`**: Restores active session state from MongoDB (`engine_state`).
- **`stop`**: Shuts down workers gracefully.
- **Crash Recovery**: Every experiment, lineage branch, and LLM call is stored with atomic timestamps. Upon process restart, the supervisor resumes from the exact lineage state without repeating completed evaluations.
