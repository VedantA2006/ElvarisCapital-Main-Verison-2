# ELVARIS CAPITAL — AUTONOMOUS QUANT ENGINE V3 MASTER ARCHITECTURE & UPGRADE PLAN
**Date**: 2026-10-06 | **Target System**: Autonomous XAUUSD Quant Research Laboratory
**Target Model**: Qwen 3.8 Max via XKiro API (`XKIRO_API_KEY`, `XKIRO_BASE_URL`, `XKIRO_MODEL=qwen-3.8-max`)
**Document Status**: COMPREHENSIVE ARCHITECTURE & IMPLEMENTATION ROADMAP

---

## 1. System Philosophy & Executive Architecture

The Elvaris Capital V3 system transforms QuantForge from an iterative script generator into an **autonomous systematic quantitative research laboratory**. The core architectural tenet is:

> **The deterministic quantitative engine maintains absolute authority over the LLM.**
> The LLM acts as the creative hypothesis and code generation intelligence, proposing hypotheses, structured strategies, parameter modifications, and diagnostic interpretations.
> The deterministic engine enforces immutable data splits, blind holdout isolation, sandbox security, lookahead/repainting AST and runtime guards, 19 statistical validation gates, and cost stress accounting. The LLM can never override deterministic gates.

```text
                                  ┌──────────────────────────────┐
                                  │   RESEARCH DIRECTOR (LLM)    │
                                  │   Taxonomy / Family Budget   │
                                  └──────────────┬───────────────┘
                                                 │ Selects hypothesis
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   HYPOTHESIS & DSL ENGINE    │
                                  │   Structured Spec (YAML/JSON)│
                                  └──────────────┬───────────────┘
                                                 │ Compiles
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │     SECURITY & AST SCAN      │
                                  │ Lookahead / Sandbox Guard    │
                                  └──────────────┬───────────────┘
                                                 │ Passes Static Scan
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   EVENT SIMULATOR BACKTEST   │
                                  │   Real Tick Friction Costs   │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   STRATEGY AUTOPSY ENGINE    │
                                  │ Metrics / Failure Reason     │
                                  └──────┬───────────────┬───────┘
                                         │               │
                               FAIL (Repairable)     PASS (Promising)
                                         │               │
                                         ▼               ▼
                        ┌──────────────────┐    ┌─────────────────────────┐
                        │ PROMPT FIREWALL  │    │  ADVERSARIAL ROBUSTNESS │
                        │ Sanitized Report │    │  Walk-Forward / MC / PBO│
                        └────────┬─────────┘    └────────────┬────────────┘
                                 │                           │
                                 ▼                           ▼
                        ┌──────────────────┐    ┌─────────────────────────┐
                        │   QWEN IMPROVE   │    │     FROZEN STRATEGY     │
                        │ Max Depth Clamped│    │   Cryptographic Hash    │
                        └────────┬─────────┘    └────────────┬────────────┘
                                 │                           │
                                 └───────────┐               ▼
                                             │  ┌─────────────────────────┐
                                             │  │  HOLDOUT VAULT (BLIND)  │
                                             │  │ Single Access Permission│
                                             │  └────────────┬────────────┘
                                             │               │
                                             │               ▼
                                             │  ┌─────────────────────────┐
                                             │  │      FORWARD TEST       │
                                             │  │ Paper Trader / Live Feed│
                                             │  └────────────┬────────────┘
                                             │               │
                                             ▼               ▼
                                 ┌────────────────────────────────────────┐
                                 │      STRATEGY LIBRARY & PORTFOLIO      │
                                 │ Correlation / Allocation / Diversify   │
                                 └────────────────────────────────────────┘
```

---

## 2. Current Architecture & Module Inventory

### 2.1 Existing Core Capabilities
| Component | Existing Modules | Capabilities Preserved & Leveraged |
| :--- | :--- | :--- |
| **Data Engine** | `core/data_loader.py`, `core/data_vault.py`, `core/splits.py` | Multi-timeframe CSV ingestion (5m, 15m, 1h, 4h), session tagging (Asia, London, NY), calendar gap analysis, train/validation/holdout partitioning with embargo. |
| **Execution Simulator** | `core/event_simulator.py`, `core/simulator.py`, `core/backtester.py`, `core/sizing.py` | Bar-by-bar chronological execution, spread, slippage, commission, swap, intraday gap handling, SL/TP simulation, ATR-based risk sizing. |
| **Sandbox & Security** | `sandbox/runner.py`, `sandbox/policy.py`, `sandbox/child.py`, `core/sandbox.py` | Subprocess isolation, forbidden imports (`os`, `sys`, `socket`), AST inspection, dunder attribute blocking (`__subclasses__`, `__bases__`), CPU/RAM watchdog. |
| **Lookahead Defense** | `core/lookahead_guard.py` | AST negative index checking (`shift(-N)`, `iloc[j]`), truncation testing across cut points, temporal perturbation testing, bitwise tape determinism check. |
| **Statistical Gates** | `validation/gates.py`, `core/gates.py`, `validation/overfit.py`, `validation/regime.py` | 19 validation gates: smoke run, determinism, truncation, signal validity, minimum sample, basic quality, parameter sensitivity, walk-forward, Monte Carlo, regime/year stability, DSR, PBO, delay test. |
| **Storage & Persistence**| `storage/mongo.py`, `storage/logger.py` | MongoDB collections (`strategies`, `backtests`, `gate_results`, `runs`, `counters`, `leaderboard`, `holdout_access`, `logs`), JSONL structured logging fallback. |
| **Forward & Holdout** | `forward/holdout_runner.py`, `forward/paper_trader.py`, `forward/feeds.py`, `forward/live_ready.py` | Cryptographic data hashing, single-access holdout gate enforcement, CSV bar streaming forward paper trading, live-readiness checklist. |
| **UI & Supervision** | `dashboard/fastapi_app.py`, `engine/supervisor.py`, `engine/worker.py`, `dashboard/tui.py` | FastAPI REST/SSE backend, background daemon worker, state supervisor, Rich terminal monitoring. |

### 2.2 Module Dependency Map
```text
main.py
  ├── core/config.py ─── config.yaml
  ├── core/data_loader.py ─── core/data_vault.py, core/splits.py
  ├── storage/mongo.py ─── pymongo (Atlas / local / mock)
  ├── storage/logger.py ─── JSONL fallback
  ├── engine/supervisor.py & worker.py
  │     ├── llm/orchestrator.py
  │     │     ├── llm/client.py ─── (Qwen 3.8 Max provider)
  │     │     ├── llm/prompts.py ─── (Templates & catalog)
  │     │     ├── sandbox/runner.py ─── sandbox/policy.py ─── sandbox/child.py
  │     │     ├── core/backtester.py ─── core/event_simulator.py
  │     │     │     └── core/indicators.py (100% leak-free indicator suite)
  │     │     ├── validation/gates.py ─── validation/overfit.py, validation/regime.py
  │     │     └── strategy/diversity.py ─── strategy fingerprinting
  │     └── dashboard/fastapi_app.py ─── Web Console & Event Streams
  └── forward/holdout_runner.py & paper_trader.py
```

---

## 3. Detailed Root Causes of Known Baseline Defects

1. **DB-1: Runs vs Trials Persistence Duplication**:
   - `llm/orchestrator.py` `_finish()` inserted each record into `runs` AND mirrored to `trials`.
   - `_stored_run_count()` sums documents in both `runs` and `trials`, resulting in double counting (2,000 recorded vs 1,000 trials).
   - *Fix*: Standardize on `runs` as source of truth. Make trial creation, run record storage, and counter increments atomic and verified. Provide explicit CLI `db verify` and `db repair-counters`.
2. **Hygiene: Swallowed Exceptions**:
   - 12 broad silent `except Exception: pass` blocks in `core/indicators.py`, `llm/orchestrator.py`, `sandbox/child.py`, and `scripts/export_analysis.py`.
   - *Fix*: Classify exceptions into `recoverable`, `fatal`, `validation_failure`, and `system_failure`. Log all recoverable exceptions with structured context and re-raise fatal ones.
3. **UI-Auth: Unauthenticated Endpoint**:
   - In `dashboard/fastapi_app.py`, `_verify_auth()` included: `if is_loopback: return dashboard_token`.
   - This bypassed authentication for all requests coming from `127.0.0.1`, `localhost`, or `testclient`.
   - *Fix*: Remove automatic loopback bypass; require valid `qf_token` cookie or Bearer token header across all endpoints.
4. **Prompt Leakage & System Prompt Substrings**:
   - `SYSTEM_PROMPT` and prompt preambles contained the word `"VALIDATION"` (`CRITICAL VALIDATION CRITERIA`), violating blind evaluation assertions.
   - *Fix*: Build `llm/prompt_firewall.py` to sanitize all outgoing research prompts, stripping holdout metrics, private gate numbers, and forbidden keywords.
5. **Prompt Truncation Inversion**:
   - `render_ideation_prompt()` popped `accepted` items first (`accepted.pop(0)`) rather than dropping lowest-priority rejected ideas, eliminating the newest accepted ideas (`A49`).
   - *Fix*: Budget context by dropping rejected items first, then oldest accepted items, preserving recent accepted strategies (`A49`) and the contract.
6. **G3 Gate Determinism Misclassification**:
   - When an unseeded or random strategy produced an inverted stop/target order, `generate_signal_tape` flagged an error.
   - `gate_determinism` checked `"failed with error" in detail_str.lower()` and marked it as `runtime_error` instead of `lookahead_leak`.
   - *Fix*: Distinguish non-deterministic crashes from standard runtime errors; classify determinism gate failures as `lookahead_leak` while capturing precise root causes.
7. **Prompt vs Config Discrepancy**:
   - Prompts claimed `min_trades >= 150`, while `config.yaml` configured `100` (1h) and `80` (4h).
   - *Fix*: Dynamically inject exact configured thresholds from `config.yaml` into prompt templates.

---

## 4. Planned Modifications & New Modules

```text
quantforge/
├── core/
│   ├── config.py                 [MODIFIED: validate xkiro config, mtf timeframes]
│   ├── event_simulator.py        [MODIFIED: audit & verify cost accounting formulas]
│   ├── splits.py                 [MODIFIED: immutable split hashing & holdout quarantine]
│   └── data_loader.py            [MODIFIED: multi-timeframe validation & dataset hashing]
├── llm/
│   ├── providers/
│   │   ├── __init__.py           [NEW]
│   │   └── xkiro.py              [NEW: Qwen 3.8 Max provider via XKiro API]
│   ├── prompt_firewall.py        [NEW: strict outgoing prompt sanitization]
│   ├── research_director.py      [NEW: autonomous research controller & policy]
│   ├── research_memory.py        [NEW: hypotheses, lineage, failure taxonomy memory]
│   ├── failure_analyzer.py       [NEW: autopsy generation & diagnosis]
│   ├── improve_loop.py           [MODIFIED: multi-turn autopsy-guided improvement]
│   ├── prompts.py                [MODIFIED: multi-role prompts, budget fixer]
│   └── orchestrator.py           [MODIFIED: integrates director, autopsy, repair loop]
├── strategy/
│   ├── dsl/
│   │   ├── __init__.py           [NEW]
│   │   ├── schema.py             [NEW: YAML/JSON Strategy DSL schema]
│   │   ├── compiler.py           [NEW: compiles DSL into executable Strategy class]
│   │   ├── validator.py          [NEW: deterministic DSL validation]
│   │   └── fingerprint.py        [NEW: logical & behavioral fingerprinting]
│   ├── families.py               [NEW: research taxonomy & budget allocation]
│   └── diversity.py              [MODIFIED: behavioral trade correlation check]
├── evolution/
│   ├── __init__.py               [NEW]
│   ├── genome.py                 [NEW: DSL gene representation]
│   ├── mutation.py               [NEW: valid indicator & parameter mutations]
│   ├── crossover.py              [NEW: rule & filter crossover]
│   ├── population.py             [NEW: population pool & Pareto ranking]
│   └── engine.py                 [NEW: tiered evolutionary search engine]
├── portfolio/
│   ├── __init__.py               [NEW]
│   ├── correlation.py            [NEW: return & trade overlap correlation]
│   ├── risk.py                   [NEW: risk contribution & drawdown limits]
│   └── allocator.py              [NEW: risk parity, inverse vol, max Sharpe]
├── terminal/
│   ├── __init__.py               [NEW]
│   └── console.py                [NEW: professional Rich/Textual research console]
├── docs/
│   ├── BASELINE_V1.md            [COMPLETED]
│   ├── UPGRADE_V3_PLAN.md        [CURRENT]
│   ├── COST_MODEL_AUDIT.md       [NEXT]
│   └── ...                       [ALL REQUIRED SPECS]
└── main.py                       [MODIFIED: CLI commands: db, autonomous, research, etc.]
```

---

## 5. Phased Implementation Roadmap (Strict Execution Order)

### Phase 0: Baseline & Foundation Fixes (Immediate)
- Preserve baseline in `docs/BASELINE_V1.md`.
- Detail architecture and dependency map in `docs/UPGRADE_V3_PLAN.md`.
- Fix the 7 known test failures:
  - Atomic run persistence in `storage/mongo.py` & `llm/orchestrator.py`.
  - Rebuildable counters and CLI `db verify` / `db repair-counters`.
  - Eliminate all swallowed exceptions with structured classification.
  - Remove loopback bypass in `dashboard/fastapi_app.py` for server-side auth.
  - Implement `llm/prompt_firewall.py` and sanitize prompts (`validation` substring).
  - Fix prompt truncation budgeting in `llm/prompts.py`.
  - Correct error classification in `validation/gates.py` for determinism.
  - Synchronize prompt trade numbers with `config.yaml`.
- Verify `pytest tests/regression/ tests/canary/ -q` passes 100%.

### Phase 1 & 2: Cost Accounting Audit & Data/Holdout Integrity
- Write `docs/COST_MODEL_AUDIT.md`:
  - Audit `core/event_simulator.py` trade cost formulas (`spread_cost`, `slippage_cost`, `commission`, `swap`, `net_pnl`).
  - Verify mathematical identity: `net_pnl = gross_pnl - spread_cost - slippage_cost - commission - swap`.
  - Add deterministic regression tests preventing double-counting.
- Data hashing & split immutability:
  - Generate dataset checksums for 5m, 15m, 1h, 4h files.
  - Enforce frozen split boundaries; prohibit unauthenticated or repeated holdout access.

### Phase 3: Strategy DSL & Compiler
- Define `strategy/dsl/schema.py` supporting indicators, price action, market structure (BOS/CHOCH/FVG), sessions, and multi-timeframe rules.
- Build `strategy/dsl/compiler.py` transforming DSL into a validated, leak-safe Python Strategy executed in the sandbox.
- Implement exact fingerprinting (`strategy/dsl/fingerprint.py`) for exact, logical, and behavioral duplicate rejection.

### Phase 4: XKiro & Qwen 3.8 Max Integration
- Create `llm/providers/xkiro.py` using `XKIRO_API_KEY`, `XKIRO_BASE_URL`, and `XKIRO_MODEL=qwen-3.8-max`.
- Support JSON Schema structured outputs with exponential backoff and error classification.
- Build specialized prompt roles: Research Director, Hypothesis Generator, Strategy Architect, Failure Analyst, Strategy Improver.

### Phase 5 & 6: Research Director & Autonomous Improvement Loop
- Implement `llm/research_director.py` to steer family allocations and prevent repetitive EMA+RSI searches.
- Implement `llm/research_memory.py` recording hypotheses, lineages, failure causes, and parameter regions.
- Build `StrategyAutopsy` in `llm/failure_analyzer.py` classifying failures (e.g. `TOO_FEW_TRADES`, `COST_SENSITIVE`, `PARAMETER_FRAGILE`).
- Build closed autonomous loop: `Generate -> Sandbox Backtest -> Autopsy -> Qwen Improve -> Retest`.

### Phase 7: Evolutionary Search & Tiered Evaluator
- Implement `evolution/` (genome, mutation, crossover, population, Pareto fitness).
- Fast tiered evaluation: Tier 1 (cheap compile & trade count), Tier 2 (cost stress & parameter stability), Tier 3 (walk-forward & Monte Carlo).

### Phase 8: Strategy Library & Portfolio Engine
- Persistent, searchable Strategy Library with lifecycle state machine:
  `GENERATED -> COMPILED -> SECURITY_PASS -> BACKTESTED -> SCREENED -> ROBUST -> FROZEN -> HOLDOUT_TESTED -> FORWARD_TESTED -> LIVE_READY`.
- Portfolio diversification engine (`portfolio/`): correlation matrices, risk parity allocation, drawdown constraints.

### Phase 9: Terminal UI & Dashboard Upgrades
- High-information-density terminal console in `terminal/console.py` using Rich.
- Dashboard improvements: real-time research event streaming, lineage visualization, candidate explorer, LLM usage breakdowns.

### Phase 10: Full Verification & Reporting
- Comprehensive tests across unit, integration, security, lookahead, cost, and end-to-end dry-run workflows.
- Produce `reports/V1_VS_V3.md` and complete documentation suite in `docs/`.

---

## 6. Migration & Safety Strategy

1. **Zero Greenfield Rewrite**:
   - All existing core files (`core/event_simulator.py`, `core/lookahead_guard.py`, `core/data_loader.py`, `sandbox/runner.py`) remain in place.
   - New capabilities are added as clean, modular extensions.
2. **Backward Compatibility**:
   - Both DSL-based strategies and raw Python sandbox strategies remain fully supported.
   - MongoDB indexes and collection structure are extended without breaking legacy schemas.
3. **No Threshold Gaming**:
   - Validation gates and hurdles (PF ≥ 1.25, Sharpe ≥ 0.80) remain strictly uncompromised. Edge discovery must occur through improved strategy design and cost resilience.
