# ELVARIS CAPITAL — AUTONOMOUS QUANT RESEARCH ENGINE
# MASTER ARCHITECTURE SPECIFICATION (VERSION 3.0)

---

## 1. Executive Overview

The Elvaris Capital Autonomous XAUUSD Quant Research Engine (V3) is a fully autonomous systematic trading laboratory. Rather than functioning as a simplistic single-prompt strategy generator, the engine operates as an autonomous research institution governed by quantitative rigor, deterministic execution safety, strict lookahead prevention, and evolutionary hypothesis optimization.

The system is engineered specifically for gold spot trading (`XAUUSD`) across four coordinated timeframes:
- **5-Minute (`5m`)**: Execution entry and microstructure liquidity analysis.
- **15-Minute (`15m`)**: Primary setup and swing structure identification.
- **1-Hour (`1h`)**: Intermediate market structure, trend filter, and session context.
- **4-Hour (`4h`)**: Macro volatility regime and structural order flow.

Intelligence is powered by **Qwen 3.8 Max** via the **XKiro API**, coordinated through specialized cognitive roles under deterministic oversight.

---

## 2. Fundamental Architectural Principle: Deterministic Primacy

```
┌─────────────────────────────────────────────────────────────┐
│                 DETERMINISTIC PRIMACY                       │
├──────────────────────────────┬──────────────────────────────┤
│  LLM (Qwen 3.8 Max) Role     │  Deterministic Engine Role   │
├──────────────────────────────┼──────────────────────────────┤
│ • Propose hypotheses         │ • AST security sandbox       │
│ • Suggest strategy DSL logic │ • Chronological enforcement  │
│ • Propose parameter ranges   │ • Non-leakage verification   │
│ • Diagnose failure autopsies │ • Transaction cost modeling  │
│ • Suggest structural fixes   │ • 19-gate statistical sieve  │
│ • Explore research lineages  │ • Holdout quarantine vault   │
│                              │ • Capital allocation limits  │
└──────────────────────────────┴──────────────────────────────┘
```

**Non-Negotiable Axiom**: Under no circumstances can the LLM override, bypass, or weaken deterministic safety gates, holdout isolation locks, or statistical validation tests.

---

## 3. High-Level System Architecture

```
                    ┌─────────────────────────┐
                    │    RESEARCH DIRECTOR    │
                    │ (Budget & Taxonomy Mgr) │
                    └───────────┬─────────────┘
                                │ Selects next hypothesis
                                ▼
                    ┌─────────────────────────┐
                    │      QWEN 3.8 MAX       │
                    │   (via XKiro Provider)  │
                    └───────────┬─────────────┘
                                │ Structured Output
                                ▼
                    ┌─────────────────────────┐
                    │      STRATEGY DSL       │
                    │  (Schema & AST Guard)   │
                    └───────────┬─────────────┘
                                │ Compiles & Verifies
                                ▼
                    ┌─────────────────────────┐
                    │   DETERMINISTIC GATES   │
                    │ (Security/Leakage Scan) │
                    └───────────┬─────────────┘
                                │ Pass
                                ▼
                    ┌─────────────────────────┐
                    │    EVENT SIMULATOR      │
                    │ (Full Friction Costs)   │
                    └───────────┬─────────────┘
                                │ Execution Telemetry
                                ▼
                    ┌─────────────────────────┐
                    │    STRATEGY AUTOPSY     │
                    │   (18 Failure Classes)  │
                    └───────┬───────────┬─────┘
               Repairable   │           │ Initial Pass
                    ┌───────┘           └───────┐
                    ▼                           ▼
        ┌───────────────────────┐   ┌───────────────────────┐
        │     QWEN IMPROVER     │   │ ADVERSARIAL CHALLENGE │
        │ (Sanitized Guidance)  │   │   (Robustness Suite)  │
        └───────────┬───────────┘   └───────────┬───────────┘
                    │                           │
                    └─────────► RETEST ◄────────┘
                                    │ Robust Pass
                                    ▼
                        ┌───────────────────────┐
                        │    STRATEGY FROZEN    │
                        │ (Cryptographic Hash)  │
                        └───────────┬───────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │     HOLDOUT VAULT     │
                        │ (Sealed Out-of-Sample)│
                        └───────────┬───────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   PORTFOLIO ENGINE    │
                        │(Parity & Corr Matrix) │
                        └───────────────────────┘
```

---

## 4. Core System Modules

### 4.1 Data Architecture (`core/data_loader.py`, `core/splits.py`)
- Independent verification and SHA-256 fingerprinting of 5m, 15m, 1h, and 4h bar datasets.
- 5-year historical horizon partitioned into immutable chronological splits:
  - **Train Split**: Continuous model hypothesis fitting and initial screening.
  - **Embargo Buffer**: 7-day neutral buffer preventing straddle contamination.
  - **Validation Split**: Out-of-sample screening and parameter neighborhood sensitivity.
  - **Holdout Vault**: Cryptographically sealed test set, accessible only once per frozen candidate.

### 4.2 Strategy DSL (`strategy/dsl/`)
- Declarative, strictly typed Pydantic schema eliminating natural language ambiguity.
- Exact quantitative formulation of ICT/SMC concepts:
  - *Displacement*: $Body / Range > 0.65 \land Range > 1.8 \times ATR_{14} \land Close > SwingHigh$.
  - *Liquidity Sweep*: $High > SwingHigh \land Close < SwingHigh$.
  - *Fair Value Gap (FVG)*: $Low[t] > High[t-2]$ (Bullish) or $High[t] < Low[t-2]$ (Bearish).
- AST compiler producing sandboxed Strategy classes that execute within the hardened sub-process isolation jail.

### 4.3 Three-Tier Fingerprinting (`strategy/dsl/fingerprint.py`)
- **Level 1 (Exact)**: Normalized whitespace, comment-stripped AST hash.
- **Level 2 (Logical)**: Canonical indicator and entry condition sequence hash.
- **Level 3 (Behavioral)**: Position vector correlation and simultaneous trade overlap ($>90\%$ threshold triggers duplicate rejection).

### 4.4 Cognitive LLM Layer (`llm/`)
- **`XKiroProvider`**: High-performance HTTP client for Qwen 3.8 Max with token budgeting, `<think>` reasoning block removal, and exponential backoff retry.
- **`PromptFirewall`**: Outgoing context sanitizer blocking holdout performance, hidden thresholds, and API secrets.
- **`FailureAnalyzer`**: 18-class failure classifier producing `StrategyAutopsy` diagnostic records.
- **`ResearchDirector`**: Top-level autonomous decision-maker steering research families and allocations.
- **`ResearchMemory`**: Persistent knowledge graph of past hypotheses, failure causes, and lineage trees.

### 4.5 Execution & Cost Accounting (`core/backtester.py`, `core/event_simulator.py`)
- Strict tick-accurate simulation with spread, slippage, commission ($7.00/lot round-trip), and triple-swap Wednesday accounting.
- Mathematical identity enforced on every trade:
  $$NetPnL = GrossPnL - SpreadCost - SlippageCost - Commission - SwapCost$$
- Gate 12 Cost Resilience audit ensuring strategy survive at least $1.5\times$ spread expansion.

### 4.6 Evolutionary Optimization (`evolution/`)
- Genetic search engine operating over StrategyDSL genotypes.
- Mutation operators: Parameter perturbation, indicator replacement, threshold nudging.
- Crossover operators: Safe combination of uncorrelated entry setups and exit rules.
- Multi-objective Pareto selection optimizing Sharpe, Profit Factor, Cost Resilience, and Drawdown protection.

### 4.7 Portfolio Engine (`portfolio/`)
- Multi-strategy return correlation analysis.
- Capital allocation models: Equal Weight, Inverse Volatility, Risk Parity, and Maximum Sharpe.
- Institutional risk budgets enforcing max $35\%$ weight per strategy and capping correlated exposures.

---

## 5. Strategy Lifecycle State Machine

A strategy advances through a deterministic 12-stage lifecycle:

```
GENERATED ➔ COMPILED ➔ SECURITY_PASS ➔ BACKTESTED ➔ SCREENED ➔ ROBUST
    ➔ FROZEN ➔ HOLDOUT_TESTED ➔ FORWARD_TESTED ➔ PORTFOLIO_ELIGIBLE ➔ LIVE_READY ➔ DEPLOYED
```

Once a strategy enters `FROZEN`, its code hash, parameter specification, and dataset reference become immutable. Any subsequent alteration creates a new descendant strategy with a distinct lineage identifier.
