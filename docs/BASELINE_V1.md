# ELVARIS CAPITAL — QUANT ENGINE BASELINE V1 AUDIT RECORD
**Generated**: 2026-10-06 | **Baseline Commit**: `3d757ff` | **Branch**: `main`
**Status**: PERMANENT HISTORICAL BASELINE — DO NOT OVERWRITE

---

## 1. Executive Summary & Core Metrics

This document preserves the exact empirical benchmark of the QuantForge V1 autonomous strategy discovery engine before the V3 autonomous quantitative research laboratory upgrade. All subsequent performance and efficiency metrics in V3 must be measurable against this baseline.

| Baseline Metric | Recorded Value in V1 | Analysis & Root Cause in V1 |
| :--- | :--- | :--- |
| **Total Evaluated Strategy Runs** | **181** | Runs evaluated across XAUUSD datasets (14:44:12 to 22:53:28 UTC). |
| **Stored Trial Records** | **155** | Discrepancy: 26 runs failed to mirror to `trials` collection. |
| **Autonomous Research Cycles** | **5,523** | Iterative pipeline execution cycles recorded before network pause. |
| **LLM Calls Count** | **206** | Effectively 100% `ideate` calls; 0 calls for structural improvement/autopsy. |
| **LLM Tokens Consumed** | **845,376 tokens** | 559,505 prompt tokens, 285,871 completion tokens; avg latency ~26.8s. |
| **Promoted Strategies (Passed All Gates)** | **0 (0.0%)** | 0 strategies survived all 19 validation gates to reach leaderboard. |
| **Runs Generating ≥1 Trade** | **65 / 181 (35.9%)** | 116 / 181 (64.1%) produced zero trades due to overly restrictive logic. |
| **Gross Price Drift PnL** | **-$55,732.39** | Sum of raw price action movement across all 5,445 backtest trades. |
| **Transaction Friction Drag** | **$408,141.74** | Cumulative drag (spread, volume slippage, commission, swap). |
| **Net PnL Across All Runs** | **-$463,874.13** | Friction drag was 7.3x larger than raw directional losses. |
| **Failing Unit / Regression Tests** | **7 failures** | Known foundational issues (counters, bare exceptions, auth, prompts, determinism). |

---

## 2. Gate Survival Funnel (181 Runs)

The 19-gate validation pipeline eliminated 100% of generated strategies:

```text
Ingested Runs (181) 
     │  100.0%
     ▼
Smoke Compile Pass (174) ──── 7 rejected (3.9%) [SyntaxError / Module load failure]
     │  96.1%
     ▼
Backtest Executed (96) ────── 78 rejected (43.1%) [minimum_sample / zero trades]
     │  53.0%
     ▼
Basic Quality Gate (0) ────── 96 rejected (53.0%) [PF < 1.25 or Sharpe < 0.80]
     │  0.0%
     ▼
Promoted Candidates (0)
```

### Primary Rejection Breakdown:
1. **Minimum Sample (`minimum_sample`)**: **78 runs (43.1%)**
   - Candidates required ≥ 100 trades (1h) or ≥ 80 trades (4h), or ≥ 30 trades/year.
   - LLM ideation produced 4–5 indicator confluences and micro session windows that rarely triggered.
2. **Determinism / State Leaks (`determinism`)**: **68 runs (37.6%)**
   - Unseeded randomness or invalid parameter configurations triggering inverted stop/target errors.
3. **Basic Quality (`basic_quality`)**: **22 runs (12.2%)**
   - Strategies generated trades but failed minimum Profit Factor (1.25) or Sharpe ratio (0.80).
4. **Smoke Run (`smoke_run`)**: **7 runs (3.9%)**
   - Syntax or compilation exceptions inside sandbox.
5. **Signal Validity (`signal_validity`)**: **5 runs (2.8%)**
   - Out-of-bounds bar indices or invalid signal actions.

---

## 3. Near-Miss Profitable Strategies in V1

Despite zero strategies passing all gates, **6 strategies produced net positive dollar profits** under institutional transaction costs, revealing existing market alpha that was rejected due to strict static hurdles:

| Strategy Name | TF | Trades | Win Rate | Net PnL (USD) | Profit Factor | Sharpe | Max DD | Gate Stopped At |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| `LondonNY_LiquiditySweep_Choch` | 1h | 97 | 43.3% | **+$17,200.00** | **1.30** | 0.68 | 7.2% | `minimum_sample` (needed 100 trades) |
| `GoldBbWidthMomentumBreakout` | 4h | 133 | 36.1% | **+$6,954.44** | **1.09** | 0.26 | 10.0% | `basic_quality` (PF < 1.25, Sharpe < 0.8) |
| `GoldAdxKeltnerRocContinuation` | 4h | 101 | 42.6% | **+$3,609.09** | **1.06** | 0.18 | 10.6% | `basic_quality` (PF < 1.25, Sharpe < 0.8) |
| `GoldZScoreSessionMomentum` | 4h | 171 | 40.4% | **+$2,460.00** | **1.03** | 0.12 | 14.8% | `basic_quality` (PF < 1.25, Sharpe < 0.8) |
| `GoldBollingerAdxTrendPullback` | 4h | 6 | 50.0% | **+$1,630.00** | **1.55** | 0.26 | 2.4% | `minimum_sample` (6 trades vs 80 min) |
| `GoldSessionRangeAtrBreakout` | 4h | 1 | 100.0% | **+$1,560.00** | **999.00** | 0.52 | 0.0% | `minimum_sample` (1 trade vs 80 min) |

---

## 4. LLM Loop Pathology in V1

1. **Monolithic Ideate-Only Cycle**:
   - Out of 206 LLM calls, **206 were `ideate`**.
   - Zero `improve`, `autopsy`, `rethink`, or `diagnose` calls were executed.
   - When a promising idea failed a gate by a small margin (e.g. 97 trades vs 100 required, or PF=1.09 vs 1.25), the engine abandoned the lineage entirely and generated a completely unrelated strategy.
2. **Context Budgeting / Truncation Flaw**:
   - Ideation prompt budgeting truncated newest accepted ideas (`A49`) instead of oldest rejected ideas.
3. **Information Leakage in System Prompt**:
   - The literal token `"validation"` was embedded in the system prompt preamble (`"CRITICAL VALIDATION CRITERIA"`), violating blind evaluation hygiene.
4. **Prompt vs Config Discrepancy**:
   - Prompts claimed `min_trades >= 150`, while `config.yaml` configured `100` (1h) and `80` (4h).

---

## 5. Transaction Cost and Microstructure Drag in V1

- Total friction paid across 5,445 backtest trades: **$408,141.74**.
- High-frequency 1h strategies churned trades with small profit margins (3–8 USD points) that were overwhelmed by:
  - Spread ($0.25–$0.40/oz)
  - Execution slippage ($0.40–$0.80/oz on volatility breakouts)
  - Overnight financing swap (Wednesday triple swap)
- In multiple strategies, friction absorbed 80–90% of gross trading gains.

---

## 6. The 7 Baseline Unit/Regression Test Failures

The V1 test suite exhibited 7 failing tests / structural defects:
1. `TestDB1RunsArePersisted::test_1000_runs_are_all_stored` — Counter and document persistence mismatch (`_stored_run_count` found 2000 vs 1000 due to dual-collection insertion).
2. `TestDB1RunsArePersisted::test_trial_counter_counts_every_attempted_idea` — Global counter drift (6 vs 3).
3. `test_no_silently_swallowed_exceptions` — 12 bare/silent `except: pass` exception blocks in `core/indicators.py`, `llm/orchestrator.py`, `scripts/export_analysis.py`, and `sandbox/child.py`.
4. `TestDashboardSecurityAndAuth::test_unauthenticated_request_rejected` — Loopback IP bypass in `dashboard/fastapi_app.py` allowed unauthenticated access to `/api/engine/state`.
5. `TestF7ImproveLoopMustPass::test_validation_and_robustness_numbers_never_reach_any_prompt` — Substring `"validation"` present in prompt preamble.
6. `TestF7PromptsMustPass::test_ideation_prompt_respects_budget_truncating_low_priority_first` — Ideation prompt dropped newest ideas instead of oldest.
7. `Test19GatesUnit::test_g3_determinism` — Determinism test classified invalid signal inversion as `runtime_error` rather than `lookahead_leak`.

---

## 7. V1 Hardware & Execution Benchmarks

- **Processor**: 12 logical CPU cores (AMD/Intel)
- **Host OS**: Windows 10 (Build 26200)
- **RAM**: 13.86 GB total
- **Python**: 3.11.9
- **Sandbox Bar Processing Throughput**: **1,135.4 bars/second** (synthetic micro-benchmark on 2,000 bars)
- **Total Test Execution Time**: Fast regression suite completed in **89.6 seconds**.
