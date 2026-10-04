# QuantForge Audit Report

_Delivered 2026-10-04. Every item tagged VERIFIED was reproduced with running code. Items tagged READ were found by reading source._

_See `tests/repro/repro_exploits.py` for executable reproductions._

---

## Severity Key

- **S1** = results cannot be trusted or security is broken
- **S2** = wrong or misleading output
- **S3** = missing feature from the spec
- **S4** = quality or polish

---

## A1. Sandbox (`core/sandbox.py`)

| ID | Sev | Status | Finding |
|---|---|---|---|
| SBX-1 | S1 | VERIFIED | **Full sandbox escape.** `type` and `object` in allowed builtins + no dunder ban → `().__class__.__base__.__subclasses__()` → real `os` module. |
| SBX-2 | S1 | VERIFIED | **Arbitrary file read.** `pd.read_csv` works in sandbox. Strategy reads price file, gets Sharpe 44.9. |
| SBX-3 | S1 | VERIFIED | **Secrets reachable.** `pd.io.common.os.environ['MONGO_URL']` returns the Mongo URL. |
| SBX-4 | S1 | VERIFIED | **Timeout cannot interrupt.** `while True: pass` hangs the engine. No memory/CPU limit. |
| SBX-5 | S2 | READ | Strategies cannot receive parameters (instantiated with no args). |
| SBX-6 | S2 | READ | Static scan doesn't block `eval`, `exec`, `open` etc. as source patterns. |

## A2. Lookahead Defence (`core/lookahead_guard.py`)

| ID | Sev | Status | Finding |
|---|---|---|---|
| LH-1 | S1 | VERIFIED | **Future data via `.base`.** `bars['close'].to_numpy().base` returns parent buffer with all future OHLCV. Sharpe 44.88. |
| LH-2 | S1 | VERIFIED | **Truncation test is wrong.** Uses Sharpe CV, not signal identity. Cheater passes with CV 0.012. |
| LH-3 | S1 | VERIFIED | **Static scan trivially evadable.** Misses `shift(periods=-1)`, `rolling(center=True)`, `.base`, dunder attrs. |
| LH-4 | S1 | READ | Suspicion flags are informational only, not hard rejections. |
| LH-5 | S2 | READ | Delay test returns ambiguous pass on non-profitable strategies. |
| LH-6 | S1 | VERIFIED | **Phase 3 tests are vacuous.** No canary strategies. |

## A3. Backtester (`core/backtester.py`)

| ID | Sev | Status | Finding |
|---|---|---|---|
| BT-1 | S1 | VERIFIED | **Inverted stop = free money.** LONG with SL above price → 100% win rate on random walk. |
| BT-2 | S1 | VERIFIED | **Entry bar stops skipped.** SL at 1990, entry bar low 1900, stop not triggered. |
| BT-3 | S2 | VERIFIED | **Costs double-counted.** Spread/slippage in fill price AND subtracted again in net PnL. |
| BT-4 | S1 | VERIFIED | **Spread 100x too small.** `pip_value` multiplication makes $0.25 spread → $0.0025. |
| BT-5 | S2 | READ | Equity realized-only; open-trade drawdown invisible. |
| BT-6 | S2 | READ | Sharpe from per-trade returns, not time-based. |
| BT-7 | S2 | READ | Gap fills at stop price, not at gap open. |
| BT-8 | S2 | READ | No bid/ask model; spread charged twice per round trip. |
| BT-9 | S2 | READ | Swap counts midnight UTC, not 17:00 NY rollover. |
| BT-10 | S2 | READ | No margin/leverage check. |
| BT-11 | S3 | READ | No trailing stop, time stop, breakeven, or CLOSE action. |
| BT-12 | S1 | READ | Strategy exceptions swallowed; crashes reported as "no trades". |
| BT-13 | S3 | READ | Single engine; no cross-check. |
| BT-14 | S2 | READ | No minimum stop/TP distance. |

## A4. Gates (`core/gates.py`)

| ID | Sev | Status | Finding |
|---|---|---|---|
| GATE-1 | S1 | VERIFIED | **DSR uses annualised Sharpe in per-period formula.** Collapses to 0 at 20 trials. |
| GATE-2 | S1 | VERIFIED | **Monte Carlo broken.** 5th percentile = actual total PnL. Gate adds nothing. |
| GATE-3 | S2 | READ | "PBO" is not real PBO (splits one strategy's trades). |
| GATE-4 | S1 | READ | Parameter sensitivity and walk-forward never executed. |
| GATE-5 | S3 | READ | Missing gates: random-entry baseline, buy-and-hold, long/short split, etc. |
| GATE-6 | S2 | READ | Gate order: lookahead checks run after validation data is touched. |
| GATE-7 | S4 | READ | Robustness score 0-1 (spec: 0-100), inputs don't exist. |

## A5. LLM Layer

| ID | Sev | Status | Finding |
|---|---|---|---|
| LLM-1 | S1 | READ | No key failover; 429 raises with no retry on other key. |
| LLM-2 | S1 | READ | Token tracking in memory only. |
| LLM-3 | S1 | READ | No improve loop. `code_fix_prompt` imported but never used. |
| LLM-4 | S2 | READ | Prompts show only past idea names, no logic summaries. |
| LLM-5 | S2 | READ | No `<think>` stripping, no truncation handling. |
| LLM-6 | S2 | READ | No `requirements.txt`. |

## A6. Orchestrator and Persistence

| ID | Sev | Status | Finding |
|---|---|---|---|
| ORCH-1 | S1 | READ | Loop is finite and stops on first survivor. |
| ORCH-2 | S2 | READ | Uses `print()`, not `QuantForgeLogger`. |
| ORCH-3 | S1 | READ | DB errors swallowed with `except: pass`. |
| DB-1 | S1 | VERIFIED | **Only first trial saved.** Unique index on `counter_id` blocks subsequent inserts. |
| ORCH-4 | S2 | READ | No resume; fixed 2s sleep after errors. |
| DB-2 | S2 | VERIFIED | `main.py leaderboard` passes dict to `get_db()`, raises TypeError. |
| DB-3 | S4 | READ | Index/field mismatches. |

## A7. Data Layer

| ID | Sev | Status | Finding |
|---|---|---|---|
| DATA-1 | S2 | READ | `patch_hole.py` aligns to midnight, not NY-close grid. |
| DATA-2 | S2 | READ | No data lock (file hash not stored at freeze). |
| DATA-3 | S4 | READ | CSV re-read on every trial. |
| DATA-4 | S3 | READ | No indicator library. |
| DATA-5 | S4 | READ | Good: data validation, DST, session labels, frozen splits, holdout lock. |

## A8. Dashboard

| ID | Sev | Status | Finding |
|---|---|---|---|
| UI-1 | S3 | READ | No start/stop/pause control. |
| UI-2 | S3 | READ | No LLM panel, charts, filtering. |
| UI-3 | S2 | READ | Leaderboard missing most stats. |
| UI-4 | S1 | READ | CORS wildcard. |
| UI-5 | S4 | READ | Polling only, no SSE. |

## A9. Tests and Hygiene

| ID | Sev | Status | Finding |
|---|---|---|---|
| TEST-1 | S2 | VERIFIED | 25 tests need live MongoDB. |
| TEST-2 | S2 | READ | No noise/edge/canary/sandbox-abuse tests. |
| OPS-1 | S3 | READ | No README, requirements.txt, docs. |
| OPS-2 | S3 | READ | No holdout run, forward trader, decay monitor, diversity. |

## A10. What is Good

- Data validation, DST/session logic
- Frozen split boundaries with embargo and atomic holdout lock
- Closed-bar signal / next-open fill model (idea correct; implementation needs hardening)
- Config-driven thresholds
- 136 passing tests (foundation to build on)
