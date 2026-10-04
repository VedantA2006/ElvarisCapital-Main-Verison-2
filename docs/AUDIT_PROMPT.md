<USER_REQUEST>
# QuantForge: Deep Audit and Antigravity Fix Prompt

Repo audited: `ElvarisCapital-Main-Verison-2` (uploaded zip, every Python file read, tests run, exploits executed).

**How to use this file**
1. Read PART A (the defect register) yourself. Every item is tagged `VERIFIED` (I ran code and reproduced it) or `READ` (found by reading the code).
2. Paste PART B (from the line `=== BEGIN ANTIGRAVITY PROMPT ===` to the end) into Antigravity. Paste the whole of PART B, because it refers to the defect IDs from PART A. Put PART A in the repo as `docs/AUDIT.md` so Antigravity can read it too.
3. Keep `repro_exploits.py` (delivered next to this file) in `tests/repro/`. It reproduces the worst defects, so the fixes can be proven.

**Honest summary:** the data layer is good. Everything that protects you from fake results (sandbox, lookahead checks, backtester, gates) has real holes, and several of them are exploitable by accident by an LLM. As it stands the engine can report a strategy with a 100% win rate on random data, or a Sharpe of 45, as a valid candidate. The fixes below are ordered so that the foundation is made trustworthy before any new feature is added.

---

# PART A: DEFECT REGISTER

Severity: **S1** = results cannot be trusted or security is broken. **S2** = wrong or misleading output. **S3** = missing feature from the spec. **S4** = quality or polish.

## A1. Sandbox (`core/sandbox.py`)

| ID | Sev | Status | Finding |
|---|---|---|---|
| SBX-1 | S1 | VERIFIED | **Full sandbox escape.** `type` and `object` are in the allowed builtins (lines ~69, 77) and the static scan has no rule for dunder attributes. `().__class__.__base__.__subclasses__()` leads to `warnings.catch_warnings()._module.__builtins__['__import__']('os')`. I obtained the real `os` module, the pid and the working directory from inside a "sandboxed" strategy. |
| SBX-2 | S1 | VERIFIED | **Arbitrary file read.** Raw `np` and `pd` are injected into the strategy namespace (lines ~118-120). `pd.read_csv('<any path>')` works, so a strategy can read the full price file, the validation split and the holdout. The holdout lock in `splits.py` only protects the `DataStore.get_data()` path. A strategy that read a CSV got Sharpe 44.9 and profit factor 55.8 and passed the static scan. |
| SBX-3 | S1 | VERIFIED | **Secrets reachable.** The strategy runs in the same process as the engine. `pd.io.common.os.environ['MONGO_URL']` returned the Mongo URL. API keys are in the same environment. |
| SBX-4 | S1 | VERIFIED | **Timeout cannot interrupt.** `_TimeoutWrapper` (lines ~212-229) checks elapsed time only after `on_bar` returns. `while True: pass` hangs the whole engine (I had to kill it from outside). There is no memory limit, no CPU limit and no process isolation. This contradicts Phase 4 of the spec ("timeout kill, memory limit kill, no network, no file access"). |
| SBX-5 | S2 | READ | `Sandbox.load_strategy` instantiates `klass()` with no arguments (line ~205). Strategies cannot receive parameters, so parameter sensitivity (gate 4) is impossible for LLM-generated code. |
| SBX-6 | S2 | READ | Static scan does not block `eval`, `exec`, `open`, `__import__`, `compile`, `globals`, `getattr` as source patterns. It relies only on the runtime builtins whitelist, which SBX-1 bypasses. |

## A2. Lookahead defence (`core/lookahead_guard.py`)

| ID | Sev | Status | Finding |
|---|---|---|---|
| LH-1 | S1 | VERIFIED | **The "strategy only sees the past" guarantee is false.** `backtester.py:328` passes `bars.iloc[:i+1]`, which is a view. `bars['close'].to_numpy().base` returns the parent buffer: for a 100-bar slice the base had shape (5, 1500), meaning all future OHLCV. A strategy using only pandas and numpy (no file reads) got Sharpe 44.88, PF 55.8, win rate 85%, and passed the static scan. |
| LH-2 | S1 | VERIFIED | **The truncation test does not test truncation.** `run_truncation_test` (lines ~358-408) computes the coefficient of variation of Sharpe across cuts and passes if CV < 2.0. The spec requires that the signals for all bars ≤ T are identical between a full run and a run cut at T. The cheater above passed with Sharpe mean 45.9 and CV 0.012. |
| LH-3 | S1 | VERIFIED | **Static scan is easy to evade.** It misses `shift(periods=-1)` (keyword form, since only `node.args` is inspected, line ~137), `rolling(center=True)`, `shift(n)` with a variable, `j = i + 1; iloc[j]`, `.base`, and every dunder attribute. The extra regex on words like "future" or "cheat" is defeated by renaming a variable. |
| LH-4 | S1 | READ | **"Suspicious" results do not reject.** Sharpe > 3, PF > 3, win rate > 85% and top-5% concentration are only appended to a text string (`gates.py` ~116-132). The spec says they trigger a deep re-audit and then rejection. |
| LH-5 | S2 | READ | Delay test (`~315`): when the original Sharpe is ≤ 0 it takes an early-return path. This must fail closed (a non-profitable strategy is simply rejected earlier), not return an ambiguous pass. It also re-runs the whole strategy for each delay, which is slow. |
| LH-6 | S1 | VERIFIED | **Phase 3 acceptance tests are vacuous.** `test_honest_strategy_survives_delay` asserts `original_trades >= 0`. `test_cheating_strategy_fails_delay` asserts `orig_sharpe != delayed_sharpe`. Truncation tests assert type and length only. None of the 5 canary strategies required by the protocol exist, and there is no canary table. |

## A3. Backtester (`core/backtester.py`)

| ID | Sev | Status | Finding |
|---|---|---|---|
| BT-1 | S1 | VERIFIED | **Free-money exploit: inverted stop.** `Signal` is never validated. A LONG with `stop_loss` *above* the entry price is "stopped out" on the next bar at that higher price. On a pure random walk this gave 916 trades, 100% win rate, Sharpe 7.35, +15,698% return. An LLM can produce this by accident, and an improve loop will then optimise toward it. |
| BT-2 | S1 | VERIFIED | **Stops ignored on the entry bar.** Exit checks are guarded by `i > pos_entry_idx` (line ~295), and the entry fills at bar `i+1`, so the whole entry bar is skipped. A stop at 1990 survived an entry bar that traded down to 1900. This is systematically optimistic. |
| BT-3 | S2 | VERIFIED | **Costs double-counted.** The entry fill price already includes spread and slippage (lines ~349-351), so `gross_pnl` (line ~247) already contains them, then `net_pnl` subtracts the entry spread and slippage again (lines ~251-265). Stop exits also include slippage in the fill and subtract it again. "Gross" is therefore not gross and `cost_gross_ratio` is wrong. |
| BT-4 | S1 | VERIFIED | **Spread is 100x too small.** `spread_points` multiplies the config value by `pip_value` (0.01) (lines ~120-130). With default 0.25 it returns $0.0025, while the config comment says $0.25 per oz. Real gold spreads are roughly $0.20 to $0.50. Costs are therefore almost absent, which inflates every result (especially 1h). |
| BT-5 | S2 | READ | **Equity is realized-only.** The equity curve is appended only when a trade closes (line ~372). Open-trade floating losses never enter drawdown, so max drawdown is understated. |
| BT-6 | S2 | READ | Sharpe/Sortino/Calmar are computed from per-trade returns divided by *initial* equity (line ~455) and annualised by trades per year, not from time-based returns. They are not comparable with buy-and-hold or with the DSR formula. |
| BT-7 | S2 | READ | Stops that gap through are filled at the stop price (lines ~317-319), not at the gap open (spec section 5.4). |
| BT-8 | S2 | READ | No bid/ask model. Spread is charged as a full amount on entry *and* again on exit; exits at TP/SL are not adjusted for bid/ask. |
| BT-9 | S2 | READ | Swap counts midnight UTC crossings (line ~155), including Saturday and Sunday nights, instead of the 17:00 New York rollover on trading days. |
| BT-10 | S2 | READ | No margin or leverage check; lots are rounded to *nearest* step (line ~178), which can exceed the risk budget; fallback risk of 1% of price when the stop is missing. |
| BT-11 | S3 | READ | Exits are limited to fixed SL/TP and reversal. No CLOSE action, trailing stop, time stop or breakeven. `OrderType` (line ~50) is defined and never used. The spec says exits matter more than entries. |
| BT-12 | S1 | READ | **Strategy exceptions are swallowed** (`except Exception: pass`, lines ~329-330). A strategy that crashes on every bar is reported as "no trades", so the LLM never gets the traceback to fix. A test (`test_strategy_exception_does_not_crash_backtest`) enshrines this behaviour. |
| BT-13 | S3 | READ | Single engine only. No numba vectorized core and no independent event-based engine, so the Phase 2 requirement (two engines must agree on 20+ strategies) is not met. Signal generation and execution are fused in one loop, so every robustness test must re-run the strategy. |
| BT-14 | S2 | READ | No minimum stop or TP distance. A stop of $0.01 combined with risk-based sizing yields enormous lot sizes, and a TP smaller than the spread is a free scalp that only exists because costs are wrong. |

## A4. Gates (`core/gates.py`)

| ID | Sev | Status | Finding |
|---|---|---|---|
| GATE-1 | S1 | VERIFIED | **DSR uses mixed units** (lines ~420-444). An *annualised* Sharpe is plugged into a formula that needs a per-period Sharpe, and the expected-maximum term lacks the cross-trial variance factor. Result: with Sharpe 1.5 and 400 trades the gate passes up to about 5 trials and the probability is 0.0 by 20 trials. After a dozen trials it rejects every honest strategy, while a Sharpe-45 cheater still passes. |
| GATE-2 | S1 | VERIFIED | **Monte Carlo is broken.** (a) "Drop" removes the *worst* trades (`argsort` then `idx[n_drop:]`, lines ~316-317), which flatters results. (b) Shuffling never changes final equity, so the 5th-percentile final equity equals the real total PnL exactly (confirmed numerically). The gate adds nothing beyond "total PnL > 0". (c) The kept trades come out sorted by value, distorting drawdown. (d) No price noise, extra slippage or bootstrap, as the spec requires. |
| GATE-3 | S2 | READ | **"PBO" is not PBO** (lines ~469-487). It splits one strategy's trade list. Real CSCV needs a matrix of many strategies over the same time periods. It also returns PASS when there are too few trades (fail-open). |
| GATE-4 | S1 | READ | **Parameter sensitivity and walk-forward are never executed.** `orchestrator.py` (~152-200) runs train, MC, regime, DSR, PBO, validation, delay and truncation. The robustness score reserves weight for walk-forward and sensitivity that never exist, so those weights are always zero. |
| GATE-5 | S3 | READ | Missing gates from the spec: random-entry baseline, buy-and-hold comparison, long/short split, cost resilience, delay-2, cross-asset check, regime labels beyond year concentration, holdout flow, effective-trial accounting. |
| GATE-6 | S2 | READ | **Gate order is wasteful and unsafe.** Delay and truncation run *after* the validation split (`orchestrator.py` ~188-198). Lookahead checks must be the cheapest and first gates, before any validation data is touched. |
| GATE-7 | S4 | READ | Robustness score is 0 to 1 (spec: 0 to 100) and built from inputs that do not exist yet. |

## A5. LLM layer (`llm/client.py`, `llm/prompts.py`, `llm/orchestrator.py`)

| ID | Sev | Status | Finding |
|---|---|---|---|
| LLM-1 | S1 | READ | **No key failover.** On HTTP 429 the client raises `LLMRateLimitError` (line ~117) with no cooldown tracking and no retry on the other key. Rotation happens only if `alternate_keys` is true (default false). Your two-key requirement is not implemented. |
| LLM-2 | S1 | READ | **No token tracking.** Counters live in memory only. Nothing is written to `llm_calls` or `key_status`, so the dashboard cannot show tokens, model or key state. |
| LLM-3 | S1 | READ | **No improve loop.** The core idea ("backtest, send the result to the LLM, fix or improve, repeat") is not implemented. `code_fix_prompt` is imported (`orchestrator.py:45`) and never used. A sandbox error rejects the idea immediately with no retry. Invalid JSON gets no repair attempt. |
| LLM-4 | S2 | READ | Prompts: only past idea *names* are shown (no logic summaries); there is no structured spec, no hypothesis ("who is on the other side"), no parameter contract, no coverage-map target, no helper library, no concept-family steering. |
| LLM-5 | S2 | READ | Qwen-class models often emit `<think>...</think>` blocks or a separate `reasoning_content`, may not support JSON mode, and `max_tokens=4096` can truncate code (`finish_reason == "length"`). None of this is handled. |
| LLM-6 | S2 | READ | `requests` is imported but there is no `requirements.txt`. A fresh install fails. |

## A6. Orchestrator and persistence

| ID | Sev | Status | Finding |
|---|---|---|---|
| ORCH-1 | S1 | READ | **The loop is not autonomous.** `run_loop(max_trials=50)` is finite, and it prints "Survivor found! Stopping loop." and exits on the first survivor (lines ~227-260). You asked for a loop that never stops until you press stop. |
| ORCH-2 | S2 | READ | The orchestrator uses `print()` and never uses `QuantForgeLogger`, so there are no engine logs in Mongo, which you explicitly asked for. |
| ORCH-3 | S1 | READ | Database errors are swallowed (`except Exception: pass`, lines ~321-322 and ~331). Trials silently fail to save and the trial count (used by DSR) goes wrong. |
| DB-1 | S1 | VERIFIED | **Only the first trial is ever saved.** `ensure_indexes()` puts a unique index on `trials.counter_id`. The orchestrator inserts its trial documents into the *same* `trials` collection with no `counter_id`. A unique non-sparse index treats missing as null, so the second insert fails with `DuplicateKeyError`, which ORCH-3 hides. Confirmed with mongomock: 4 inserts, 1 saved, `count_documents` = 2. The DSR trial count therefore stays at about 2 forever, and the dashboard shows one trial. |
| ORCH-4 | S2 | READ | No resume. `_past_ideas` lives in memory, so a restart forgets everything. A fixed 2-second sleep is used after any outcome, so a persistent error (e.g. 429) spins quickly and records junk trials. |
| DB-2 | S2 | VERIFIED | `main.py:138` calls `get_db(cfg)` with a dict, but `get_db` expects a database name. `python main.py leaderboard` raises `TypeError`. |
| DB-3 | S4 | READ | `data_reports` is indexed on `timestamp` but documents use `created_at`. `get_logger(**kwargs)` ignores kwargs after the first call. Logger Mongo failures are silent (no counter of dropped logs). |

## A7. Data layer (`core/data_loader.py`, `core/splits.py`, `patch_hole.py`)

| ID | Sev | Status | Finding |
|---|---|---|---|
| DATA-1 | S2 | READ | **`patch_hole.py` risks.** It fills a September 2025 hole in the 1h/4h files by resampling 5m bars with `resample("240min")`, which aligns bins to midnight, while the native 4h bars may follow the NY-close grid (the config describes a 17:00 NY daily break). Misaligned patch bars will not match their neighbours at the hole edges. It overwrites the source CSVs (a .bak is kept), sits in the repo root, and the patched bars come from a different source than the rest of the file. If September 2025 falls in the holdout split, the holdout is contaminated by synthetic bars. |
| DATA-2 | S2 | READ | **No data lock.** The split freeze stores boundaries but not file hashes. Editing a CSV after the freeze (like `patch_hole.py` does) changes the data silently while old backtests keep the old hash. |
| DATA-3 | S4 | READ | `DataStore` re-reads and re-validates the CSV on every trial (the orchestrator creates one per trial). Cache by file hash. |
| DATA-4 | S3 | READ | `core/indicators.py` (spec section 4) does not exist: no tested indicator library, no `htf()` for closed higher-timeframe bars, no confirmed-pivot or market-structure helpers. The LLM re-implements them each time, which is both slow and a lookahead risk. |
| DATA-5 | S4 | READ | Good: data validation, DST handling, session labels, frozen split boundaries, the atomic holdout lock. Keep these. |

## A8. Dashboard (`dashboard/app.py`)

| ID | Sev | Status | Finding |
|---|---|---|---|
| UI-1 | S3 | READ | No start/stop/pause control, no engine state, no heartbeat. The engine and the dashboard are not separate processes. |
| UI-2 | S3 | READ | No LLM panel (model, active key, key state, tokens used per key/model/day, calls, failures, key switches), no live log tail, no funnel, no coverage map, no charts, no strategy detail page, no sorting or filtering. |
| UI-3 | S2 | READ | Leaderboard shows only a handful of fields. The spec's section 14 stats (monthly averages, monthly drawdown, yearly table, holdout and forward columns) are absent. |
| UI-4 | S1 | READ | Security: `Access-Control-Allow-Origin: *` (line ~75) lets any web page read your data from the browser. Once start/stop buttons exist this becomes a remote-control risk. Errors return `str(e)` to the client. `HTTPServer` is single-threaded, so one slow request blocks the rest. |
| UI-5 | S4 | READ | Polling every 5 s with no push updates; the UI is a small static page. |

## A9. Tests and project hygiene

| ID | Sev | Status | Finding |
|---|---|---|---|
| TEST-1 | S2 | VERIFIED | 25 tests need a live MongoDB (`MONGO_URL`) and cannot run in CI or on a fresh machine. 136 tests pass without it. Orchestrator tests assert only `status in ("survived","rejected","error")`, which passes even if every trial errors. |
| TEST-2 | S2 | READ | No noise test, no known-edge test, no key-failover test, no resume test, no canary tests, no tests proving the sandbox blocks file, network, memory or time abuse. |
| OPS-1 | S3 | READ | No README, no `requirements.txt`, no `docs/METHODOLOGY.md`. `main.py forward` is a stub. Strategies are labelled "survived" instead of "candidate (unproven)". |
| OPS-2 | S3 | READ | Holdout run, forward/paper trader, decay monitor and diversity system (fingerprint, behaviour correlation, coverage map) are absent. |

## A10. What is good and must be preserved
- Data validation report and DST/session logic (`data_loader.py`).
- Frozen split boundaries with embargo, and the atomic holdout lock backed by a unique Mongo index (`splits.py`, `mongo.py`).
- Closed-bar signal and next-open fill model (the idea is right; LH-1 shows the implementation must be hardened).
- Config-driven thresholds and the secrets-not-in-config test.
- The 136 passing tests, except those that need rewriting (listed in TEST-1, LH-6, BT-12).

---

# PART B: THE ANTIGRAVITY PROMPT

=== BEGIN ANTIGRAVITY PROMPT ===

# QuantForge v2: Hardening, Completion and Autonomous Operation

You are a senior quant developer and security-minded research engineer. You are taking over an existing repository, `quantforge`, that was built phase by phase and now has to be repaired and completed. A full audit has been done (`docs/AUDIT.md`, defect IDs such as `SBX-1`, `BT-4`). Read `docs/AUDIT.md` completely before touching any code. The audit shows that the safety layer, which exists to reject fake strategies, can currently be defeated, sometimes by accident. Your job, in order of priority:

1. Make the safety layer impossible to cheat and make the backtester honest.
2. Complete the missing core of the product: the autonomous generate, backtest, feed back, improve loop.
3. Make that loop run forever until the user presses Stop in a professional dashboard, including two-key LLM failover and token tracking.
4. Add diversity control, the overfit suite, leaderboard statistics, holdout and forward testing.

The project goal has not changed: an engine that autonomously finds XAUUSD strategies (1h and 4h first) that are real, robust and likely to survive live trading. Most ideas SHOULD be rejected. Honest rejection beats pretty numbers. No engine can guarantee live profit; never claim it.

## 0. HARD PRINCIPLES (override everything, including speed)
1. A strategy must never be able to access future data, the validation split, the holdout, the filesystem, the network, the environment, or the engine's memory. This is enforced by process isolation and by never sending future data to the strategy process. It is NOT enforced by a regex, a builtins whitelist or trust in the LLM.
2. Signals are decided on a CLOSED bar; fills happen at the NEXT bar's open. No exceptions.
3. Costs (spread, slippage, commission, swap) are always applied exactly once, in realistic units. Gross-only numbers are never used for ranking.
4. The final holdout is locked. It is touched at most once per strategy, only through one audited function.
5. Every real backtest of a strategy variant increments a persistent trial counter. Statistics (DSR) use that counter. A lost trial record is a bug.
6. Errors are never swallowed silently. A strategy that crashes is reported with its traceback; a database write that fails is logged and retried; nothing that matters is hidden behind `except: pass`.
7. Everything is reproducible: seeds, data hash, code hash, config hash are stored with every result.
8. Secrets live only in `.env`. They never appear in logs, prompts, Mongo documents, error messages, or the dashboard. Keys are referred to by label (`key_1`, `key_2`) only.
9. Never weaken a test to make it pass. If a test fails, fix the code. If a test is wrong, say so in the Phase Report and explain.
10. Pick the more conservative option when unsure and document it.
11. Never claim a strategy is profitable or live-ready before the forward test (section 14).

## 0B. WORKING PROTOCOL (MANDATORY)

Work strictly phase by phase (F0, F1, ...). The previous build ran ahead of its own acceptance tests; that is not allowed again.

1. **Regression-first.** Before you fix a defect, write a test that reproduces it and show that it FAILS on the current code. `tests/repro/repro_exploits.py` already reproduces the worst ones; convert them into proper pytest tests under `tests/regression/`, one per defect ID. After the fix, the same test must pass. Keep these tests forever.
2. At the end of each phase, STOP and give a Phase Report containing: what was built (files, functions), the exact test commands and full pass/fail output, the defect IDs closed, anything skipped or uncertain (be honest), and `READY FOR NEXT PHASE: YES/NO`.
3. Wait for my message "approved, continue to phase N+1". Never continue on your own.
4. Never stub, mock silently, or skip a hard component. If something is hard, say so and ask.
5. Do not change code from an earlier approved phase without telling me what and why, and re-running that phase's tests.
6. Run the full test suite after every phase and include the summary.
7. Ask before adding any new paid or external service. Free pip packages listed in F0 are pre-approved.
8. Cross-platform: assume I may be on Windows. Code must run on Windows, macOS and Linux. Never rely on `SIGALRM`, `fork`, or `/tmp`. Use `pathlib`, `tempfile`, `psutil`, and `multiprocessing` with the `spawn` start method.

## 0C. SCOPE
- 1h and 4h only (`enabled_timeframes: ["1h","4h"]`). 5m and 15m stay disabled until I approve, and only after at least one strategy has passed every gate on 1h/4h.
- Keep timeframe-generic code.

## 1. TARGET ARCHITECTURE

```
quantforge/
  config.yaml  .env.example  requirements.txt  README.md  main.py
  docs/        AUDIT.md  METHODOLOGY.md  ARCHITECTURE.md  RUNBOOK.md
  scripts/     patch_hole.py (moved)  verify_alignment.py
  core/
    config.py  config_schema.py        # pydantic schema, strict, rejects duplicate YAML keys
    data_loader.py  splits.py  data_vault.py
    indicators.py                      # tested leak-free helpers (htf, pivots, structure)
    signals.py                         # Signal, SignalTape, validation
    costs.py                           # spread/slippage/commission/swap, bid/ask
    simulator.py                       # numba execution of a SignalTape (fast core)
    event_simulator.py                 # independent pure-Python engine for cross-checking
    analytics.py                       # MTM equity, monthly stats, all leaderboard stats
    sizing.py
  sandbox/
    runner.py                          # parent: spawns & talks to isolated child
    child_main.py                      # child: audit hooks, whitelisted namespace, bar loop
    policy.py                          # AST ban-list + allowed imports
  validation/
    static_checks.py  truncation.py  determinism.py  delay.py  permutation.py
    canaries/                          # deliberately cheating strategies + clean controls
    gates.py  overfit.py  regime.py  baselines.py  robustness.py
  llm/
    client.py  key_manager.py  usage.py  prompts/  schemas.py  diagnostics.py
    improve_loop.py  orchestrator.py
  diversity/  fingerprint.py  behavior.py  coverage.py
  engine/
    state.py  control.py  worker.py  supervisor.py  budgets.py  watchdog.py
  forward/  paper_trader.py  decay.py  feeds.py
  storage/  mongo.py  logger.py  repo.py
  dashboard/
    server.py  api/  static/ (app.js, styles.css, vendor/chart.min.js)  tui.py
  tests/  unit/  regression/  repro/  chaos/  canary/  e2e/
```

Process model:
- `python main.py up` starts the dashboard (FastAPI) and the supervisor. The engine worker does NOT start until I press Start in the dashboard (or pass `--autostart`).
- The dashboard writes the desired state (`running | paused | stopped`) into Mongo (`engine_control`). The supervisor spawns/stops the worker accordingly and restarts it if it crashes while desired state is `running`. The worker writes a heartbeat every 5 s into `engine_state`.
- Strategy code runs ONLY in short-lived isolated child processes started by the worker (F3).

---

## PHASE F0: Repo hygiene and test foundation
Closes: LLM-6, TEST-1, DB-2, DB-1, DB-3, OPS-1 (partial), UI-4 (CORS part)

Build:
1. `requirements.txt` with pinned minimum versions: pandas, numpy, scipy, numba, pyyaml, python-dotenv, pymongo, requests (or httpx; use ONE), tenacity, fastapi, uvicorn, psutil, scikit-learn, jinja2, pytest, pytest-timeout, mongomock, tiktoken (token estimate fallback), rich. Add `cryptography` only for F1b (ask first). Verify `pip install -r requirements.txt` on a clean venv.
2. `README.md`: setup (venv, `.env`, data folder), commands, how to run tests, how to start the dashboard.
3. Test mode without MongoDB: a fixture that uses `mongomock` when `MONGO_URL` is not set or `QF_TEST_MONGO=mock`. All 25 currently Mongo-dependent tests must run in CI without a database. Real-Mongo runs remain available with `QF_TEST_MONGO=real`.
4. Fix `DB-1`: never store counter documents and trial records in the same collection. Create `counters` (for the global trial counter) and `runs` (one document per pipeline run of a strategy variant); keep `strategies`, `backtests`, `gate_results`, `llm_calls`, `key_status`, `logs`, `coverage_map`, `holdout_access`, `leaderboard`, `forward_trades`, `engine_control`, `engine_state`, `engine_counters`, `cycles`. Add a migration that moves the old data safely. Add a test that inserts 1000 run documents and asserts all 1000 are stored and the counter equals the number of real backtests.
5. Fix `DB-2` and add a CLI smoke test for every `main.py` sub-command.
6. Strict config: pydantic schema for `config.yaml`, YAML loader that raises on duplicate keys and unknown keys, and a test for both. Add new config sections as later phases need them (never hardcode thresholds).
7. Replace every `except Exception: pass` in non-test code with logged handling (grep and list them in the report). The logger must count dropped Mongo log writes and expose the count.
8. `.gitignore`: add `*.bak`, `data/`, `logs/`, `reports/`, `.env`. Move `patch_hole.py` to `scripts/`.
9. Remove `Access-Control-Allow-Origin: *` immediately (full dashboard rebuild is F11).

Must pass: the full existing suite without Mongo; the regression tests for DB-1 and DB-2 (first shown failing on old code).

STOP and report.

## PHASE F1: Data integrity and the data vault
Closes: DATA-1, DATA-2, DATA-3, and prepares SBX-2

Build:
1. **Data lock.** At split freeze time, store each timeframe's SHA256 (file bytes) and row count in Mongo. `DataStore` refuses to serve any split if the current file hash differs from the frozen one, with a clear message. A `main.py refreeze --timeframe 1h --confirm "<phrase>"` command re-freezes and records an audit entry; refreezing invalidates all earlier holdout accesses and marks existing backtests `data_changed`.
2. **Patch review (`scripts/verify_alignment.py`).** Before any patch, detect the native bar grid of each file (set of start minutes-of-day in UTC and in NY time, DST aware) and report it. Rebuild `patch_hole.py` to: (a) aggregate 5m into the *same* grid as the native file, (b) validate on a month that exists in both sources (the existing `cross_timeframe_consistency`; require OHLC mismatch rate below a configured threshold, otherwise abort), (c) write to a NEW file (`XAUUSD_H1.patched.csv`), never overwrite the source, (d) report whether the patched range falls inside train, validation or holdout. If any patched bar falls in the holdout, mark those bars `synthetic` and EXCLUDE that range from the holdout, shifting the holdout boundary only through the audited refreeze path. Report this to me before applying.
3. Cache loaded and validated frames by file hash (in-process LRU) so a trial does not re-read the CSV.
4. **Data vault (F1b, ask me first because it adds the `cryptography` package).** After freeze, the validation and holdout rows are removed from the working CSVs and stored in an encrypted file; the key is held only in the parent engine process (loaded from `.env` `VAULT_KEY`). The strategy child process never has the key. This makes SBX-2 irrelevant even in the event of a sandbox escape. If I decline, document the residual risk in `docs/METHODOLOGY.md`.

Must pass: changing one byte in a CSV makes `DataStore` refuse; patch script refuses to overwrite; a misaligned synthetic test patch is detected; vault round-trip returns identical frames.

STOP and report.

## PHASE F2: Honest execution engine (signals, simulator, analytics)
Closes: BT-1 to BT-14, LH-5 (shift-based delay)

This is the heart of the repair. Design it exactly as follows.

### F2.1 Separate signal generation from execution
- A strategy (running in the sandbox, F3) produces a **SignalTape**: arrays over bars `0..N-1` with `action` (0 none, 1 enter long, 2 enter short, 3 close), `sl_distance`, `tp_distance`, `trail_distance`, `time_stop_bars`, `breakeven_after_r`, all as numbers (NaN = unused). The tape is cached by (code hash, params hash, data hash, split, window size).
- The **simulator** executes a tape against OHLC. Because signals depend only on past bars, one tape can be executed many times: shifted by 1 or 2 bars (delay test), with different costs (cost resilience), with random entries (baseline), under Monte Carlo, without ever re-running the strategy.
- Strategy code never sees positions or equity.

### F2.2 Signal contract and validation (`core/signals.py`)
Use **distances**, never absolute stop prices:
- `sl_distance > 0` is required. Minimum: `max(min_sl_usd, min_sl_atr_multiple * ATR_prev)`, both from config (defaults: $1.0 and 0.3 ATR). Maximum from config.
- `tp_distance` optional; if present it must be at least `min_tp_spread_multiple * current_spread` (default 3x) and at least `min_tp_usd`. This kills the "TP smaller than the spread" scalp.
- A legacy helper `Signal.from_prices(close, direction, stop, target)` converts absolute prices to distances and REJECTS a stop on the wrong side of price (this is the BT-1 exploit) with an explicit error.
- Invalid signals are not silently dropped. The tape records `invalid_signal` counts with reasons. If more than 2% of non-empty signals are invalid, or any NaN/inf appears, the run fails as `code_error` with a message sent back to the LLM.
- Max 6 tunable parameters (spec). Strategies receive parameters through `__init__(self, params)` (SBX-5) and declare `PARAMS = {name: {"default":..,"min":..,"max":..}}`.

### F2.3 Execution model (`core/simulator.py`, numba) and independent `core/event_simulator.py`
Rules (both engines must implement exactly these, written independently):
- **Price basis:** config `data.price_basis: bid` (default). Spread `s(t)` is in USD per oz and applied via bid/ask:
  - Long entry fills at `open_next + s + slippage`; long exit (SL/TP/close) fills at the bid path (no extra spread) minus slippage on stop orders.
  - Short entry fills at `open_next - slippage` (bid); short exit fills at the ask: `price + s`, plus slippage on stops.
  - SL/TP triggers: long exit triggers on the bid (`low <= sl`, `high >= tp`); short exit triggers on the ask (`high + s >= sl`, `low + s <= tp`).
  - Spread is therefore paid once per round trip, as in real life, and never subtracted a second time.
- **Entry bar is live.** From the entry bar onward SL and TP are checked on every bar including the bar that contains the fill. Within the entry bar only the part after the open applies; the open itself is the fill.
- **Same-bar SL and TP:** assume the stop was hit first. 
- **Gaps:** stop orders: if a bar opens beyond the stop level (long: `open <= sl`; short: `open + s >= sl`), fill at the OPEN (worse than the stop) plus stop slippage, never at the stop price. Otherwise, if the stop is touched inside the bar, fill at the stop price plus stop slippage. Take-profit (limit) orders: fill at the TP price even if the open gaps beyond it (conservative: the better gap price is not credited). Document both rules in `docs/METHODOLOGY.md`.
- **Trailing, breakeven, time stop:** evaluated at bar close using that bar's data; the new stop level applies from the NEXT bar. Never use the current bar's extreme to move a stop that can be hit within the same bar.
- **Close action:** exits at the next bar's open (bid or ask side as above).
- **Reversal:** configurable (`allow_reversal`, default true): an opposite entry signal closes at next open and opens the opposite side in the same bar; same-direction signals while in position are ignored; one position at a time.
- **Slippage:** `slippage = slip_atr_fraction * ATR_prev` (prior-bar ATR), larger for stop orders and news bars (config), never negative.
- **Spread table:** `costs.spread` in **USD per oz** (typical gold: London/NY $0.25, Asia $0.35, rollover $0.60+). Remove the `pip_value` multiplication (BT-4). Add a unit test asserting a London bar spread equals the config USD value. Optional news widening from `data/news_calendar.csv` if present.
- **Commission** per lot per side from config. **Swap** charged at the 17:00 New York rollover on trading days only (config long/short rates; Wednesday multiplier configurable), not at midnight UTC and not on weekends.
- **Sizing** (`core/sizing.py`): `lots = floor(risk_fraction * equity / ((sl_distance + spread + expected_slippage) * contract_size) / lot_step) * lot_step`. Round DOWN. If `lots < min_lot` the trade is skipped and counted (`skipped_min_lot`), unless the risk at min lot is within `1.5x` the budget. Apply `max_lot`.
- **Margin:** `required = lots * contract_size * price / leverage`. Skip the trade (count `skipped_margin`) if it exceeds `max_margin_utilisation * equity`. Implement margin-call stop-out at `stop_out_level`.
- **End of split:** close any open position at the last bar's close with costs and flag `forced_close`.
- **Gross vs net:** each trade stores gross PnL (zero-cost path) and each cost component separately, with `gross - spread - slippage - commission - swap = net` verified by assertion in tests. No cost is counted twice (BT-3).
- **Equity is mark-to-market per bar** (longs valued at bid, shorts at ask). Drawdown, Sharpe and everything in `analytics.py` use this MTM series (BT-5, BT-6).

### F2.4 Analytics (`core/analytics.py`)
From the MTM equity and the trade list compute, per split: total return, CAGR, daily-return Sharpe/Sortino (annualised with sqrt(252)), Calmar, Omega, max drawdown (MTM), average drawdown, max drawdown duration, time under water %, Ulcer index, recovery factor, profit factor, win rate, expectancy ($ and R), average win/loss, payoff, largest win/loss, consecutive streaks, average holding time, exposure %, trades per month, long/short breakdown, exit-reason breakdown, MAE/MFE, **monthly table (calendar months, compounded), average and median monthly return, best/worst month, % profitable months, average winning month, average losing month, average monthly max drawdown (mean over months of the worst intramonth MTM drawdown), yearly table**, and cost-to-gross ratio. Include buy-and-hold for the same period and the strategy's beta to it.

### F2.5 Independent cross-check
`event_simulator.py` is a bar-by-bar object-style implementation written WITHOUT importing simulator code. A differential test generates at least 30 random SignalTapes (different seeds, with and without trailing, time stops, gaps, Wednesday swaps, weekend gaps, same-bar SL/TP, margin skips) and requires identical trades and equity within `1e-6`. Any mismatch fails the phase. Also add 15 hand-computed trades in a table-driven test (write the arithmetic in comments).

### F2.6 Strategy exceptions (BT-12)
Exceptions from the strategy are returned with traceback by the sandbox (F3) and make the run status `code_error`. Rewrite the test that currently blesses swallowing.

Must pass (all as regression tests): BT-1 (inverted-stop strategy is rejected with an explicit error and can NOT show 100% win on a random walk), BT-2 (entry-bar crash hits the stop), BT-3/BT-4 (flat-market costs hand-computed), BT-5 (open-trade drawdown visible in max DD), BT-7 (gap fill), BT-9 (Wednesday/weekend swap), BT-14, differential test, hand-computed table, cost reconciliation `gross - costs = net` on 1000 random trades, performance: 3 years of 1h simulated in under 1 second per tape.

STOP and report.

## PHASE F3: Real sandbox (process isolation)
Closes: SBX-1 to SBX-6, LH-1, LH-3 (partially), BT-12 (transport)

Principle: **the strategy process never possesses future data, secrets or filesystem access.** Defence in depth, strongest first:

1. **Incremental feed.** The parent starts a child process and sends bars ONE AT A TIME over a pipe: after the child replies with the signal for bar `k`, the parent sends bar `k+1`. The child keeps only a bounded window (`max_lookback_bars`, default 1500 for 1h, 600 for 4h) and gives the strategy a fresh COPY (`DataFrame.copy()`, new buffers; never a view) of that window each call. There is no parent buffer to reach through `.base` (LH-1) and no full dataset in child memory.
2. **Process isolation.** `multiprocessing` with `spawn`. The child is started with `python -I`, an EMPTY environment (no `MONGO_URL`, no keys, no PATH tricks beyond what Python needs), a fresh empty temp working directory, and resource limits: POSIX uses `resource.setrlimit` (address space, CPU); Windows uses a Job Object if `pywin32` is present, otherwise a `psutil` watchdog thread in the parent that polls RSS every 100 ms. Per-bar timeout and total timeout are enforced in the PARENT with `Connection.poll(timeout)`; on timeout the parent kills the whole process tree. A run killed for time or memory is reported as `sandbox_timeout` or `sandbox_memory` with the bar index.
3. **Audit hooks.** In the child, after pre-importing the allowed libraries and doing a warm-up call, install `sys.addaudithook` that raises on: `open` (except read-only access to the Python stdlib and site-packages directories), `socket.*`, `subprocess.*`, `os.system/exec/spawn/fork/remove/rename/mkdir/chdir`, `ctypes.*`, `import` of any module outside the allow-list, `shutil.*`, `pickle`/`marshal` loads. Hooks cannot be removed from Python code. Keep the allow-list in `sandbox/policy.py` and cover it with tests.
4. **Restricted namespace.** Provide only a whitelisted subset of numpy/pandas through wrapper namespaces that exclude IO (`read_*`, `to_*`, `load`, `save`, `fromfile`, `memmap`, `ctypes`, `os`, `io`, `testing`, `compat`, `util`). Builtins whitelist without `type`, `object`, `getattr`, `setattr`, `delattr`, `vars`, `globals`, `locals`, `dir`, `eval`, `exec`, `compile`, `open`, `__import__`, `input`, `breakpoint`, `memoryview`.
5. **AST policy (`sandbox/policy.py`, applied before the code is sent to the child).** Reject, with line numbers: any attribute starting and ending with `__` (`__class__`, `__bases__`, `__subclasses__`, `__globals__`, `__builtins__`, `__dict__`, `__mro__`, `__code__`, `__getattribute__`, ...), `.base`, `.ctypes`, `.data`, `__array_interface__`, `f_back`, `tb_frame`, `gi_frame`, `cr_frame`, `.environ`; calls to `eval/exec/compile/open/getattr/setattr/__import__/globals/locals/vars`; imports outside {numpy, pandas, math, numba (optional), core.indicators helpers}; `shift` / `diff` / `pct_change` / `rolling` / `iloc` with negative or non-literal offsets that could be positive, including keyword forms (`periods=-1`, `center=True`); star imports; nested functions that assign `global`; `while True` without a bounded counter (warn, the timeout still protects). This is advisory defence; processes and hooks are the guarantee.
6. **Fail-closed protocol.** Any malformed reply, extra output, crash or unexpected message ends the run as `sandbox_error`. The parent validates every signal (F2.2).
7. **Determinism.** The child seeds `random` and `numpy.random` from a value sent by the parent; the same input must produce identical tapes. A determinism test (run twice, compare tapes bit-exact) is part of the lookahead suite (F4).
8. **Pool.** A pool of pre-warmed children (config `sandbox.pool_size`, default `cpu_count-1`) to avoid paying import time per run. Children are discarded after each run (no cross-run memory).

Escape tests (each is a canary in F4 and must be blocked, with the strategy source in `tests/canary/`): subclass-chain to `os`, `pd.read_csv` of the price file, `np.load`/`np.fromfile`, `pd.io.common.os.environ`, `.to_numpy().base`, `open()` read and write, `socket` connect, `subprocess` launch, `while True`, memory bomb (`np.zeros(10**10)`), recursion bomb, writing a file in the temp dir and reading it in a later run, `print`/`sys.stdout` flooding, `time.sleep` stalls, exceptions with huge messages.

Must pass: all escape tests; the three VERIFIED exploits SBX-1, SBX-2, SBX-3, SBX-4 and LH-1 from `repro_exploits.py` now report SAFE; honest strategies still run with the same signals as before (compare to a reference tape produced by an in-process run on test data); throughput: at least 2000 bars/second per child for a simple SMA strategy on Windows or Linux (report the measured number).

STOP and report. (Do not start F4 until I approve.)

## PHASE F4: Lookahead defence and canary tests (most important phase)
Closes: LH-2, LH-3, LH-4, LH-5, LH-6, GATE-6 (ordering)

Everything here runs on top of the F3 sandbox and F2 tapes.

### F4.1 Static policy
Reuse `sandbox/policy.py` (F3.5) as the first gate; keep the old source-pattern list but treat it as advisory only.

### F4.2 Truncation test (the real one)
1. Run the strategy on the full train split through the sandbox and keep the tape `T_full`.
2. Choose 20 cut points `c` (seeded, spread over the split, including the very last bar and one just after warm-up).
3. For each, start a NEW child, feed only bars `0..c`, and get `T_c`.
4. Require `T_c[0..c] == T_full[0..c]` for action, `sl_distance`, `tp_distance` and the other fields (equal, NaN-aware, tolerance 1e-9).
5. On a mismatch, report the first diverging bar index and timestamp and the field. This is lookahead (or non-determinism).
Delete the old Sharpe-CV implementation.

### F4.3 Determinism test
Run the full train feed twice with the same seed; the tapes must be bit-identical. Different results = unseeded randomness, time-dependence or hidden state = reject with the reason.

### F4.4 Delay test (on tapes)
Shift the tape by 1 and 2 bars (no strategy re-run) and simulate. Fail if net Sharpe drops by more than the configured percentage at +1 bar (default 60% on 1h/4h; stricter on 5m/15m later), or the +2 bar result is negative while the base is strongly positive. A non-profitable base strategy is rejected earlier by the performance gate; this test is not allowed to return an ambiguous pass.

### F4.5 Permutation / block-shuffle test
Block-shuffle the bar returns (block length from config) and rebuild synthetic price paths; run the strategy through the sandbox on at least 30 synthetic paths (cached per timeframe, seeded). The real net Sharpe must exceed the 95th percentile of the synthetic distribution.

### F4.6 Suspicion rules become hard gates
Net Sharpe above `suspicion.max_sharpe` on 1h/4h, profit factor above `suspicion.max_pf`, win rate above `suspicion.max_win_rate`, top-5% trades above 70% of profit, or almost no drawdown trigger a **deep audit**: rerun F4.2, F4.3, F4.4 with 40 cuts and a different seed, plus a manual-review flag. If the strategy is still suspicious, REJECT as `suspected_leak` (the rejection reason is sent back to the LLM as "result too good to be true; check for lookahead"). The strategy never proceeds to validation in that state.

### F4.7 Gate order
Cheapest and most safety-critical first: policy scan, compile/smoke run, determinism, truncation, then train backtest and basic performance, delay, suspicion audit, and only then the heavier robustness gates and finally the validation split. Lookahead checks must run BEFORE any validation data is used.

### F4.8 Canary suite (required, the foundation of trust)
Create `validation/canaries/` and `tests/canary/test_canaries.py`. Each canary is a deliberately cheating or abusive strategy, and the test asserts it is rejected BY THE EXPECTED GATE with the expected reason:

| # | Canary | Expected gate |
|---|---|---|
| 1 | `shift(-1)` on close | static policy |
| 2 | `shift(periods=-1)` keyword form | static policy |
| 3 | `rolling(5, center=True)` | static policy |
| 4 | `j = i + 1; df.iloc[j]` through a variable | static policy or truncation |
| 5 | `.to_numpy().base` future peek (LH-1) | static policy; and sandbox gives no parent buffer |
| 6 | `pd.read_csv` of the price file (SBX-2) | audit hook / sandbox_error |
| 7 | subclass-chain escape to `os` (SBX-1) | static policy; hook blocks if reached |
| 8 | read `MONGO_URL` from the environment (SBX-3) | blocked: environment is empty |
| 9 | infinite loop (SBX-4) | sandbox_timeout |
| 10 | memory bomb | sandbox_memory |
| 11 | network call | audit hook |
| 12 | write a file then read it in a later run | audit hook / fresh child |
| 13 | full-sample z-score using statistics of the whole series | cannot be built in the sandbox; the test proves the child never has the full series (assert the child's accessible window length) |
| 14 | unclosed higher-timeframe candle (uses the current partial HTF bar's high/low) | truncation (repaint) or `htf()` rule |
| 15 | pivot/swing used before it is confirmed | truncation |
| 16 | unseeded `random` signals | determinism |
| 17 | wrong-side stop (BT-1) | signal validation |
| 18 | `sl_distance = 0.01` with huge sizing (BT-14) | signal validation |
| 19 | TP smaller than the spread | signal validation |
| 20 | strategy that raises on every bar (BT-12) | `code_error` with traceback (not "no trades") |

Add at least 5 clean **control** strategies (SMA cross, RSI mean reversion, Donchian breakout, Asian-range breakout, ATR volatility filter), each written with `core/indicators` helpers where possible. Controls MUST pass F4.1 to F4.5 (they may fail performance gates; that is irrelevant here). This proves the checks are not simply rejecting everything.

The Phase Report MUST print a table: canary name, gate that caught it, rejection message, and the control strategies' results. If even one canary slips through, the phase FAILED.

**I will personally read the canary table and open at least two canaries before approving.** Also, I may invent my own canary; the system must catch it or the phase fails.

STOP and wait for my review.

## PHASE F5: Tested indicator and structure library (`core/indicators.py`)
Closes: DATA-4

Provide fast, vectorised, leak-safe helpers with docstrings the LLM can read. Every function takes only past-and-current data and is covered by a **property test** that proves it by truncation (value at bar `t` computed on `df[:t+1]` equals the value from the full run).

- Trend/momentum: `sma, ema, wma, macd, adx, supertrend, donchian, keltner, roc`
- Mean reversion/volatility: `rsi, stoch, bollinger, zscore, atr, true_range, realized_vol, vol_percentile, squeeze`
- Volume/price location: `session_vwap, anchored_vwap, previous_day_hl, previous_week_hl, session_range (Asia/London/NY), opening_range`
- Time features: `hour_utc, dow, is_session(name), minutes_since_session_open, month_end_flag`
- **`htf(df, "4h", func)`**: builds higher-timeframe bars from closed candles only; a HTF bar is exposed only after its last constituent bar has closed (the value for HTF bar k is available only from the first lower-TF bar after bar k ends). Include the grid-alignment handling from F1.
- **Confirmed pivots/swings:** `swing_high/low(n)` become known only `n` bars after they occur and the confirmation time is exposed (`pivot_value`, `pivot_bar`, `confirmed_at`).
- **Mechanical market structure:** `bos`, `choch`, `order_block`, `fvg` (fair value gap), `liquidity_sweep`, `premium_discount` with explicit numeric definitions and confirmation delays, documented in the docstring. No subjective versions.
- Regimes: `trend_range_regime` (ADX and efficiency ratio), `vol_regime` (percentile of past ATR), `session_regime`, all using only past data.

Add `docs/INDICATORS.md` listing every helper, its parameters and leak-safety notes; the prompts (F7) include this list so the LLM calls helpers instead of writing its own.

Must pass: truncation property test for every helper; numerical checks against known reference values; throughput test.

STOP and report.

## PHASE F6: LLM client, two-key failover, token tracking
Closes: LLM-1, LLM-2, LLM-5, LLM-6

### F6.1 Client (`llm/client.py`)
OpenAI-compatible `/chat/completions`, with `LLM_BASE_URL` and `LLM_MODEL` from `.env`; no hardcoded vendor. Timeout 120 s. Use one HTTP library only (declared in requirements).
- Strip `<think>...</think>` blocks and ignore `reasoning_content` when extracting content. Never include chain-of-thought in logs or Mongo beyond a truncated preview.
- If `finish_reason == "length"`, retry once asking the model to continue or with a higher `max_tokens` (config); if still truncated, return a typed `TruncatedResponse` so the caller re-asks for a shorter answer. `max_tokens` is configurable per purpose (ideate 3000, code 6000, improve 6000).
- JSON mode only if supported (config flag `supports_json_mode`); always validate with pydantic; on invalid JSON make up to 2 repair calls that include the validation error.
- Parse `usage`; if absent estimate with tiktoken (fallback: chars/4) and set `estimated=true`.

### F6.2 KeyManager (`llm/key_manager.py`)
Holds `LLM_API_KEY_1` and `LLM_API_KEY_2` by label. State per key persisted in Mongo `key_status` (survives restarts): `active | cooling_down(reset_at) | invalid | budget_exhausted`.
- Start on key 1 (or round-robin if `llm.alternate_keys: true`).
- On HTTP 429, quota-exceeded messages, "insufficient credits", `Retry-After`, or provider rate-limit codes: mark that key `cooling_down` until `Retry-After` (or config default), and IMMEDIATELY retry the SAME request on the other key. The cycle must not lose its place.
- On 401/403: mark `invalid`, raise a dashboard alert, use the other key.
- On 5xx/timeouts: exponential backoff with jitter (up to 5 tries) on the same key before counting a key problem.
- If both keys are unavailable: engine state becomes `WAITING_FOR_QUOTA` with the earliest reset time; the worker sleeps in short slices (checking the Stop command), then resumes automatically. It does not crash and does not record junk trials.
- Optional per-key daily token budget from config; reaching it behaves like a cooldown until 00:00 UTC.
- Never log or store the key itself, only the label. Redact keys from any exception message.

### F6.3 Usage tracking (`llm/usage.py`)
Every call inserts into `llm_calls`: timestamp, key label, model, purpose (`ideate|fix_code|fix_lookahead|improve|repair_json`), strategy_id, cycle_id, prompt_tokens, completion_tokens, total_tokens, latency_ms, status, error_class, estimated flag. Maintain running totals per key, per model, per day, per purpose and overall, in `key_status` / `engine_counters`, updated atomically. Expose them to the dashboard (F11). Optional cost estimate if prices per 1M tokens are in config.

Must pass (mock the HTTP layer): 429 on key 1 is retried on key 2 with the same request and no lost cycle; both keys limited leads to WAITING_FOR_QUOTA and resumes when the reset passes (use a fake clock); 401 marks the key invalid; cooldowns survive a restart; `<think>` is stripped; truncated output handled; token totals equal the sum of `llm_calls`; no key value appears in any log line or Mongo document (scan test).

STOP and report.

## PHASE F7: Strategy contract, prompts and the improve loop
Closes: LLM-3, LLM-4, BT-12 (feedback), SBX-5

### F7.1 Contract
The LLM returns ONE JSON object with `spec` and `code`:
- `spec`: `name`, `timeframe`, `hypothesis` (why the edge should exist and who is on the other side of the trade; required BEFORE code), `concept_family` (trend, mean_reversion, breakout, volatility, session, time_effect, market_structure, multi_timeframe, intermarket, event, regime, other), `indicators_used`, `entry_logic`, `exit_logic`, `filters`, `session_filter`, `direction`, `parameters` (name: default/min/max, hard cap 6), `expected_trades_per_year`, `expected_failure_conditions`.
- `code`: a class `Strategy` with `PARAMS`, `__init__(self, params)` and `on_bar(self, bars) -> Signal | None`, using `Signal` (distances) and helpers from `core.indicators`. Stateless between runs; seeded randomness only.
Validate with pydantic (`llm/schemas.py`). Reject specs with no hypothesis.

### F7.2 Prompt templates (`llm/prompts/`, jinja2)
- `ideate`: target coverage cell + timeframe, the contract, the helper list, the lookahead rules and the exact reasons past ideas were rejected (category only), the last 50 accepted and rejected ideas as `name | concept | one-line logic` (not names alone), explicit "avoid these logics".
- `fix_code`: the traceback (truncated to the relevant frames, no paths), the source, and the line numbers. Max 3 attempts; not counted as a trial.
- `fix_lookahead`: the exact check that failed, the diverging bar and field, and the rule it violated.
- `improve`: described below.
- `repair_json`: the validation error text.
Keep each prompt compact (diagnostics as small tables, never raw trades). Truncate the lowest-priority sections first. Low temperature (0.2) for fix steps, high (0.9) for ideation.

### F7.3 Improve loop (`llm/improve_loop.py`, `llm/diagnostics.py`)
This is the core feedback mechanism you asked for: backtest, show the LLM what happened, let it fix or improve, repeat.
1. After a strategy passes lookahead gates and the basic train run, build **train-only diagnostics**: net performance by year, month, session, weekday and hour; exit-reason breakdown; MAE/MFE summaries; holding-time distribution; long vs short split; cost impact; losing streaks; worst drawdown period; parameter sensitivity summary (if cheap). Never include validation or holdout numbers.
2. Ask the LLM for ONE targeted structural change with a stated reason (for example "add a volatility filter because 70% of losses occur in low-ATR hours"). Forbid blanket parameter tuning and adding parameters beyond the cap of 6.
3. Each revised version is a NEW trial (counter +1), is linked with `parent_id`/`version`, and must pass the same gates. A version that only changes constants is not a new idea but still counts as a trial.
4. At most 5 rounds per idea; stop after 2 rounds with no improvement in train expectancy-per-trade or train Sharpe (config).
5. Failures in robustness gates (parameter fragility, walk-forward, Monte Carlo) are NOT optimisation targets: the LLM receives only the category ("failed robustness: parameter fragility") and may attempt at most one structural rethink.
6. Validation outcomes: the LLM learns only "validation passed/failed" plus a coarse category. No numbers.
7. A crashed strategy or invalid signal gets the traceback via `fix_code` (max 3), never a silent "no trades".

Must pass: with a scripted fake LLM, a strategy with a planted weakness (for example it loses in a specific session) is improved in round 2 after receiving the diagnostics; the loop stops at 5 rounds or 2 non-improving rounds; versions are linked; validation numbers never appear in any prompt (scan test over all prompts built in the test run); a code error triggers `fix_code`, not rejection.

STOP and report.

## PHASE F8: Gate suite rebuilt and wired end to end
Closes: GATE-1 to GATE-7

All thresholds in `config.yaml`. All gates run on tapes through the F2 simulator unless they need a different parameter set (then a new sandbox run is made and cached). The orchestrator calls the gates through ONE function, `validation/gates.run_pipeline(...)`, which returns an ordered list of `GateResult(name, passed, category, details, public_summary)` where `public_summary` is the only part ever shown to the LLM (category only for robustness and validation gates).

Gate order (stop at the first failure; each failure is stored in `gate_results`):
1. policy scan, 2. smoke run, 3. determinism, 4. truncation, 5. signal validity,
6. train backtest with costs, 7. minimum sample, 8. basic quality, 9. delay, 10. suspicion audit,
11. beat baselines, 12. cost resilience, 13. parameter sensitivity, 14. walk-forward,
15. Monte Carlo, 16. regime and year concentration, 17. DSR, 18. validation (pass/fail only),
19. candidate promotion (holdout is a separate, manual step, F12).

Specifications:

**G7 minimum sample.** At least `min_trades` net trades on train (200 on 1h, 100 on 4h) and at least 30 per full calendar year (1h) / 15 (4h).

**G8 basic quality (net, MTM).** profit factor >= 1.25, daily Sharpe >= 0.8, max drawdown <= 25%, expectancy > 0, cost-to-gross <= 40%.

**G11 baselines.** (a) Random-entry baseline: 1000 random tapes with the same number of trades, the same session mix and the same exit-distance distribution, same costs; the strategy's net Sharpe AND net return must exceed the 95th percentile. (b) Random-direction variant at the strategy's own entry times. (c) Buy-and-hold comparison: regress strategy daily returns on buy-and-hold; report alpha and beta; reject if the strategy does not beat buy-and-hold Sharpe and its alpha is not significant (long-biased edge that just rides gold's bull trend). (d) Long-only and short-only reports; for two-sided strategies each side must have profit factor >= 1.0. For one-sided strategies, require non-negative performance in the sideways/down gold regimes of the sample.

**G12 cost resilience.** Costs x1.5 must still give profit factor >= 1.0; x2.0 is reported.

**G13 parameter sensitivity.** Perturb each parameter by +/-10%, 20%, 30% (respect min/max, round integers) plus 30 random joint perturbations. At least 70% of neighbours must be profitable with Sharpe >= 50% of the base Sharpe. Produce the 2-D heatmap for the two most influential parameters. A sharp peak (base Sharpe > 2x median neighbour Sharpe) is a rejection. Result category on failure: `parameter_fragility` (no numbers to the LLM).

**G14 walk-forward.** Rolling windows inside train (12 months context, 3 months out-of-sample, step 3 months). Require >= 65% of OOS windows profitable and the stitched OOS equity profitable with efficiency (OOS Sharpe / full-train Sharpe) >= 0.5. Optional true re-optimisation mode (`walk_forward.optimise`, default false): choose the best parameters on each in-sample window from a coarse grid (max 25 combos) and evaluate on the next OOS window. Count every grid point as trials.

**G15 Monte Carlo (5000 runs).** Fix the old implementation completely:
- bootstrap trades WITH replacement (this makes final equity vary);
- randomly (uniformly) drop 10% of trades in half of the runs;
- stress costs by re-simulating the tape with costs x1.5 and with fill noise (Gaussian, sigma = 0.1 ATR) in 200 runs;
- compute drawdown on the resampled equity path in the order generated (never sorted by value).
Pass: 5th percentile final return > 0, 95th percentile max drawdown <= limit. The test suite must show that the 5th-percentile final equity is NOT equal to the actual total PnL (regression for GATE-2).

**G16 regime and year concentration.** Label bars by trend/range x low/high volatility using past data only (`validation/regime.py`). Reject if more than 60% of net profit comes from one calendar year or one regime. Report profit by gold's trailing-12-month trend (up / flat / down).

**G17 Deflated Sharpe Ratio (correct).** Use per-period (daily) statistics, NOT annualised:
- `SR` = daily mean / daily std of MTM returns; `T` = number of daily observations; `g3` = skewness, `g4` = kurtosis (non-excess) of the daily returns.
- `N` = number of real trials so far (the persistent counter from F0).
- `V` = variance of the per-period Sharpe estimates of all stored trials (store `sr_daily` for every run). If fewer than 10 are stored, use `V = 1/(T-1)`.
- `SR0 = sqrt(V) * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e)))`, `gamma` = 0.5772156649; `SR0 = 0` when `N = 1`.
- `DSR = Phi( (SR - SR0) * sqrt(T - 1) / sqrt(1 - g3*SR + ((g4 - 1)/4) * SR^2) )`.
- Pass if `DSR >= 0.95`.
Unit tests: monotone decreasing in `N`; matches a worked reference example; an honest annualised Sharpe of 1.5 with 3 years of data must not collapse to probability 0 at N = 20 (reference value around 0.75 by hand calculation, which fails the 0.95 gate legitimately, but through the correct arithmetic).

**G18 validation.** Single run, no tuning. Require Sharpe >= 50% of train Sharpe, profit factor >= 1.15, minimum trades, drawdown within limit. The lookahead and determinism checks are re-run on the validation feed. Stored for the human leaderboard; the LLM gets only `passed` or `failed: category`.

**PBO (population-level, `validation/overfit.py`).** Real CSCV: build a matrix of aligned daily returns (rows = days, columns = the most recent >= 20 candidates that passed the basic gates), split into 16 blocks, enumerate (or sample 2000 of) the 12870 half/half combinations, choose the best in-sample column by Sharpe, compute its out-of-sample rank omega, logit lambda = ln(omega/(1-omega)); PBO = share of combinations with lambda <= 0. Recompute every 10 new candidates. If PBO > 0.5, set `population_overfit_warning` (shown in the dashboard) and tighten the DSR threshold to 0.99 for new promotions until it recovers. Remove the fake single-strategy PBO and make "not enough data" an explicit non-pass state, never a pass.

**Robustness score (0 to 100).** Weighted, weights in config: DSR 15, walk-forward 15, parameter plateau 15, Monte Carlo 15, regime breadth 10, cost resilience 10, delay resilience 10, beat-random percentile 10. Holdout and forward results are shown next to the score and never folded into it. Every component must be computed from data that now exists.

**Statuses.** `pending | rejected | candidate | holdout_passed | holdout_failed | forward_testing | live_ready | degraded`. After promotion a strategy is `candidate (unproven)` everywhere in the UI and logs. The word "survived" is removed.

**Two required validation tests of the gate suite itself (tests/e2e, slow marker):**
1. **Noise test:** 300 random-signal strategies on random-walk data with realistic costs. At least 99% must fail the full pipeline (including the correct DSR with a realistic trial count).
2. **Known-edge test:** a synthetic series with a planted, realistic, tradable edge (for example session-specific drift or return autocorrelation) and a strategy that exploits it must pass the pipeline in at least 80% of 20 seeds. This proves the gates are neither too loose nor so strict that nothing can ever pass.

Must pass: both tests above, plus a unit test per gate with a hand-made passing and failing example, plus `GATE-1`, `GATE-2`, `GATE-3`, `GATE-4`, `GATE-6` regression tests.

STOP and report.

## PHASE F9: Diversity system (never the same strategy twice)
Closes: OPS-2 (diversity part), LLM-4

1. **Logic fingerprint:** canonical JSON of (concept_family, sorted indicators, entry type, exit type, sorted filters, session, direction, timeframe) hashed with SHA-256. Exact duplicates are rejected BEFORE any backtest and do not count as trials. The rejection tells the LLM which stored strategy it duplicates.
2. **Code-structure fingerprint:** normalise the AST (rename identifiers and replace constants by placeholders), extract the multiset of (helper called, comparison/arithmetic operator, exit style) and compare with Jaccard similarity; >= 0.85 is a near-duplicate.
3. **Semantic similarity:** TF-IDF (scikit-learn) cosine on `hypothesis + entry_logic + exit_logic`; >= 0.80 flags a near-duplicate. If I approve a local embedding model (`sentence-transformers`, ask first), use cosine >= 0.88 instead.
4. **Behaviour fingerprint:** after the train backtest compute the correlation of aligned daily returns and the overlap of entry bars (+/- 1 bar) with every candidate; correlation > 0.7 means "same idea in disguise" and the strategy is marked `redundant` (counts as a trial, not promoted).
5. **Coverage map:** cells = concept_family x timeframe x session_bias x regime_bias. For each ideation choose the emptiest cell with 20% exploration. If a cell fails K times in a row (config), deprioritise it for a cooling period so tokens are not burnt on an unproductive cell. Persist in `coverage_map`; show it in the dashboard.
6. **Versions:** a revision created by the improve loop is compared against its own ancestors only by logic fingerprint (a constants-only change is not a new idea) and against everyone else by all of the above.

Must pass: a reworded copy of an existing idea is caught; a clearly different idea passes; a constants-only revision is flagged as the same logic; behaviour-correlation test with two strategies sharing the same entries; coverage chooses empty cells.

STOP and report.

## PHASE F10: Autonomous, never-stopping engine with state machine
Closes: ORCH-1 to ORCH-4

You asked for a loop that keeps generating strategies until you press Stop. That is the contract:

**The engine stops only when (a) the stored desired state becomes `stopped`, or (b) the process is killed. No other event may end the loop.** Finding a survivor does not stop it. An error does not stop it.

### F10.1 States (`engine/state.py`, persisted in `engine_state`, also visible in the dashboard)
`STOPPED, STARTING, RUNNING, PAUSED, STOPPING, WAITING_FOR_QUOTA, BUDGET_PAUSED, ERROR_BACKOFF, WAITING_FOR_DB, BLOCKED`
- `BLOCKED` (for example data validation failed, no API keys configured, frozen hash mismatch) shows a human-readable reason and keeps re-checking every 10 s, so the loop resumes by itself when the problem is fixed. It never exits.
- `WAITING_FOR_DB`: MongoDB unreachable; the worker retries with backoff and resumes.

### F10.2 Control and supervision
- `engine_control` document: `desired_state` (`running|paused|stopped`), `requested_by`, `requested_at`, plus optional `force_kill_at`.
- `supervisor.py` runs next to the dashboard: it spawns the worker when `desired_state = running`, restarts it with backoff if it crashes (record the crash and traceback), and stops it on `stopped`. Graceful stop: the worker finishes the current atomic step (a running backtest) or aborts an in-flight LLM call, marks the cycle `aborted_by_user` (not counted as a trial), then exits within `engine.graceful_stop_seconds`; after that the supervisor kills the process tree. A separate "Force stop" kills at once.
- `worker.py` writes a heartbeat every 5 s. The watchdog marks the engine `stale` in the UI after 30 s without a heartbeat and the supervisor restarts it.
- Single-instance lock (Mongo lease with TTL) so two workers can never run at once.

### F10.3 The cycle (persisted in `cycles`, resumable)
`pick_target -> ideate -> duplicate_check -> code_check -> lookahead_gates -> train_backtest -> improve_loop -> robustness_gates -> validation -> promote -> log -> next`.
- After each stage the cycle document is updated with the stage, artefacts (idea, code hash, attempts) and the LLM call ids. On restart the worker resumes unfinished cycles at the last completed stage, or aborts them cleanly if the artefacts cannot be trusted.
- The trial counter is incremented exactly once per real backtest through an idempotency key `(cycle_id, version)`, so a resume can never double count.
- Between cycles: refresh the leaderboard, update the coverage map, recompute PBO when due, persist counters.

### F10.4 Error taxonomy and behaviour (each has a test)
| Event | Behaviour |
|---|---|
| LLM transient (5xx, timeout) | backoff with jitter, retry the same step |
| LLM rate limit / quota | switch key; both limited -> `WAITING_FOR_QUOTA` until the earliest reset |
| LLM invalid JSON / truncated | repair or re-ask (bounded), then abandon the idea (not a trial) and continue |
| Strategy code error | `fix_code` loop (max 3), then reject as `code_error` and continue |
| Lookahead failure | `fix_lookahead` (max 3), then reject and continue |
| Sandbox timeout / memory | reject as `sandbox_*`, continue |
| Mongo down | `WAITING_FOR_DB`, retry forever with capped backoff, never lose the in-memory cycle |
| Unhandled exception in a cycle | log traceback to Mongo and JSONL, mark cycle `failed_internal`, continue with the next cycle; after 20 consecutive internal failures go to `ERROR_BACKOFF` (backoff up to 15 min) and keep trying |
| Daily token / call / trial cap reached | `BUDGET_PAUSED` until 00:00 UTC, resume automatically |
| Data / config problem | `BLOCKED` with the reason, keep re-checking |

No fixed `sleep(2)` after errors. A rate limit must never be recorded as a trial.

### F10.5 Concurrency
LLM calls are sequential through one rate limiter (config `llm.max_concurrent`, default 1). Backtests and robustness runs use the sandbox pool (`engine.max_parallel_backtests`). Optional `engine.pipeline_concurrency` (default 1) lets several ideas be in different stages at once; leave it at 1 until everything else is proven.

### F10.6 Logging
Use `QuantForgeLogger` everywhere (no `print`). Log every stage transition, every LLM call (purpose, key label, tokens, latency), every gate result with its reason, every state change, and every control action, with `cycle_id`, `stage`, `strategy_id`. Mongo plus rotating JSONL. Surface the count of dropped log writes.

### F10.7 Chaos and soak tests (`tests/chaos`)
Using a fake LLM and mongomock (and real Mongo when available): kill -9 the worker mid-cycle then verify resume and no double-counted trial; Mongo unavailable for 60 s then verify recovery; LLM 500 storm; both keys limited then verify wait and resume; invalid keys; corrupted CSV leads to `BLOCKED` then recovers when fixed; Stop during an LLM call and during a backtest; pause/resume; two worker start attempts (second refuses). **Soak:** run 2 hours (or 5000 accelerated fake cycles) and assert no unbounded memory growth, no leftover child processes (check with `psutil`), and monotonic counters.

Must pass: all of the above, plus a test proving the loop continues after a survivor is found and keeps going until `desired_state = stopped`.

STOP and report.

## PHASE F11: Professional dashboard with engine controls
Closes: UI-1 to UI-5, ORCH-2 (visibility)

Replace the single-page `http.server` dashboard. The old page was a basic, read-only table; the new one must feel like a real trading-research console.

### F11.1 Tech
FastAPI + uvicorn (threaded, async-safe Mongo access with `motor` is NOT required; use pymongo in a thread executor). Static front end: vanilla JavaScript ES modules + CSS variables, with Chart.js vendored into `dashboard/static/vendor/` (no CDN at runtime, works offline). No build step. Live updates through Server-Sent Events at `/api/stream` (state, counters, new log lines, LLM usage) every 1 to 2 seconds; polling fallback. Responsive layout (desktop first, usable on a phone). Dark theme by default with a light toggle. Numbers in tabular monospace, consistent 8-px spacing grid, clear colour semantics (green pass, red reject, amber waiting, blue info, grey neutral), skeleton loaders, explicit empty and error states.

### F11.2 Security (mandatory because it has control buttons)
Bind to `127.0.0.1` by default. Token login: `DASHBOARD_TOKEN` (generated on first run, printed once, kept in `.env`); HttpOnly SameSite=Strict cookie; CSRF header on every POST; Host and Origin validation; NO CORS wildcard; rate limit on control endpoints; every control action written to an audit log. Errors returned to the browser are generic; details go to logs. No secret (keys, Mongo URL, vault key) is ever sent to the browser.

### F11.3 Pages
1. **Overview.**
   - A large engine card: state pill (STOPPED, RUNNING, PAUSED, WAITING FOR QUOTA with a countdown, BUDGET PAUSED, ERROR BACKOFF, BLOCKED with reason), uptime, current cycle id, current stage, current strategy name, heartbeat age.
   - **Control buttons: Start, Pause/Resume, Stop (graceful, with a confirm dialog), Force stop (separate, red, double confirm).** Buttons are disabled in states where they make no sense. A visible "Autonomous loop: runs until you press Stop" note.
   - Counters: ideas generated, duplicates rejected, code errors, lookahead rejections, failed train gates, failed robustness gates, failed validation, redundant, candidates, ideas per hour, average cycle time.
   - **Funnel chart** (how many strategies survive each gate) and a **trials-per-hour** chart.
   - LLM mini panel (see 4), top-10 leaderboard, alert strip (key invalid, quota wait, data hash changed, DB down, `population_overfit_warning`).
2. **Leaderboard.** Table of candidates with a split switcher (Train, Validation, Holdout, Forward, Overall in-sample). Column groups: Returns (total, CAGR, average monthly, median monthly, best/worst month, % profitable months, average winning/losing month), Drawdown (max, average monthly max drawdown, average drawdown, max duration, time under water, Ulcer, recovery factor), Risk-adjusted (Sharpe, Sortino, Calmar, Omega, DSR probability, monthly volatility), Trades (count, per month, win rate, profit factor, expectancy in R and $, payoff, streaks, holding time, exposure, long/short), Robustness (score 0 to 100, walk-forward efficiency, Monte Carlo P5 return and P95 drawdown, plateau score, delay resilience, cost/gross, beat-random percentile, regime breadth), Meta (name, timeframe, concept family, version/parent, status, date, trial count at discovery). Sort by any column, column chooser, filters (timeframe, family, status, minimum trades, minimum score), search, CSV export, pagination. Default ranking: robustness score. Status badges say "CANDIDATE (unproven)" until the forward test is complete.
3. **Strategy detail.** Tabs: Summary (verdict, key stats, gate timeline), Equity (train / validation / holdout / forward drawn in distinct colours, with drawdown underneath), Monthly (calendar heatmap plus table, average monthly return and average monthly drawdown), Yearly, Trades (paginated, MAE/MFE scatter, exit-reason breakdown), Robustness (Monte Carlo fan, parameter heatmap, walk-forward windows, random-baseline histogram with the strategy marked, cost-stress table), Code and Hypothesis (read-only, highlighted), LLM history (prompts and responses with secrets redacted, version tree), Logs.
4. **LLM and Tokens.** Model name and endpoint host. One card per key (`key_1`, `key_2`): state (active / cooling down until HH:MM:SS with countdown / invalid / budget exhausted), tokens used (prompt, completion, total) for this session, today and all time, calls, failures, average latency, key-switch count. Charts: tokens over time by key and by purpose (ideate, fix_code, fix_lookahead, improve). Table of the latest 100 calls. Estimated cost if prices are configured. Editable daily token budgets per key.
5. **Live log.** SSE tail with filters (level, stage, strategy), pause-scroll, search, download.
6. **Coverage map.** Heatmap of concept x timeframe x session x regime with counts; click a cell to list its strategies; shows cooled-down cells.
7. **Data and integrity.** Validation reports, frozen file hashes, split boundaries, holdout access audit (strategy, time), patched or synthetic ranges, canary-suite status (last run, pass/fail).
8. **Settings.** Runtime options that are safe to change (token budgets, parallelism, enabled concept families, timeframes) validated and stored in `runtime_overrides`. Gate thresholds are shown read-only and marked locked (changing thresholds after seeing results is forbidden). 

### F11.4 API
`GET /api/stream`, `GET /api/engine/state`, `POST /api/engine/start|pause|resume|stop|force_stop`, `GET /api/stats/funnel`, `GET /api/leaderboard?split=&sort=&filters=`, `GET /api/strategies/{id}` (+ `/equity`, `/monthly`, `/trades`, `/gates`, `/llm`, `/logs`), `GET /api/llm/usage`, `GET /api/logs`, `GET /api/coverage`, `GET /api/data`, `GET /api/settings`, `PATCH /api/settings`. List endpoints are paginated and projection-limited; equity series are downsampled to at most 1500 points; Mongo indexes exist for every query.

### F11.5 Terminal dashboard (secondary)
`python main.py tui` (Rich/Textual) mirrors the Overview page: engine state, counters, funnel, LLM panel (model, active key, per-key state, tokens), log tail, top-10 leaderboard, plus start/stop key bindings that use the same control document.

Must pass: API tests with a test client (auth required, CSRF enforced, no CORS wildcard, no secret in any response); displayed token totals equal the database totals; leaderboard sorting and filtering are correct; the Start button results in `desired_state=running` and the supervisor starting the worker; Stop produces a graceful stop; state transitions are shown; a Playwright (or equivalent) smoke test that loads each page without console errors, if available. Provide screenshots of every page in the Phase Report.

STOP and report.

## PHASE F12: Holdout and forward (paper) testing
Closes: OPS-2, OPS-3

1. **Holdout run** (one time per strategy): triggered by `main.py holdout --strategy <id>` or by a dashboard button with a confirmation; `auto_run_holdout` is false by default. Only strategies that passed every earlier gate are eligible. The audited path in `DataStore` still records the access atomically; the strategy runs in the sandbox with the holdout fed incrementally. Require profitable result and Sharpe >= 50% of train. A failure marks `holdout_failed` and the holdout is burned for that strategy forever. The LLM is never told the holdout numbers.
2. **Paper trader** (`forward/paper_trader.py`): runs `holdout_passed` strategies on NEW data as closed bars arrive through a pluggable feed (`forward/feeds.py`; implement a folder/CSV watcher first and a clearly marked stub for MT5 or a broker API). It uses the same sandbox, signal validation and simulator costs, and records `forward_trades` with the actual simulated spread and slippage paid.
3. **Decay monitor:** compare rolling forward results with the confidence band from bootstrapping the backtest trades. If the rolling mean R falls below the lower band for N trades, mark `degraded` automatically.
4. **LIVE-READY gate (non-negotiable):** a strategy can be labelled `live_ready` only after the forward test has run at least 60 calendar days AND at least 50 trades with results inside the expected band. These floors are hard-coded as minimums; the config cannot go below them, and no UI button, flag or manual override can shorten them. Until then every view, report and log says "CANDIDATE (unproven)" or "FORWARD TESTING".
5. Forward results are shown in their own leaderboard columns and are never given to the LLM as an optimisation target.
6. The system only produces signals and reports. It does not place real orders. A separate execution module would need my explicit approval.

Must pass: simulated forward feed; correct status transitions candidate -> holdout_passed -> forward_testing -> live_ready / degraded; a strategy cannot become `live_ready` before 60 days and 50 trades (test with a fake clock and with a config attempting a lower value); second holdout access still raises.

STOP and report.

## PHASE F13: Documentation, full dry run, real run
1. `docs/METHODOLOGY.md`: every gate, formula (DSR, CSCV/PBO, Monte Carlo), execution rule (bid/ask, gaps, swaps, sizing, margin), and why it exists; the gap/stop/TP rules; the residual risks (for example if the vault was declined).
2. `docs/ARCHITECTURE.md` (process model, collections, state machine) and `docs/RUNBOOK.md` (how to start, stop, recover, rotate keys, re-freeze data).
3. Full dry run with the fake LLM on synthetic data, then on the real 1h/4h data with the real LLM keys for a small bounded session (run it from the dashboard using the Start button, then press Stop and show the graceful stop).
4. Final report: funnel counts, rejection reasons, top candidates with their gate history, token usage, and an **honest limitations section**. State plainly if the data or the results do not support a conclusion. If the first real run finds nothing, that is a valid result; do NOT loosen gates to get a survivor.

STOP.

---

## CONFIG ADDITIONS (summary; everything configurable, nothing hardcoded)
`costs.spread` in USD (London/NY 0.25, Asia 0.35, rollover 0.60 as starting values, to be checked against your broker); `costs.slip_atr_fraction`; `costs.commission_per_lot_side`; `costs.swap` (long/short, Wednesday multiplier, rollover 17:00 New York); `data.price_basis: bid`; `signals` (min SL USD and ATR multiple, max SL, min TP spread multiple, max invalid share, `max_lookback_bars` per timeframe); `sandbox` (pool size, per-bar and total timeouts, memory limit, backend: process|docker); `llm` (max_tokens per purpose, supports_json_mode, alternate_keys, daily token budgets, cooldown default, max_concurrent, prices per 1M tokens); `engine` (graceful_stop_seconds, heartbeat, backoff caps, max_parallel_backtests, pipeline_concurrency, internal-failure threshold); `gates` (all thresholds listed above, suspicion thresholds, DSR threshold 0.95 and the tightened 0.99, PBO settings, Monte Carlo settings, sensitivity percentages, walk-forward windows, baseline runs); `diversity` (similarity thresholds, behaviour correlation 0.7, coverage exploration 0.2, cooldown); `forward` (minimum days 60 and trades 50 with hard floors in code); `dashboard` (host, port, token env name, SSE interval).

## FINAL ACCEPTANCE CHECKLIST (I will verify these myself)
1. `tests/repro/repro_exploits.py` reports SAFE for every exploit; the regression tests exist and pass.
2. The canary table shows every canary caught by the expected gate, and the clean controls pass the lookahead gates.
3. Differential test: the numba simulator and the event simulator agree on 30+ random tapes; the hand-computed table passes.
4. A random-walk dataset gives no profitable strategy in the noise test; the known-edge test finds the planted edge.
5. Pressing Start in the dashboard starts the autonomous loop; it keeps running after a survivor appears, after an LLM error, after a key hits its limit (it switches to the other key and the dashboard shows it), and after the worker is killed (the supervisor restarts it); pressing Stop ends it gracefully.
6. The dashboard shows the model name, the active key, each key's state and countdown, and token totals that equal the database sums.
7. No secret appears in logs, Mongo or API responses (scan tests).
8. All tests pass on a fresh machine without MongoDB (mongomock) and with real MongoDB.
9. No `except Exception: pass` remains in non-test code.
10. Nothing in the UI or logs calls a strategy profitable or live-ready before the forward test.

## ABSOLUTE RULES FOR YOU (THE BUILDER)
- Follow section 0B. Stop after every phase and wait for my approval.
- Regression-first: failing test, then the fix, then the passing test, for every defect ID.
- Do not simplify or remove any gate to make results look better. If something is slow, optimise it.
- Do not tune thresholds after seeing which strategies pass.
- Never expose validation or holdout numbers to the LLM. Never expose any secret anywhere.
- Never run strategy code in the engine process. Never give a strategy process future data.
- The Phase F4 canary table is the foundation: show it, and do not continue until I confirm.
- 1h and 4h only until I approve otherwise.
- Never claim profitability or live-readiness before the forward test.
- If you are unsure, choose the more conservative design, document it in `docs/METHODOLOGY.md`, and tell me.
- Be honest in every report. "This does not work yet" is an acceptable and valued answer.

=== END ANTIGRAVITY PROMPT ===
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-04T20:32:41+05:30.

The user's current state is as follows:
Active Document: c:\Users\vedan\Desktop\ELVS LABS VERSION2.0\quantforge\tests\repro\repro_exploits.py (LANGUAGE_PYTHON)
Cursor is on line: 1
Other open documents:
- c:\Users\vedan\Desktop\ELVS LABS VERSION2.0\quantforge\tests\repro\repro_exploits.py (LANGUAGE_PYTHON)
- c:\Users\vedan\Desktop\ELVS LABS VERSION2.0\quantforge\config.yaml (LANGUAGE_YAML)
No browser pages are currently open.
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from Gemini 3.1 Pro (High) to Claude Opus 4.6 (Thinking). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>