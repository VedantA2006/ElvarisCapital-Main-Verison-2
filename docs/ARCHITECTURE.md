# QuantForge System Architecture & Engineering Specification

## 1. System Architecture Overview

QuantForge is structured as a fault-tolerant, distributed multi-process system with clear separation of concerns between **Engine Supervision**, **Autonomous Discovery Work**, **Secure Strategy Sandboxing**, and **Web/Terminal Control Consoles**.

```
                           +---------------------------+
                           |   FastAPI Web Console     |
                           |   & Terminal Rich TUI     |
                           +-------------+-------------+
                                         |
                                         | SSE / REST / CSRF
                                         v
+------------------------+  Lease / State  +------------------------+
|   Engine Supervisor    |<--------------->|  MongoDB Persistence   |
+-----------+------------+                 +-----------+------------+
            |                                          ^
            | Spawns / Monitors Heartbeat              | Telemetry & Cycles
            v                                          |
+------------------------+                             |
|  Engine Worker Loop    +-----------------------------+
+-----------+------------+
            |
            | Subprocess Execution
            v
+------------------------+
| Deterministic Sandbox  |
+------------------------+
```

---

## 2. Process Model and Concurrency

To ensure stability, security, and split-brain prevention, QuantForge separates responsibilities across independent OS processes:

### 2.1 The Supervisor Daemon (`engine/supervisor.py`)
- Responsible for process lifecycle management of the worker.
- Periodically checks desired execution state in `engine_control` (`running`, `paused`, `stopped`).
- Spawns the worker subprocess when `desired_state = running`.
- Monitors worker heartbeats every $5\text{s}$; initiates graceful restart with exponential backoff if worker crashes or heartbeat ages beyond $30\text{s}$.
- Manages graceful shutdown: waits for active trial completion up to `graceful_stop_seconds` (default $30\text{s}$) before sending `SIGTERM`/`SIGKILL`.

### 2.2 The Autonomous Worker (`engine/worker.py`)
- Executes the continuous discovery loop contract (`run_until_stopped`).
- Acquires and periodically renews an atomic TTL lease lock (`singleton_worker` in `engine_lease`).
- Updates heartbeat in `engine_state` every $5\text{s}$.
- Executes the 12-stage resumable cycle pipeline.
- Handles rate limits and quota resets without crashing or terminating the process.

### 2.3 The Sandbox (`core/sandbox.py`)
- Executes generated Python strategy code in an isolated subprocess.
- Enforces strict memory caps ($512\text{MB}$), per-bar execution timeouts ($50\text{ms}$), and global run timeouts ($10\text{s}$).
- Neutralizes dangerous builtins (`__import__`, `open`, `exec`, `eval`, `os`, `sys`, `socket`).

### 2.4 The Dashboard Async Server (`dashboard/fastapi_app.py`)
- High-performance asynchronous FastAPI + Uvicorn server bound to `127.0.0.1`.
- Provides `/api/stream` Server-Sent Events (SSE) broadcasting telemetry every $1.5\text{s}$.
- Exposes secure engine control and telemetry endpoints.

---

## 3. The 10-State Distributed State Machine

The worker's execution lifecycle is governed by an explicit 10-state machine persisted in `engine_state`:

```
                 +-----------+
                 |  STOPPED  |<-----------------------+
                 +-----+-----+                        |
                       | user start                   |
                       v                              |
                 +-----------+                        |
                 | STARTING  |                        |
                 +-----+-----+                        |
                       | lease acquired               |
                       v                              |
       +------------> RUNNING <------------+          |
       |               | | |               |          |
       | user resume   | | | pause         |          | force kill
       |               | | v               |          | / stop
       |         +-----+-----+-------+     |          |
       |         |      PAUSED       |     |          |
       |         +-------------------+     |          |
       |                                   |          |
       | quota reset         429 limit     |          |
       +------------- WAITING_FOR_QUOTA <--+          |
       |                                   |          |
       | 00:00 UTC          daily cap hit  |          |
       +------------- BUDGET_PAUSED <------+          |
       |                                   |          |
       | retry OK           consec errors  |          |
       +------------- ERROR_BACKOFF <------+          |
       |                                   |          |
       | reconnect          mongo down     |          |
       +------------- WAITING_FOR_DB <-----+          |
                                           |          |
                       +-------------------+          |
                       | graceful stop                |
                       v                              |
                 +-----------+                        |
                 | STOPPING  |------------------------+
                 +-----------+
```

### State Specifications:
1. `STOPPED`: Worker is inactive; desired state is `stopped`.
2. `STARTING`: Worker is initializing database indexes, validating data splits, and acquiring lease.
3. `RUNNING`: Worker is actively executing strategy discovery cycles.
4. `PAUSED`: User requested temporary pause; worker idles without releasing cycle state.
5. `STOPPING`: Graceful stop initiated; finishing current in-flight atomic backtest.
6. `WAITING_FOR_QUOTA`: LLM rate limit encountered ($429$); waiting for exponential reset window.
7. `BUDGET_PAUSED`: Daily token or trial budget exhausted; sleeps until next $00:00$ UTC.
8. `ERROR_BACKOFF`: Encountered $20$ consecutive internal failures; backing off with jittered retry.
9. `WAITING_FOR_DB`: MongoDB connection interrupted; preserving active cycle in memory while retrying.
10. `BLOCKED`: Data validation or critical configuration failure; logs diagnostic reason and pauses.

---

## 4. The 12-Stage Resumable Cycle Pipeline

To eliminate lost work during restarts and prevent duplicate trials, each discovery iteration is partitioned into 12 checkpointed stages saved to the `cycles` collection:

$$\text{pick\_target} \to \text{ideate} \to \text{duplicate\_check} \to \text{code\_check} \to \text{lookahead\_gates} \to \text{train\_backtest} \to \text{improve\_loop} \to \text{robustness\_gates} \to \text{validation} \to \text{promote} \to \text{log} \to \text{next}$$

- **Resumption Invariant:** If the worker crashes or restarts mid-cycle, it reads the active cycle from `cycles` and resumes from the exact stage last checkpointed.
- **Idempotency Guard:** Every evaluated strategy version checks `trial_idempotency` with key `(cycle_id, version)` to ensure trials are never double-counted in trial statistics or DSR calculations.

---

## 5. Single-Instance TTL Lease Lock

To guarantee that multiple competing engine workers never run simultaneously on the same database:
- The worker creates a unique index on `engine_lease._id` with a $15\text{s}$ TTL.
- The worker acquires the lease `singleton_worker` with an atomic insert:
  ```json
  {
    "_id": "singleton_worker",
    "worker_id": "worker-1",
    "acquired_at": 1700000000.0,
    "expires_at": 1700000015.0
  }
  ```
- The worker renews the lease every $5\text{s}$. If a secondary process attempts startup, it fails with `LockAcquisitionError`.

---

## 6. Database Collections Schema

| Collection | Key Fields | Purpose |
|---|---|---|
| `engine_state` | `_id="current_state"`, `state`, `worker_id`, `last_heartbeat`, `reason` | Real-time worker state and watchdog telemetry |
| `engine_control` | `_id="desired_state"`, `desired_state`, `requested_by` | Command channel from UI/CLI to Supervisor |
| `engine_lease` | `_id="singleton_worker"`, `worker_id`, `expires_at` | Atomic single-instance worker lock |
| `cycles` | `_id=cycle_id`, `stage`, `target`, `checkpoint_data` | 12-stage resumable cycle state |
| `trial_idempotency` | `_id="(cycle_id, version)"`, `recorded_at` | Prevents duplicate trial counting |
| `trials` | `trial_id`, `strategy_name`, `status`, `rejected_at`, `gate_results` | Complete ledger of all generated strategies |
| `candidates` | `strategy_id`, `source_code`, `robustness_score`, `train_metrics` | Qualified candidate strategies |
| `split_boundaries` | `timeframe`, `train_start`, `train_end`, `file_hash` | Frozen split partitions and hash locks |
| `holdout_accesses` | `strategy_id`, `timeframe`, `timestamp`, `triggered_by` | Immutable holdout access audit log |
| `forward_trades` | `strategy_id`, `trade_id`, `entry_time`, `exit_time`, `net_pnl` | Forward paper trading execution trades |
| `llm_calls` | `timestamp`, `key_label`, `model`, `purpose`, `total_tokens` | Comprehensive LLM token ledger |
| `llm_usage` | `_id="global"`, `total_tokens`, `total_calls`, `failed_calls` | Global running token counters |
| `audit_logs` | `action`, `requested_by`, `timestamp`, `details` | Security and control audit log |
| `runtime_overrides`| `_id="current"`, `daily_token_budget`, `max_parallel_backtests` | Safe runtime overrides |

---

## 7. Security Architecture

1. **Origin Validation:** Wildcard CORS (`*`) is strictly forbidden. Allowed origins are locked to `http://127.0.0.1:8000` and `http://localhost:8000`.
2. **Session Cookies:** `qf_token` session cookie is configured with `HttpOnly=True` and `SameSite=Strict`.
3. **CSRF Protection:** Mutation endpoints (`POST`, `PATCH`) require a cryptographically valid `X-CSRF-Token` header.
4. **Secret Projection:** Database connection strings, API keys, and vault encryption keys are scrubbed and projected out of all API responses and log outputs.
5. **Locked Gate Protection:** Thresholds for rejection gates (`min_sharpe`, `min_profit_factor`, `max_drawdown`, `pbo_max`) cannot be modified at runtime.
