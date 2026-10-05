# QuantForge Operational Runbook & Administrator Guide

## 1. System Setup and Prerequisites

### 1.1 Requirements
- **Python:** 3.11 or higher
- **MongoDB:** 6.0+ running on `localhost:27017` (or configured via `MONGO_URL`)
- **Operating System:** Windows, Linux, or macOS

### 1.2 Installation
```bash
# Clone the repository
git clone https://github.com/VedantA2006/ElvarisCapital-Main-Verison-2.git quantforge
cd quantforge

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 1.3 Configuration (`.env`)
Create `.env` in the project root:
```ini
MONGO_URL=mongodb://localhost:27017
MONGO_DATABASE=quantforge
GEMINI_API_KEY_1=AIzaSyYourPrimaryApiKeyHere
GEMINI_API_KEY_2=AIzaSyYourFailoverApiKeyHere
DASHBOARD_TOKEN=qf_secret_console_token_12345
```

---

## 2. Common CLI Commands

| Operation | Command | Description |
|---|---|---|
| **Data Verification** | `python main.py verify-data` | Validates data integrity, bar alignment, checks file SHA256 hashes, and freezes split partitions. |
| **Launch Console & Supervisor** | `python main.py up` | Starts the web console and engine supervisor daemon concurrently. |
| **Launch Console Only** | `python main.py dashboard` | Launches the FastAPI web console on `127.0.0.1:8000`. |
| **Terminal TUI Console** | `python main.py tui` | Displays live multi-panel Rich terminal console. |
| **Holdout Evaluation** | `python main.py holdout -s <strat_id>` | Executes audited single-use holdout evaluation. |
| **Forward Paper Trader** | `python main.py forward -s <strat_id>` | Runs paper trading session on incoming closed bars. |
| **Verify LIVE-READY Gate** | `python main.py live-ready -s <strat_id>` | Verifies 60-day and 50-trade non-negotiable criteria. |
| **Refreeze Split Data** | `python main.py refreeze -t 1h --confirm "REFREEZE 1h"` | Refreezes data hashes and marks past backtests. |
| **Run Regression Suite** | `python -m pytest tests/regression/` | Executes all 214+ regression tests. |

---

## 3. Operational Procedures

### 3.1 Starting the Engine
1. Launch the system:
   ```bash
   python main.py up
   ```
2. Navigate to `http://127.0.0.1:8000` in your web browser.
3. Authenticate using the `DASHBOARD_TOKEN` defined in your `.env`.
4. On the **Overview** page, click the green **▶ Start** button.
5. The engine supervisor will spawn the worker subprocess, acquire the lease, and transition state to `RUNNING`.

### 3.2 Pausing and Resuming
- Click **⏸ Pause** on the web console or execute through API.
- The worker will pause at the next stage boundary.
- Click **▶ Resume** to continue execution without losing active cycle context.

### 3.3 Graceful vs Force Stop
- **Graceful Stop (Recommended):** Click **⏹ Graceful Stop**. The engine will complete the current atomic backtest, flush candidate metrics, mark the cycle saved, and terminate cleanly within $30\text{s}$.
- **Force Stop (Emergency Only):** Click **⚡ Force Stop**. Immediately terminates the worker process tree. Any in-flight uncommitted task will be aborted and rolled back.

### 3.4 Out-of-Sample Holdout Run
When a candidate strategy has passed all 19 gates and appears on the leaderboard:
```bash
python main.py holdout --strategy <strategy_id>
```
- **If Passed:** Status transitions to `holdout_passed`. Strategy becomes eligible for forward paper trading.
- **If Failed:** Status transitions to `holdout_failed`. The holdout dataset is burned for this strategy forever.

### 3.5 Forward Paper Trading & Live-Ready Promotion
1. Run paper trading sessions as closed bars arrive:
   ```bash
   python main.py forward --strategy <strategy_id> --feed csv --folder data/live_feed
   ```
2. Monitor rolling forward metrics and decay bands on the **Leaderboard** and **Strategy Detail** console pages.
3. Once the strategy has accumulated $\ge 60$ calendar days and $\ge 50$ forward trades:
   ```bash
   python main.py live-ready --strategy <strategy_id>
   ```
   If all hard floors and decay requirements are satisfied, the strategy is officially promoted to `LIVE_READY`.

---

## 4. Disaster Recovery & Troubleshooting

### 4.1 "Worker Stale" or Missing Heartbeats
If the worker process hangs or crashes:
- The supervisor watchdog automatically detects when `heartbeat_age > 30s`.
- The supervisor terminates the unresponsive worker process tree and spawns a clean worker subprocess.
- The new worker acquires the lease and resumes the active cycle from the `cycles` collection.

### 4.2 Split-Brain / Stale Lease Lock
If a process died abruptly without releasing the lease lock:
- The `engine_lease` collection has a strict $15\text{s}$ TTL.
- MongoDB will automatically expire and release the lease within $15\text{s}$.
- Alternatively, clear the stale lock manually:
  ```bash
  python -c "from storage.mongo import get_db; get_db('quantforge')['engine_lease'].delete_many({})"
  ```

### 4.3 LLM Quota Limits & Key Failover
- If `key_1` hits a rate limit ($429$), the engine automatically switches to `key_2` and enters exponential cooldown for `key_1`.
- If both keys hit daily token caps, the engine automatically enters `BUDGET_PAUSED` and sleeps until $00:00$ UTC.
- Adjust runtime token budgets in the web console under the **Settings** tab.
