import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import load_config, load_env
load_env()
cfg = load_config()

from llm.orchestrator import Orchestrator

print("[TEST] Initializing Orchestrator...")
orch = Orchestrator(cfg=cfg)
print("[TEST] Running 1 trial on 1h timeframe...")
t0 = time.time()
record = orch.run_trial("1h")
dt = time.time() - t0

print("\n" + "=" * 60)
print(f"TRIAL FINISHED in {dt:.1f}s")
print(f"Strategy: {record.strategy_name}")
print(f"Status: {record.status}")
print(f"Rejected At: {record.rejected_at}")
print(f"Error Message: {record.error_message}")
if record.train_result:
    print(f"Train Trades: {record.train_result.get('metrics', {}).get('total_trades')}")
    print(f"Train Sharpe: {record.train_result.get('metrics', {}).get('sharpe')}")
    print(f"Train Profit Factor: {record.train_result.get('metrics', {}).get('profit_factor')}")
print("=" * 60)
