"""
run_autonomous_loop.py – Autonomous LLM Strategy Discovery & Validation Loop.

Runs the live LLM strategy generation loop with:
  1. Primary & Backup API key failover.
  2. Automatic code fix loop (_eval_with_code_fix) on runtime errors.
  3. 19-stage quantitative validation pipeline.
  4. Automatic leaderboard registration for passing candidates.
"""
import sys
import os
sys.path.insert(0, '.')
import time
import logging

# Ensure stdout and stderr flush immediately
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    force=True
)
_log = logging.getLogger("autonomous_loop")

from core.config import load_env, load_config, mongo_db_name
load_env()
from storage.mongo import get_db
from llm.client import KeyManager

def main():
    cfg = load_config()
    db = get_db(mongo_db_name(cfg))
    
    print("=" * 70, flush=True)
    print("   QUANTFORGE AUTONOMOUS LLM DISCOVERY LOOP", flush=True)
    print("=" * 70, flush=True)
    
    # Key manager status
    km = KeyManager.get_instance()
    k1 = os.getenv("LLM_API_KEY_1") or os.getenv("LLM_API_KEY", "")
    k2 = os.getenv("LLM_API_KEY_2", "")
    print(f"Primary API Key:   {k1[:10]}...{k1[-4:] if len(k1) > 14 else ''}", flush=True)
    print(f"Backup API Key:    {k2[:10]}...{k2[-4:] if len(k2) > 14 else ''}", flush=True)
    print(f"Database:          {mongo_db_name(cfg)}", flush=True)
    
    # Check current leaderboard
    current_candidates = list(db['leaderboard'].find().sort('robustness_score', -1))
    print(f"Current Leaderboard Candidates: {len(current_candidates)}", flush=True)
    for c in current_candidates[:5]:
        print(f"  - {c.get('strategy_name')}: Score={c.get('robustness_score'):.1f} (TF={c.get('timeframe')})", flush=True)
    print("=" * 70, flush=True)
    
    # Initialize Orchestrator
    from llm.orchestrator import Orchestrator
    orch = Orchestrator(cfg)
    
    # Run loop for 20 trials
    print("\nStarting discovery loop (20 trials)...", flush=True)
    t_start = time.time()
    records = orch.run_loop(max_trials=20)
    elapsed = time.time() - t_start
    
    # Summary
    candidates = [r for r in records if r.status in ('candidate', 'survived')]
    rejected = [r for r in records if r.status == 'rejected']
    errors = [r for r in records if r.status == 'error']
    
    print("\n" + "=" * 70, flush=True)
    print("   AUTONOMOUS DISCOVERY LOOP COMPLETED", flush=True)
    print("=" * 70, flush=True)
    print(f"Total Trials:      {len(records)}", flush=True)
    print(f"Candidates Found:  {len(candidates)}", flush=True)
    print(f"Rejected:          {len(rejected)}", flush=True)
    print(f"Errors:            {len(errors)}", flush=True)
    print(f"Total Runtime:     {elapsed / 60:.1f} minutes", flush=True)
    
    if candidates:
        print("\n=== PROMOTED CANDIDATES ===", flush=True)
        for c in candidates:
            score = c.train_gates.get('robustness_score', 0.0) if c.train_gates else 0.0
            print(f"  * {c.strategy_name} | TF={c.timeframe} | Score={score:.1f}", flush=True)
    
    # Updated leaderboard
    final_candidates = list(db['leaderboard'].find().sort('robustness_score', -1))
    print(f"\nFinal Leaderboard Count: {len(final_candidates)}", flush=True)
    for c in final_candidates:
        print(f"  - {c.get('strategy_name')}: Score={c.get('robustness_score'):.1f} | TF={c.get('timeframe')}", flush=True)

if __name__ == "__main__":
    main()
