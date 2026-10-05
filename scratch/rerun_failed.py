"""
rerun_failed.py – Extract previously failed strategies from MongoDB and rerun 
them through the fixed pipeline (runtime fix loop + better error classification).

This script:
1. Extracts all strategies that failed at 'determinism' (runtime errors)
   and 'minimum_sample' from MongoDB
2. Runs each through the new pipeline with the runtime fix loop
3. Logs results back to MongoDB
"""
import sys
sys.path.insert(0, '.')

import time
import hashlib
import traceback
from datetime import datetime, timezone

from core.config import load_env, load_config, mongo_db_name
load_env()

from storage.mongo import get_db
from llm.orchestrator import Orchestrator, TrialRecord
from core.backtester import generate_signal_tape
from validation.gates import run_pipeline
from core.splits import DataStore
from core.lookahead_guard import static_scan

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
_log = logging.getLogger("rerun_failed")


def main():
    cfg = load_config()
    db = get_db(mongo_db_name(cfg))
    store = DataStore(cfg)
    orch = Orchestrator(cfg)

    # Find all strategies that failed at determinism with runtime errors
    # or at minimum_sample with decent code
    failed_runs = list(db['runs'].find({
        'rejected_at': {'$in': ['determinism', 'minimum_sample']},
        'source_code': {'$exists': True, '$ne': ''},
    }).sort('started_at', -1))

    _log.info("Found %d failed strategies to rerun", len(failed_runs))

    # Deduplicate by source_hash
    seen_hashes = set()
    unique_runs = []
    for r in failed_runs:
        sh = r.get('source_hash', '')
        if sh and sh not in seen_hashes:
            seen_hashes.add(sh)
            unique_runs.append(r)
    
    _log.info("After deduplication: %d unique strategies", len(unique_runs))

    rerun_results = {'passed_runtime': 0, 'fixed_and_passed': 0, 
                     'still_failed': 0, 'promoted': 0, 'total': 0}

    for i, run_doc in enumerate(unique_runs):
        name = run_doc.get('strategy_name', 'Unknown')
        code = run_doc.get('source_code', '')
        tf = run_doc.get('timeframe', '1h')
        idea = run_doc.get('idea', {})
        old_rejected_at = run_doc.get('rejected_at', 'unknown')

        if not code or 'class Strategy' not in code:
            _log.info("[%d/%d] Skipping %s: no valid source code", i+1, len(unique_runs), name)
            continue

        rerun_results['total'] += 1
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(unique_runs)}] Rerunning: {name} (was: {old_rejected_at}) TF={tf}")
        print(f"{'='*60}")

        try:
            train_df = store.get_data(tf, "train")
            val_df = store.get_data(tf, "validation")

            # Step 1: Check if static scan still passes
            scan = static_scan(code)
            if not scan.passed:
                _log.info("  Static scan failed, skipping: %s", scan.summary)
                rerun_results['still_failed'] += 1
                continue

            # Step 2: Runtime smoke test on small sample
            smoke_df = train_df.iloc[:200].copy()
            smoke_df.attrs = train_df.attrs.copy()
            tape, errs, first_err = generate_signal_tape(code, smoke_df, cfg)
            
            if errs > 0:
                _log.info("  Runtime error: %s", str(first_err)[:150])
                # Try to fix with LLM
                record = TrialRecord(
                    trial_id=f"rerun-{hashlib.sha256(code.encode()).hexdigest()[:8]}",
                    strategy_name=name,
                    timeframe=tf,
                    idea=idea,
                    source_code=code,
                    source_hash=hashlib.sha256(code.encode()).hexdigest()[:16],
                )
                fixed_code = orch._runtime_fix_loop(code, train_df, record, idea)
                
                # Recheck
                tape2, errs2, err2 = generate_signal_tape(fixed_code, smoke_df, cfg)
                if errs2 > 0:
                    _log.info("  Still failing after fix: %s", str(err2)[:150])
                    rerun_results['still_failed'] += 1
                    continue
                
                code = fixed_code
                rerun_results['fixed_and_passed'] += 1
                _log.info("  Runtime fix succeeded!")
            else:
                rerun_results['passed_runtime'] += 1
                _log.info("  Runtime smoke test passed directly")

            # Step 3: Run full pipeline
            params = None
            if isinstance(idea, dict):
                spec = idea.get('spec', idea)
                params = spec.get('parameters') or idea.get('parameters')

            pipeline_res = run_pipeline(
                source=code,
                df_train=train_df,
                df_val=val_df,
                cfg=cfg,
                params=params,
            )

            status_str = pipeline_res.status
            stopped_at = pipeline_res.stopped_at
            
            if pipeline_res.all_passed:
                rerun_results['promoted'] += 1
                print(f"  ✓ PROMOTED to candidate! Score={pipeline_res.robustness_score:.1f}")
            else:
                print(f"  ✗ Rejected at: {stopped_at} ({status_str})")

            # Log the rerun result
            rerun_doc = {
                'strategy_name': name,
                'timeframe': tf,
                'source_code': code,
                'source_hash': hashlib.sha256(code.encode()).hexdigest()[:16],
                'idea': idea,
                'status': status_str,
                'rejected_at': stopped_at,
                'train_gates': pipeline_res.to_doc(),
                'started_at': datetime.now(timezone.utc),
                'finished_at': datetime.now(timezone.utc),
                'is_rerun': True,
                'original_rejected_at': old_rejected_at,
            }
            db['runs'].insert_one(rerun_doc)

        except Exception as exc:
            _log.error("  Error rerunning %s: %s", name, exc)
            rerun_results['still_failed'] += 1
            continue

        # Brief cooldown between runs
        time.sleep(1)

    print(f"\n{'='*60}")
    print("RERUN SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rerun:        {rerun_results['total']}")
    print(f"  Passed runtime:     {rerun_results['passed_runtime']}")
    print(f"  Fixed and passed:   {rerun_results['fixed_and_passed']}")
    print(f"  Still failed:       {rerun_results['still_failed']}")
    print(f"  PROMOTED:           {rerun_results['promoted']}")


if __name__ == "__main__":
    main()
