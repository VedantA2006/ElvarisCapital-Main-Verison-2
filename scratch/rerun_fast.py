"""
rerun_fast.py – Quick triage + focused rerun of previously failed strategies.

Phase 1 (fast, ~1 min): Smoke test all 94 strategies against 200 bars.
Phase 2 (focused): Run full pipeline only on strategies that pass smoke test.
"""
import sys
sys.path.insert(0, '.')
import time
import hashlib
from datetime import datetime, timezone

from core.config import load_env, load_config, mongo_db_name
load_env()

from storage.mongo import get_db
from core.backtester import generate_signal_tape
from core.splits import DataStore
from core.lookahead_guard import static_scan
from validation.gates import run_pipeline

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
_log = logging.getLogger("rerun")


def main():
    cfg = load_config()
    db = get_db(mongo_db_name(cfg))
    store = DataStore(cfg)

    failed_runs = list(db['runs'].find({
        'rejected_at': {'$in': ['determinism', 'minimum_sample']},
        'source_code': {'$exists': True, '$ne': ''},
    }).sort('started_at', -1))

    # Deduplicate
    seen_hashes = set()
    unique_runs = []
    for r in failed_runs:
        sh = r.get('source_hash', '')
        if sh and sh not in seen_hashes:
            seen_hashes.add(sh)
            unique_runs.append(r)

    print(f"Found {len(unique_runs)} unique failed strategies")

    # Phase 1: Fast triage
    print("\n=== PHASE 1: SMOKE TEST TRIAGE ===")
    passable = []
    for i, run_doc in enumerate(unique_runs):
        name = run_doc.get('strategy_name', 'Unknown')
        code = run_doc.get('source_code', '')
        tf = run_doc.get('timeframe', '1h')

        if not code or 'class Strategy' not in code:
            continue

        # Static scan
        scan = static_scan(code)
        if not scan.passed:
            print(f"  [{i+1}] {name}: static scan failed")
            continue

        # Smoke test on 200 bars
        train_df = store.get_data(tf, "train")
        smoke_df = train_df.iloc[:200].copy()
        smoke_df.attrs = train_df.attrs.copy()

        try:
            tape, errs, first_err = generate_signal_tape(code, smoke_df, cfg)
        except Exception as e:
            errs = 1
            first_err = str(e)

        if errs > 0:
            err_brief = str(first_err).split('\n')[-1][:80] if first_err else 'unknown'
            print(f"  [{i+1}] {name}: RUNTIME ERROR - {err_brief}")
        else:
            old_rej = run_doc.get('rejected_at', '?')
            print(f"  [{i+1}] {name}: [PASS] (was: {old_rej})")
            passable.append(run_doc)

    print(f"\n=== PHASE 1 RESULT: {len(passable)}/{len(unique_runs)} pass smoke test ===\n")

    if not passable:
        print("No strategies pass smoke test. Starting normal loop instead...")
        return

    # Phase 2: Run full pipeline on passable strategies
    print(f"=== PHASE 2: FULL PIPELINE ({len(passable)} strategies) ===")
    promoted = 0
    for i, run_doc in enumerate(passable):
        name = run_doc.get('strategy_name', 'Unknown')
        code = run_doc.get('source_code', '')
        tf = run_doc.get('timeframe', '1h')
        idea = run_doc.get('idea', {})
        old_rej = run_doc.get('rejected_at', '?')

        print(f"\n[{i+1}/{len(passable)}] {name} (was: {old_rej}, TF={tf})")

        train_df = store.get_data(tf, "train")
        val_df = store.get_data(tf, "validation")

        params = None
        if isinstance(idea, dict):
            spec = idea.get('spec', idea)
            params = spec.get('parameters') or idea.get('parameters')

        try:
            t0 = time.time()
            res = run_pipeline(
                source=code,
                df_train=train_df,
                df_val=val_df,
                cfg=cfg,
                params=params,
            )
            elapsed = time.time() - t0

            if res.all_passed:
                promoted += 1
                print(f"  [PROMOTED] Score={res.robustness_score:.1f} [{elapsed:.1f}s]")

                # Register as candidate in DB
                from llm.orchestrator import TrialRecord, update_leaderboard
                record = TrialRecord(
                    trial_id=f"rerun-{hashlib.sha256(code.encode()).hexdigest()[:8]}",
                    strategy_name=name,
                    timeframe=tf,
                    idea=idea,
                    source_code=code,
                    source_hash=hashlib.sha256(code.encode()).hexdigest()[:16],
                    status="candidate",
                    train_result={"metrics": res.train_result.metrics} if res.train_result else None,
                    val_result={"metrics": res.val_result.metrics} if res.val_result else None,
                    train_gates=res.to_doc(),
                )
                update_leaderboard(record, cfg)
            else:
                print(f"  [REJECTED] at: {res.stopped_at} [{elapsed:.1f}s]")

            # Log rerun result
            rerun_doc = {
                'strategy_name': name,
                'timeframe': tf,
                'source_code': code,
                'source_hash': hashlib.sha256(code.encode()).hexdigest()[:16],
                'idea': idea,
                'status': res.status,
                'rejected_at': res.stopped_at,
                'train_gates': res.to_doc(),
                'started_at': datetime.now(timezone.utc),
                'is_rerun': True,
                'original_rejected_at': old_rej,
            }
            db['runs'].insert_one(rerun_doc)

        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\n{'='*60}")
    print(f"RERUN COMPLETE: {promoted}/{len(passable)} promoted to candidate")
    print(f"{'='*60}")

    # Now start the normal discovery loop with fixes
    print("\n=== STARTING NORMAL DISCOVERY LOOP ===")
    from llm.orchestrator import Orchestrator
    orch = Orchestrator(cfg)
    records = orch.run_loop(max_trials=20)

    survived = sum(1 for r in records if r.status in ('candidate', 'survived'))
    rejected = sum(1 for r in records if r.status == 'rejected')
    print(f"\nLoop complete: {survived} candidates, {rejected} rejected out of {len(records)} trials")


if __name__ == "__main__":
    main()
