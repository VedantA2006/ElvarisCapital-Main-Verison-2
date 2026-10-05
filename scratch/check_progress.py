import sys
sys.path.insert(0, '.')
from core.config import load_env, load_config, mongo_db_name
load_env()
from storage.mongo import get_db

db = get_db(mongo_db_name(load_config()))
rerun_count = db['runs'].count_documents({'is_rerun': True})
recent = list(db['runs'].find({'is_rerun': True}).sort('started_at', -1).limit(10))
print(f"Total reruns recorded so far: {rerun_count}")
for r in recent:
    t = r.get('started_at')
    name = r.get('strategy_name')
    status = r.get('status')
    rej = r.get('rejected_at')
    print(f"  [{t}] {name}: {status} (rejected_at: {rej})")

# Also check candidates
candidates = list(db['leaderboard'].find())
print(f"\nTotal candidates in leaderboard: {len(candidates)}")
for c in candidates:
    print(f"  {c.get('strategy_name')}: score={c.get('robustness_score')}")

failed_runs = list(db['runs'].find({
    'rejected_at': {'$in': ['determinism', 'minimum_sample']},
    'source_code': {'$exists': True, '$ne': ''},
    'is_rerun': {'$ne': True}
}))
seen = set()
unique_failed = []
for r in failed_runs:
    sh = r.get('source_hash', '')
    if sh and sh not in seen:
        seen.add(sh)
        unique_failed.append(r)
print(f"\nTotal unique failed strategies in original pool: {len(unique_failed)}")
print(f"Processed so far: {rerun_count} / {len(unique_failed)}")

