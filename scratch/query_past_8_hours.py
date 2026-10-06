import sys
sys.path.insert(0, '.')
from datetime import datetime, timedelta, timezone
from core.config import load_env, load_config, mongo_db_name
load_env()
from storage.mongo import get_db

db = get_db(mongo_db_name(load_config()))

now_utc = datetime.now(timezone.utc)
eight_hours_ago = now_utc - timedelta(hours=8)

print(f"Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"Window Start: {eight_hours_ago.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"Local Window: {(eight_hours_ago + timedelta(hours=5, minutes=30)).strftime('%Y-%m-%d %H:%M:%S IST')} -> {(now_utc + timedelta(hours=5, minutes=30)).strftime('%Y-%m-%d %H:%M:%S IST')}")

# Query runs
runs = list(db['runs'].find({
    '$or': [
        {'started_at': {'$gte': eight_hours_ago}},
        {'finished_at': {'$gte': eight_hours_ago}}
    ]
}).sort('started_at', 1))

print(f"\n==================================================")
print(f"TOTAL STRATEGIES GENERATED IN PAST 8 HOURS: {len(runs)}")
print(f"==================================================")

statuses = {}
rejections = {}
tf_counts = {}

for r in runs:
    st = r.get('status', 'unknown')
    rej = r.get('rejected_at', 'none')
    tf = r.get('timeframe', 'unknown')
    
    statuses[st] = statuses.get(st, 0) + 1
    rejections[rej] = rejections.get(rej, 0) + 1
    tf_counts[tf] = tf_counts.get(tf, 0) + 1

print('\nStatus Breakdown:')
for k, v in statuses.items():
    print(f'  - {k}: {v}')

print('\nRejection Reason Breakdown:')
for k, v in sorted(rejections.items(), key=lambda x: x[1], reverse=True):
    print(f'  - {k}: {v}')

print('\nTimeframe Breakdown:')
for k, v in tf_counts.items():
    print(f'  - {k}: {v}')

if runs:
    print('\nAll Strategies Generated in Past 8 Hours (Chronological):')
    for i, r in enumerate(runs, 1):
        name = r.get('strategy_name', 'Unknown')
        st = r.get('status', '?')
        rej = r.get('rejected_at', '-')
        tf = r.get('timeframe', '?')
        start = r.get('started_at')
        start_str = start.strftime('%H:%M:%S') if isinstance(start, datetime) else str(start)[:19]
        wall = r.get('wall_seconds', 0)
        print(f"  {i:2d}. [{start_str}] {name} (TF={tf}) -> {st} (stage: {rej}) [{wall:.1f}s]")

# Check Leaderboard
candidates = list(db['leaderboard'].find().sort('robustness_score', -1))
print(f"\nLeaderboard Candidates: {len(candidates)}")
for c in candidates:
    print(f"  * {c.get('strategy_name')}: Score={c.get('robustness_score')} | TF={c.get('timeframe')}")

# Also check current engine state
state_doc = db['engine_state'].find_one({'_id': 'current_state'}) or {}
print(f"\nCurrent Engine State in DB: {state_doc.get('state')} (reason: {state_doc.get('reason')})")
print(f"Last Heartbeat: {datetime.fromtimestamp(state_doc.get('last_heartbeat', 0), tz=timezone.utc) if state_doc.get('last_heartbeat') else 'None'}")
