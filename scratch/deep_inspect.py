"""Deep inspection of all runs: failure categories, error messages, and strategy details."""
import sys
sys.path.insert(0, '.')
from core.config import load_env
load_env()
from storage.mongo import get_db
from core.config import load_config, mongo_db_name

cfg = load_config()
db = get_db(mongo_db_name(cfg))

print("=" * 80)
print("COMPLETE RUN ANALYSIS")
print("=" * 80)

# 1. Aggregate by rejected_at
print("\n=== RUNS BY REJECTED_AT ===")
for doc in db['runs'].aggregate([
    {'$group': {'_id': '$rejected_at', 'count': {'$sum': 1}}},
    {'$sort': {'count': -1}},
]):
    print(f"  {doc['_id']}: {doc['count']}")

# 2. Aggregate by status
print("\n=== RUNS BY STATUS ===")
for doc in db['runs'].aggregate([
    {'$group': {'_id': '$status', 'count': {'$sum': 1}}},
    {'$sort': {'count': -1}},
]):
    print(f"  {doc['_id']}: {doc['count']}")

# 3. Detailed error messages for each failure category
print("\n=== DETERMINISM FAILURES (last 5) ===")
for r in db['runs'].find({'rejected_at': 'determinism'}).sort('started_at', -1).limit(5):
    print(f"\n  Strategy: {r.get('strategy_name')}")
    gates = r.get('train_gates', {}).get('gates', [])
    for g in gates:
        if g.get('name') == 'determinism' and not g.get('passed'):
            print(f"    Error: {g.get('details', {}).get('detail', 'N/A')}")

print("\n=== MINIMUM_SAMPLE FAILURES (last 5) ===")
for r in db['runs'].find({'rejected_at': 'minimum_sample'}).sort('started_at', -1).limit(5):
    print(f"\n  Strategy: {r.get('strategy_name')}")
    gates = r.get('train_gates', {}).get('gates', [])
    for g in gates:
        if g.get('name') == 'minimum_sample' and not g.get('passed'):
            print(f"    Details: {g.get('details', {})}")

print("\n=== SMOKE_RUN FAILURES (last 5) ===")
for r in db['runs'].find({'rejected_at': 'smoke_run'}).sort('started_at', -1).limit(5):
    print(f"\n  Strategy: {r.get('strategy_name')}")
    gates = r.get('train_gates', {}).get('gates', [])
    for g in gates:
        if g.get('name') == 'smoke_run' and not g.get('passed'):
            print(f"    Error: {g.get('details', {}).get('error', 'N/A')[:200]}")

print("\n=== SANDBOX FAILURES (last 5) ===")
for r in db['runs'].find({'rejected_at': 'sandbox'}).sort('started_at', -1).limit(5):
    print(f"\n  Strategy: {r.get('strategy_name')}")
    print(f"    Error: {str(r.get('error_message', 'N/A'))[:200]}")

print("\n=== BASIC_QUALITY FAILURES (last 5) ===")
for r in db['runs'].find({'rejected_at': 'basic_quality'}).sort('started_at', -1).limit(5):
    print(f"\n  Strategy: {r.get('strategy_name')}")
    gates = r.get('train_gates', {}).get('gates', [])
    for g in gates:
        if g.get('name') == 'basic_quality' and not g.get('passed'):
            print(f"    Failures: {g.get('details', {}).get('failures', [])}")

print("\n=== TRUNCATION FAILURES (last 5) ===")
for r in db['runs'].find({'rejected_at': 'truncation'}).sort('started_at', -1).limit(5):
    print(f"\n  Strategy: {r.get('strategy_name')}")
    gates = r.get('train_gates', {}).get('gates', [])
    for g in gates:
        if g.get('name') == 'truncation' and not g.get('passed'):
            print(f"    Detail: {g.get('details', {}).get('detail', 'N/A')[:200]}")

print("\n=== SIGNAL_VALIDITY FAILURES (last 5) ===")
for r in db['runs'].find({'rejected_at': 'signal_validity'}).sort('started_at', -1).limit(5):
    print(f"\n  Strategy: {r.get('strategy_name')}")
    gates = r.get('train_gates', {}).get('gates', [])
    for g in gates:
        if g.get('name') == 'signal_validity' and not g.get('passed'):
            print(f"    Error: {g.get('details', {})}")

print("\n=== ERROR STATUS DETAILS (last 5) ===")
for r in db['runs'].find({'status': 'error'}).sort('started_at', -1).limit(5):
    print(f"\n  Strategy: {r.get('strategy_name')}")
    print(f"    Error: {str(r.get('error_message', 'N/A'))[:300]}")

# 4. Candidates / survivors
print("\n=== CANDIDATES ===")
for c in db['candidates'].find().sort('robustness_score', -1).limit(5):
    print(f"\n  Strategy: {c.get('strategy_name')}")
    print(f"    Score: {c.get('robustness_score')}")
    print(f"    Status: {c.get('status')}")
    print(f"    Sharpe: {c.get('train_sharpe')}")
    print(f"    PF: {c.get('train_pf')}")
    print(f"    Trades: {c.get('train_trades')}")

print("\n=== LEADERBOARD ===")
for l in db['leaderboard'].find().sort('robustness_score', -1).limit(5):
    print(f"\n  Strategy: {l.get('strategy_name')}")
    print(f"    Score: {l.get('robustness_score')}")

# 5. Total counts
print("\n=== TOTAL COUNTS ===")
print(f"  runs: {db['runs'].count_documents({})}")
print(f"  candidates: {db['candidates'].count_documents({})}")
print(f"  leaderboard: {db['leaderboard'].count_documents({})}")
