import sys
sys.path.insert(0, '.')
from core.config import load_env
load_env()
from storage.mongo import get_db

db = get_db('quantforge')

print('=== TOTAL RUNS BY REJECTED_AT ===')
for doc in db['runs'].aggregate([
    {'$group': {'_id': '$rejected_at', 'count': {'$sum': 1}}},
    {'$sort': {'count': -1}}
]):
    print(f"{doc['_id']}: {doc['count']}")

print('\n=== LAST 15 LOGS IN MONGODB ===')
for log in db['logs'].find().sort('created_at', -1).limit(15):
    ts = str(log.get('created_at'))[:19]
    lvl = str(log.get('level', 'INFO'))
    msg = str(log.get('message', ''))[:100]
    print(f"{ts} | {lvl:<8} | {msg}")

