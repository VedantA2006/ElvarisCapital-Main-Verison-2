import sys
sys.path.insert(0, '.')
import time
from core.config import load_env, load_config, mongo_db_name
load_env()
from storage.mongo import get_db

db = get_db(mongo_db_name(load_config()))
db['engine_lease'].delete_many({})
db['engine_control'].update_one(
    {'_id': 'engine_control'},
    {'$set': {'desired_state': 'running', 'requested_by': 'system_reset', 'requested_at': time.time()}},
    upsert=True
)
db['engine_state'].update_one(
    {'_id': 'current_state'},
    {'$set': {'state': 'RUNNING', 'reason': 'Clean engine start', 'last_heartbeat': time.time(), 'updated_at': time.time()}},
    upsert=True
)
print('Database engine state reset cleanly to RUNNING!')
