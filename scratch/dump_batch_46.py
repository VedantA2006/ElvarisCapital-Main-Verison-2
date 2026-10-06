import sys, json
sys.path.insert(0, '.')
from datetime import datetime, timedelta, timezone
from core.config import load_env, load_config, mongo_db_name
load_env()
from storage.mongo import get_db

db = get_db(mongo_db_name(load_config()))
start_utc = datetime(2026, 10, 5, 19, 30, tzinfo=timezone.utc)

runs = list(db['runs'].find({
    'started_at': {'$gte': start_utc},
    'is_rerun': {'$ne': True}
}).sort('started_at', 1))

summary_data = []
for r in runs:
    name = r.get('strategy_name', 'Unknown')
    tf = r.get('timeframe', '?')
    status = r.get('status', '?')
    rej = r.get('rejected_at', '-')
    wall = r.get('wall_seconds', 0)
    start_time = r.get('started_at')
    fin_time = r.get('finished_at')
    
    # Extract metrics from train_result
    tr = r.get('train_result') or {}
    metrics = tr.get('metrics', {})
    
    # Extract failure details
    tg = r.get('train_gates', {})
    gates_list = tg.get('gates', [])
    failed_reasons = []
    failed_gate_name = ""
    for g in gates_list:
        if not g.get('passed'):
            failed_gate_name = g.get('name', '')
            details = g.get('details', {})
            failed_reasons = details.get('failures', [])
            if not failed_reasons:
                failed_reasons = [details.get('reason') or details.get('error') or failed_gate_name]
            break
            
    summary_data.append({
        'name': name,
        'timeframe': tf,
        'status': status,
        'rejected_at': rej,
        'wall_seconds': wall,
        'started_at': start_time.strftime('%Y-%m-%d %H:%M:%S UTC') if isinstance(start_time, datetime) else str(start_time),
        'finished_at': fin_time.strftime('%Y-%m-%d %H:%M:%S UTC') if isinstance(fin_time, datetime) else str(fin_time),
        'trades': metrics.get('total_trades', 0),
        'win_rate': metrics.get('win_rate', 0.0),
        'net_pnl': metrics.get('total_net_pnl', 0.0),
        'gross_pnl': metrics.get('total_gross_pnl', 0.0),
        'costs': metrics.get('total_costs', 0.0),
        'spread_cost': metrics.get('spread_cost_total', 0.0),
        'slippage_cost': metrics.get('slippage_cost_total', 0.0),
        'commission_cost': metrics.get('commission_total', 0.0),
        'profit_factor': metrics.get('profit_factor', 0.0),
        'sharpe': metrics.get('sharpe', 0.0),
        'sortino': metrics.get('sortino', 0.0),
        'max_drawdown': metrics.get('max_drawdown', 0.0),
        'payoff_ratio': metrics.get('payoff_ratio', 0.0),
        'avg_win': metrics.get('avg_win', 0.0),
        'avg_loss': metrics.get('avg_loss', 0.0),
        'failed_gate': failed_gate_name or rej,
        'failed_reasons': failed_reasons,
        'concept': r.get('idea', {}).get('concept_family', r.get('idea', {}).get('strategy_family', 'unknown')),
        'source_code_len': len(r.get('source_code', '')),
    })

with open('scratch/batch_46_metrics.json', 'w') as f:
    json.dump(summary_data, f, indent=2, default=str)

print(f'Successfully dumped {len(summary_data)} strategies to scratch/batch_46_metrics.json')
