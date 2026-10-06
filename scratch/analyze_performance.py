import sys
sys.path.insert(0, '.')
from datetime import datetime, timedelta, timezone
from core.config import load_env, load_config, mongo_db_name
load_env()
from storage.mongo import get_db

db = get_db(mongo_db_name(load_config()))
now_utc = datetime.now(timezone.utc)
eight_hours_ago = now_utc - timedelta(hours=8)

runs = list(db['runs'].find({
    '$or': [
        {'started_at': {'$gte': eight_hours_ago}},
        {'finished_at': {'$gte': eight_hours_ago}}
    ]
}).sort('started_at', 1))

print(f"Total runs analyzed: {len(runs)}")
print(f"{'Strategy Name':34s} | {'Rejected At':15s} | {'Trades':6s} | {'Net PnL ($)':11s} | {'PF':5s} | {'Sharpe':6s} | {'WinRate':7s} | {'MaxDD':7s}")
print("-" * 105)

profitable = []
unprofitable = []
zero_trades = []

for r in runs:
    name = r.get('strategy_name', 'Unknown')
    rej = r.get('rejected_at', '-')
    train_res = r.get('train_result') or {}
    metrics = train_res.get('metrics', {})
    
    trades = metrics.get('total_trades', 0)
    net_profit = metrics.get('net_profit', 0.0)
    sharpe = metrics.get('sharpe_ratio', 0.0)
    win_rate = metrics.get('win_rate', 0.0)
    pf = metrics.get('profit_factor', 0.0)
    mdd = metrics.get('max_drawdown', 0.0)
    
    if trades == 0:
        zero_trades.append((name, rej))
    elif net_profit > 0 or pf > 1.0:
        profitable.append((name, rej, trades, net_profit, pf, sharpe, win_rate, mdd))
    else:
        unprofitable.append((name, rej, trades, net_profit, pf, sharpe, win_rate, mdd))

    print(f"{name:34s} | {rej:15s} | {trades:6d} | ${net_profit:10.2f} | {pf:5.2f} | {sharpe:6.2f} | {win_rate*100:6.1f}% | {mdd*100:6.1f}%")

print("\n" + "=" * 60)
print(f"SUMMARY BREAKDOWN:")
print(f"  * Actually Profitable (Net PnL > $0 or PF > 1.0): {len(profitable)} / {len(runs)}")
print(f"  * Unprofitable (Negative PnL / PF < 1.0):          {len(unprofitable)} / {len(runs)}")
print(f"  * Zero Trades Taken:                               {len(zero_trades)} / {len(runs)}")
print("=" * 60)

if profitable:
    print("\nPROFITABLE STRATEGIES (NET PROFIT > $0 OR PF > 1.0):")
    for p in profitable:
        r = db['runs'].find_one({'strategy_name': p[0]})
        tg = r.get('train_gates', {}) if r else {}
        fail_reasons = []
        for g in tg.get('gates', []):
            if not g.get('passed'):
                fail_reasons = g.get('details', {}).get('failures', [g.get('name')])
                metrics = g.get('details', {}).get('metrics', {})
                net_usd = metrics.get('total_net_pnl', p[3])
                gross_usd = metrics.get('total_gross_pnl', 0)
                costs = metrics.get('total_costs', 0)
                sharpe_val = metrics.get('sharpe', p[5])
                print(f"  * {p[0]}:")
                print(f"      Net PnL:     +${net_usd:,.2f} (Gross: +${gross_usd:,.2f}, Friction Costs: -${costs:,.2f})")
                print(f"      PF:          {p[4]:.2f} (Threshold requires >= 1.25)")
                print(f"      Sharpe:      {sharpe_val:.2f} (Threshold requires >= 0.80)")
                print(f"      Trades:      {p[2]} (Min trades gate requires >= 80/100)")
                print(f"      Win Rate:    {p[6]*100:.1f}%, Max Drawdown: {p[7]*100:.1f}%")
                print(f"      Gate Failed: {p[1]} -> Reasons: {fail_reasons}")


# Inspect the gate thresholds from config
cfg = load_config()
print("\nGATE THRESHOLDS IN CONFIG:")
print("  Minimum Trades required:", cfg.get('validation', {}).get('min_trades', cfg.get('gates', {}).get('min_trades')))
print("  Basic Quality Sharpe min:", cfg.get('validation', {}).get('min_sharpe', cfg.get('gates', {}).get('min_sharpe')))
print("  Basic Quality PF min:", cfg.get('validation', {}).get('min_profit_factor', cfg.get('gates', {}).get('min_profit_factor')))
print("  Max Drawdown allowed:", cfg.get('validation', {}).get('max_drawdown', cfg.get('gates', {}).get('max_drawdown')))
print("  Spread & Slippage model:", cfg.get('backtester', {}).get('spread_usd'), cfg.get('backtester', {}).get('slippage_usd'))
