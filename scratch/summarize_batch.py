import json

with open('scratch/batch_46_metrics.json') as f:
    data = json.load(f)

print(f"Total strategies dumped: {len(data)}")
pf_gt_1 = [d for d in data if d['profit_factor'] > 1.0]
pf_gt_0 = [d for d in data if 0 < d['profit_factor'] <= 1.0]
zero_trades = [d for d in data if d['trades'] == 0]

print(f"Profit Factor > 1.0 (Profitable): {len(pf_gt_1)}")
print(f"Profit Factor 0 - 1.0 (Losing):     {len(pf_gt_0)}")
print(f"Zero Trades (Over-filtered/Crash): {len(zero_trades)}")

print("\nTop Profitable Strategies:")
for d in pf_gt_1:
    print(f"  - {d['name']} ({d['timeframe']}): PF={d['profit_factor']:.2f}, Net=${d['net_pnl']:,.2f}, Gross=${d['gross_pnl']:,.2f}, Costs=${d['costs']:,.2f}, Trades={d['trades']}, Sharpe={d['sharpe']:.2f}, MaxDD={d['max_drawdown']*100:.1f}%, Rej={d['rejected_at']}")

concepts = {}
rejections = {}
total_trades = sum(d['trades'] for d in data)
total_gross = sum(d['gross_pnl'] for d in data)
total_costs = sum(d['costs'] for d in data)
total_net = sum(d['net_pnl'] for d in data)

for d in data:
    c = d.get('concept', 'unknown')
    concepts[c] = concepts.get(c, 0) + 1
    r = d.get('rejected_at', 'unknown')
    rejections[r] = rejections.get(r, 0) + 1

print("\nRejection Stage Distribution:")
for k, v in sorted(rejections.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k}: {v}")

print("\nConcepts Tested:")
for k, v in sorted(concepts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k}: {v}")

print("\nAggregate Financials across all tested strategies:")
print(f"  Total Trades Simulated: {total_trades:,}")
print(f"  Total Gross PnL:        ${total_gross:,.2f}")
print(f"  Total Friction Costs:   ${total_costs:,.2f}")
print(f"  Total Net PnL:          ${total_net:,.2f}")
