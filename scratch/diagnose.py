"""Diagnostic script to analyze why no strategies hit the leaderboard."""
import sys
sys.path.insert(0, '.')
from core.config import load_env
load_env()
from storage.mongo import get_db
from collections import Counter

db = get_db('quantforge')
runs = list(db['runs'].find())

print(f"Total runs: {len(runs)}")
print()

# Status breakdown
status_counts = Counter(r.get('status') for r in runs)
print("=== STATUS BREAKDOWN ===")
for s, c in status_counts.most_common():
    print(f"  {s}: {c}")

print()

# Rejection gate breakdown
reject_counts = Counter(r.get('rejected_at') for r in runs if r.get('status') in ('rejected', 'error'))
print("=== REJECTION GATE BREAKDOWN ===")
for g, c in reject_counts.most_common():
    print(f"  {g}: {c}")

print()

# Error messages
error_runs = [r for r in runs if r.get('status') == 'error']
print(f"=== ERROR MESSAGES ({len(error_runs)} errors) ===")
for r in error_runs[:15]:
    name = r.get('strategy_name', '?')
    err = str(r.get('error_message', ''))[:150]
    print(f"  {name}: {err}")

print()

# Check strategies that got furthest
for r in runs:
    gates = r.get('train_gates', {})
    if isinstance(gates, dict):
        gate_list = gates.get('gates', [])
        passed = [g for g in gate_list if g.get('passed')]
        if len(passed) >= 8:
            print(f"FURTHEST: {r.get('strategy_name')} passed {len(passed)} gates, stopped at: {r.get('rejected_at')}")

print()

# Specific gate details for minimum_sample rejections
min_sample_runs = [r for r in runs if r.get('rejected_at') == 'minimum_sample']
print(f"=== MINIMUM SAMPLE REJECTIONS ({len(min_sample_runs)}) ===")
for r in min_sample_runs[:5]:
    gates = r.get('train_gates', {})
    if isinstance(gates, dict):
        gate_list = gates.get('gates', [])
        for g in gate_list:
            if g.get('name') == 'minimum_sample':
                print(f"  {r.get('strategy_name')}: {g.get('details', {})}")

# basic_quality rejections
bq_runs = [r for r in runs if r.get('rejected_at') == 'basic_quality']
print(f"\n=== BASIC QUALITY REJECTIONS ({len(bq_runs)}) ===")
for r in bq_runs[:5]:
    gates = r.get('train_gates', {})
    if isinstance(gates, dict):
        gate_list = gates.get('gates', [])
        for g in gate_list:
            if g.get('name') == 'basic_quality':
                print(f"  {r.get('strategy_name')}: {g.get('details', {}).get('failures', [])}")
