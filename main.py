"""
QuantForge CLI entry point.

    python main.py verify-data      # Phase 1: validate all data files, freeze splits, save reports
    python main.py run              # Phase 6+
    python main.py dashboard        # Phase 9
    python main.py forward          # Phase 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `core`, `storage`, ... importable when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import config_hash, load_config, load_env  # noqa: E402


def cmd_verify_data(args: argparse.Namespace) -> int:
    from core.data_loader import (
        add_session_labels, cross_timeframe_consistency,
        load_and_validate, session_map_utc,
    )
    from core.splits import DataStore
    from storage import mongo
    from storage.logger import get_logger

    cfg = load_config(args.config)
    mongo.ensure_indexes()
    log = get_logger(jsonl_dir=str(Path(__file__).resolve().parent / cfg["logging"]["jsonl_dir"]))
    log.info(f"verify-data start (config_hash={config_hash(cfg)[:12]})", stage="verify_data")

    # Validate ALL configured files (incl. disabled 5m/15m) without halting on the first
    halt_cfg = json.loads(json.dumps(cfg))
    halt_cfg["validation"]["critical_errors_halt"] = False
    frames, any_critical = {}, False
    for tf in cfg["data"]["all_timeframes"]:
        df, fhash, report = load_and_validate(tf, halt_cfg)
        report["enabled"] = tf in cfg["data"]["enabled_timeframes"]
        mongo.col_data_reports().insert_one(dict(report))
        frames[tf] = df
        status = "OK" if not report["critical_errors"] else "CRITICAL"
        any_critical |= bool(report["critical_errors"])
        print(f"\n=== {tf} [{status}] {'(enabled)' if report['enabled'] else '(disabled)'} ===")
        print(f"  file: {report['file_name']}  sha256: {fhash[:16]}...")
        print(f"  rows: {report['total_rows']}  range: {report.get('date_range_start')} -> {report.get('date_range_end')}")
        print(f"  gaps: {report['info'].get('gaps')}")
        for e in report["critical_errors"]:
            print(f"  CRITICAL: {e}")
        for w in report["warnings"]:
            brief = {k: v for k, v in w.items() if k not in ("examples", "largest")}
            print(f"  WARNING: {brief}")

    # Cross-timeframe consistency (informational)
    print("\n=== Cross-timeframe consistency (native vs aggregated) ===")
    for low, high in (("5m", "15m"), ("15m", "1h"), ("1h", "4h")):
        if low in frames and high in frames:
            res = cross_timeframe_consistency(frames[low], frames[high], high)
            mongo.col_data_reports().insert_one({"type": "cross_tf", "low_tf": low, **res,
                                                 "created_at": mongo.utcnow()})
            print(f"  {low}->{high}: {res}")

    # Session map
    smap = session_map_utc(cfg)
    mongo.col_session_map().replace_one({"_id": "current"}, {"_id": "current", **smap}, upsert=True)
    print("\n=== Session map (UTC) ===")
    for season in ("summer", "winter"):
        print(f"  {season}: {smap[season]}")
    sample = add_session_labels(frames["1h"], cfg)["session"].value_counts().to_dict()
    print(f"  1h bars per primary session: {sample}")

    if any_critical:
        log.critical("verify-data found CRITICAL errors - stopping. Fix the data before any run.",
                     stage="verify_data")
        return 2

    # Freeze / load split boundaries and show split sizes for enabled TFs
    store = DataStore(cfg)
    b = store.boundaries
    print("\n=== Splits (frozen in Mongo: split_boundaries) ===")
    print(f"  train      : {b.train_start} -> {b.train_end}")
    print(f"  embargo    : {b.embargo_days} days")
    print(f"  validation : {b.validation_start} -> {b.validation_end}")
    print(f"  holdout    : {b.holdout_start} -> {b.holdout_end}  (contents never printed)")
    for tf in cfg["data"]["enabled_timeframes"]:
        info = store.split_info(tf)
        print(f"  {tf}: train={info['train']['bars']} val={info['validation']['bars']} "
              f"holdout={info['holdout']['bars']} bars, post_holdout={info['post_holdout_bars']}")
    log.info("verify-data finished OK", stage="verify_data")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the autonomous strategy discovery loop."""
    from llm.orchestrator import Orchestrator, update_leaderboard

    cfg = load_config(args.config)
    orch = Orchestrator(cfg)

    timeframes = args.timeframes.split(",") if args.timeframes else None
    records = orch.run_loop(
        max_trials=args.trials,
        timeframes=timeframes,
        stop_on_survivor=args.stop_on_survivor,
    )

    # Summary
    survived = [r for r in records if r.status == "survived"]
    rejected = [r for r in records if r.status == "rejected"]
    errors = [r for r in records if r.status == "error"]

    print(f"\n{'='*60}")
    print(f"DISCOVERY COMPLETE: {len(records)} trials")
    print(f"  Survived:  {len(survived)}")
    print(f"  Rejected:  {len(rejected)}")
    print(f"  Errors:    {len(errors)}")
    print(f"{'='*60}")

    for r in survived:
        entry = update_leaderboard(r, cfg)
        print(f"\n  [LEADERBOARD] {r.strategy_name}")
        print(f"    Robustness: {entry.get('robustness_score', 0):.4f}")
        print(f"    Train Sharpe: {entry.get('train_sharpe', 0):.3f}")
        print(f"    Val Sharpe: {entry.get('val_sharpe', 0):.3f}")

    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    """Show the current leaderboard."""
    from storage.mongo import get_db
    from core.config import mongo_db_name

    cfg = load_config(args.config)
    db = get_db(mongo_db_name(cfg))
    entries = list(db["leaderboard"].find().sort("robustness_score", -1).limit(args.top))

    if not entries:
        print("Leaderboard is empty. Run `python main.py run` first.")
        return 0

    print(f"\n{'='*80}")
    print(f"{'Rank':<5} {'Name':<30} {'TF':<5} {'Score':<8} {'Train S':<9} {'Val S':<9} {'Trades':<8}")
    print(f"{'='*80}")
    for i, e in enumerate(entries, 1):
        print(f"{i:<5} {e.get('strategy_name', '?'):<30} "
              f"{e.get('timeframe', '?'):<5} "
              f"{e.get('robustness_score', 0):<8.4f} "
              f"{e.get('train_sharpe', 0):<9.3f} "
              f"{e.get('val_sharpe', 0):<9.3f} "
              f"{e.get('train_trades', 0):<8}")
    print(f"{'='*80}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_env()
    p = argparse.ArgumentParser(prog="quantforge")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify-data").set_defaults(fn=cmd_verify_data)

    run_p = sub.add_parser("run", help="Run autonomous strategy discovery")
    run_p.add_argument("--trials", type=int, default=10, help="Max trials to run")
    run_p.add_argument("--timeframes", type=str, default=None,
                       help="Comma-separated timeframes (default: enabled_timeframes)")
    run_p.add_argument("--stop-on-survivor", action="store_true",
                       help="Stop after first surviving strategy")
    run_p.set_defaults(fn=cmd_run)

    lb_p = sub.add_parser("leaderboard", help="Show leaderboard")
    lb_p.add_argument("--top", type=int, default=20, help="Number of entries to show")
    lb_p.set_defaults(fn=cmd_leaderboard)

    def cmd_dashboard(args):
        import uvicorn
        from dashboard.fastapi_app import create_app
        cfg = load_config(args.config)
        dash_cfg = cfg.get("dashboard", {})
        host = args.host or dash_cfg.get("host", "127.0.0.1")
        port = args.port or dash_cfg.get("port", 8000)
        app = create_app(cfg=cfg)
        print(f"\n[QuantForge] Launching research console at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
        return 0

    dash_p = sub.add_parser("dashboard", help="Launch web dashboard (FastAPI)")
    dash_p.add_argument("--host", default=None, help="Host address (default 127.0.0.1)")
    dash_p.add_argument("--port", type=int, default=None, help="Port (default 8000)")
    dash_p.set_defaults(fn=cmd_dashboard)

    def cmd_up(args):
        import threading
        import uvicorn
        from dashboard.fastapi_app import create_app
        from engine.supervisor import EngineSupervisor
        from storage.mongo import get_db
        from core.config import mongo_db_name

        cfg = load_config(args.config)
        db = get_db(mongo_db_name(cfg))
        dash_cfg = cfg.get("dashboard", {})
        host = args.host or dash_cfg.get("host", "127.0.0.1")
        port = args.port or dash_cfg.get("port", 8000)

        supervisor = EngineSupervisor(db=db, cfg=cfg)
        if args.autostart:
            supervisor.set_desired_state("running", requested_by="cli_autostart")

        sup_thread = threading.Thread(target=supervisor.run_supervision_loop, daemon=True)
        sup_thread.start()

        app = create_app(db=db, cfg=cfg)
        print(f"\n[QuantForge] Engine supervisor active. Research console at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
        return 0

    up_p = sub.add_parser("up", help="Launch dashboard and engine supervisor")
    up_p.add_argument("--host", default=None, help="Host address (default 127.0.0.1)")
    up_p.add_argument("--port", type=int, default=None, help="Port (default 8000)")
    up_p.add_argument("--autostart", action="store_true", help="Automatically set desired_state to running")
    up_p.set_defaults(fn=cmd_up)

    def cmd_tui(args):
        from dashboard.tui import run_tui
        cfg = load_config(args.config)
        return run_tui(cfg)

    sub.add_parser("tui", help="Launch terminal dashboard (Rich)").set_defaults(fn=cmd_tui)
    sub.add_parser("forward").set_defaults(fn=lambda a: print("Phase 12: Holdout and forward testing"))

    def cmd_refreeze(args):
        from core.splits import refreeze
        cfg = load_config(args.config)
        try:
            rep = refreeze(args.timeframe, args.confirm, cfg)
            print(f"Refreeze successful for {args.timeframe}:")
            print(f"  Old hash: {rep['old_hash'][:16] if rep['old_hash'] else 'None'}...")
            print(f"  New hash: {rep['new_hash'][:16]}...")
            print(f"  Invalidated holdouts: {rep['invalidated_holdout_accesses']}")
            print(f"  Marked backtests: {rep['marked_backtests']}")
            return 0
        except Exception as exc:
            print(f"Refreeze failed: {exc}", file=sys.stderr)
            return 1

    rf_p = sub.add_parser("refreeze", help="Re-freeze split boundaries and data lock for a timeframe")
    rf_p.add_argument("--timeframe", "-t", required=True, help="Timeframe to refreeze (e.g. 1h)")
    rf_p.add_argument("--confirm", required=True, help="Confirmation phrase ('REFREEZE <timeframe>')")
    rf_p.set_defaults(fn=cmd_refreeze)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

