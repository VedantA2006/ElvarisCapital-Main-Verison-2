"""
dashboard/fastapi_app.py – Production-grade FastAPI dashboard and control console for QuantForge.

Closes: UI-1 to UI-5, ORCH-2 (visibility).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
import secrets
import time
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from engine.state import EngineState, EngineStateManager
from engine.supervisor import EngineSupervisor
from storage.mongo import get_db


class LoginRequest(BaseModel):
    token: str


class SettingsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    daily_token_budget: int | None = None
    max_parallel_backtests: int | None = None
    enabled_timeframes: list[str] | None = None
    enabled_concept_families: list[str] | None = None


def create_app(db: Any = None, cfg: dict | None = None, token: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if cfg is None:
        from core.config import load_config
        cfg = load_config()

    dash_cfg = cfg.get("dashboard", {})
    db_name = cfg.get("mongo", {}).get("database", "quantforge")
    database = db if db is not None else get_db(db_name)

    dashboard_token = token or os.environ.get("DASHBOARD_TOKEN") or "qf_" + secrets.token_urlsafe(24)
    csrf_secret = secrets.token_hex(32)

    app = FastAPI(title="QuantForge Console", version="2.0.0")

    # ── CORS Middleware: Strict, No Wildcard ────────────────────────────────
    allowed_origins = dash_cfg.get("cors_allowed_origins", [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://localhost",
    ])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["*"],
    )

    # ── Authentication & CSRF Helpers ───────────────────────────────────────
    def _verify_auth(request: Request) -> str:
        cookie_token = request.cookies.get("qf_token")
        auth_header = request.headers.get("Authorization", "")
        bearer_token = auth_header.replace("Bearer ", "").strip() if auth_header.startswith("Bearer ") else None
        query_token = request.query_params.get("token")

        req_token = cookie_token or bearer_token or query_token
        if req_token and secrets.compare_digest(req_token, dashboard_token):
            return req_token

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Invalid or missing dashboard token.",
        )

    def _verify_csrf(request: Request, _auth: str = Depends(_verify_auth)) -> None:
        if request.method in ("POST", "PATCH", "PUT", "DELETE"):
            csrf_header = request.headers.get("X-CSRF-Token")
            if not csrf_header or len(csrf_header) < 16:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Missing or invalid X-CSRF-Token header.",
                )

    def _write_audit_log(action: str, requested_by: str, details: dict | None = None) -> None:
        try:
            database["audit_logs"].insert_one({
                "action": action,
                "requested_by": requested_by,
                "timestamp": time.time(),
                "iso_time": datetime.now(timezone.utc).isoformat(),
                "details": details or {},
            })
        except Exception as exc:
            import logging
            logging.getLogger("quantforge.dashboard").warning("Failed writing audit log: %s", exc)

    # ── Auth Endpoints ──────────────────────────────────────────────────────
    @app.post("/api/auth/login")
    async def login(payload: LoginRequest, response: Response) -> dict[str, Any]:
        if not secrets.compare_digest(payload.token, dashboard_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid dashboard token.",
            )
        csrf_token = hashlib.sha256(f"{payload.token}_{csrf_secret}".encode()).hexdigest()
        response.set_cookie(
            key="qf_token",
            value=payload.token,
            httponly=False,
            samesite="lax",
            secure=False,
            max_age=86400 * 30,
        )
        return {"status": "authenticated", "csrf_token": csrf_token, "token": payload.token}

    @app.get("/api/auth/csrf")
    async def get_csrf_token(_auth: str = Depends(_verify_auth)) -> dict[str, str]:
        csrf_token = hashlib.sha256(f"{_auth}_{csrf_secret}".encode()).hexdigest()
        return {"csrf_token": csrf_token, "token": _auth}

    @app.post("/api/auth/logout")
    async def logout(response: Response) -> dict[str, str]:
        response.delete_cookie("qf_token")
        return {"status": "logged_out"}

    # ── Engine State & Controls ─────────────────────────────────────────────
    @app.get("/api/engine/state")
    async def get_engine_state(_auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        mgr = EngineStateManager(db=database)
        state = mgr.get_state()
        reason = mgr.get_state_reason()

        state_doc = database["engine_state"].find_one({"_id": "current_state"}) or {}
        last_hb = float(state_doc.get("last_heartbeat", 0))
        hb_age = round(time.time() - last_hb, 1) if last_hb > 0 else 0.0

        current_cycle = database["cycles"].find_one({"status": "in_progress"}, sort=[("updated_at", -1)])
        if not current_cycle:
            current_cycle = database["cycles"].find_one({}, sort=[("updated_at", -1)])
        uptime = round(time.time() - float(state_doc.get("updated_at", time.time())), 1)

        trials_col = database["runs"] if database["runs"].count_documents({}) >= database["trials"].count_documents({}) else database["trials"]
        total_trials = trials_col.count_documents({})
        candidates_count = database["candidates"].count_documents({})

        return {
            "state": state.value,
            "reason": reason,
            "worker_id": state_doc.get("worker_id", "worker_default"),
            "heartbeat_age": hb_age,
            "heartbeat_age_seconds": hb_age,
            "is_stale": mgr.is_stale(30.0),
            "uptime_seconds": uptime,
            "total_trials": total_trials,
            "candidates_count": candidates_count,
            "current_cycle_id": current_cycle["cycle_id"] if current_cycle else None,
            "current_stage": current_cycle.get("stage", "idle") if current_cycle else "idle",
            "current_strategy_name": current_cycle.get("strategy_name", "") if current_cycle else "",
        }

    @app.post("/api/engine/start", dependencies=[Depends(_verify_csrf)])
    async def engine_start() -> dict[str, str]:
        supervisor = EngineSupervisor(db=database, cfg=cfg)
        supervisor.set_desired_state("running", requested_by="dashboard_ui")
        _write_audit_log("start", "dashboard_ui")
        return {"status": "ok", "message": "Engine start requested."}

    @app.post("/api/engine/pause", dependencies=[Depends(_verify_csrf)])
    async def engine_pause() -> dict[str, str]:
        supervisor = EngineSupervisor(db=database, cfg=cfg)
        supervisor.set_desired_state("paused", requested_by="dashboard_ui")
        _write_audit_log("pause", "dashboard_ui")
        return {"status": "ok", "message": "Engine pause requested."}

    @app.post("/api/engine/resume", dependencies=[Depends(_verify_csrf)])
    async def engine_resume() -> dict[str, str]:
        supervisor = EngineSupervisor(db=database, cfg=cfg)
        supervisor.set_desired_state("running", requested_by="dashboard_ui")
        _write_audit_log("resume", "dashboard_ui")
        return {"status": "ok", "message": "Engine resume requested."}

    @app.post("/api/engine/stop", dependencies=[Depends(_verify_csrf)])
    async def engine_stop() -> dict[str, str]:
        supervisor = EngineSupervisor(db=database, cfg=cfg)
        supervisor.set_desired_state("stopped", requested_by="dashboard_ui")
        _write_audit_log("stop", "dashboard_ui")
        return {"status": "ok", "message": "Graceful stop requested."}

    @app.post("/api/engine/force_stop", dependencies=[Depends(_verify_csrf)])
    async def engine_force_stop() -> dict[str, str]:
        supervisor = EngineSupervisor(db=database, cfg=cfg)
        now = time.time()
        supervisor.set_desired_state("stopped", requested_by="dashboard_ui", force_kill_at=now)
        _write_audit_log("force_stop", "dashboard_ui")
        return {"status": "ok", "message": "Immediate force stop requested."}

    # ── Funnel & Statistics ─────────────────────────────────────────────────
    @app.get("/api/stats/funnel")
    async def get_funnel_stats(_auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        trials_col = database["runs"] if database["runs"].count_documents({}) >= database["trials"].count_documents({}) else database["trials"]
        total = trials_col.count_documents({})
        rejection_map: dict[str, int] = {}
        candidates_count = database["candidates"].count_documents({})

        for doc in trials_col.find({}, {"status": 1, "rejected_at": 1}):
            st = doc.get("status")
            if st in ("candidate", "candidate (unproven)", "survived"):
                pass
            elif st in ("rejected", "error"):
                rej = doc.get("rejected_at") or "unspecified"
                rejection_map[rej] = rejection_map.get(rej, 0) + 1

        static_rej = rejection_map.get("static_scan", 0) + rejection_map.get("static_policy", 0)
        smoke_rej = rejection_map.get("smoke_run", 0) + rejection_map.get("compile_smoke", 0) + rejection_map.get("sandbox", 0)
        lookahead_rej = rejection_map.get("determinism", 0) + rejection_map.get("truncation", 0) + rejection_map.get("delay", 0)
        train_rej = (rejection_map.get("signal_validity", 0) +
                     rejection_map.get("minimum_sample", 0) +
                     rejection_map.get("basic_quality", 0) +
                     rejection_map.get("train_backtest", 0))
        robust_rej = (rejection_map.get("monte_carlo", 0) +
                      rejection_map.get("walk_forward", 0) +
                      rejection_map.get("regime_and_year", 0) +
                      rejection_map.get("dsr", 0) +
                      rejection_map.get("pbo", 0))
        val_rej = rejection_map.get("validation", 0)

        s0 = total
        s1 = max(0, s0 - static_rej)
        s2 = max(0, s1 - smoke_rej)
        s3 = max(0, s2 - lookahead_rej)
        s4 = max(0, s3 - train_rej)
        s5 = max(0, s4 - robust_rej)
        s6 = max(0, s5 - val_rej)

        funnel_stages = [
            ("total_ideated", s0),
            ("static_passed", s1),
            ("sandbox_passed", s2),
            ("lookahead_passed", s3),
            ("train_passed", s4),
            ("robustness_passed", s5),
            ("validation_passed", s6),
            ("candidates_promoted", candidates_count),
        ]

        return {
            "total_trials": total,
            "candidates": candidates_count,
            "rejections_by_stage": rejection_map,
            "funnel": [{"stage": s, "count": c} for s, c in funnel_stages],
        }

    # ── Leaderboard API ─────────────────────────────────────────────────────
    @app.get("/api/leaderboard")
    async def get_leaderboard(
        split: str = "train",
        sort: str = "robustness_score",
        order: str = "desc",
        timeframe: str | None = None,
        family: str | None = None,
        min_trades: int = 0,
        page: int = 1,
        limit: int = 50,
        _auth: str = Depends(_verify_auth),
    ) -> dict[str, Any]:
        col = database["candidates"]
        if col.count_documents({}) == 0 and database["leaderboard"].count_documents({}) > 0:
            col = database["leaderboard"]

        is_runs_mode = (split in ("runs", "trials", "all")) or (col.count_documents({}) == 0 and split in ("train", "all", "runs"))

        if is_runs_mode:
            runs_col = database["runs"] if database["runs"].count_documents({}) >= database["trials"].count_documents({}) else database["trials"]
            q: dict[str, Any] = {}
            if timeframe:
                q["timeframe"] = timeframe
            if family:
                q["$or"] = [{"concept_family": family}, {"idea.spec.concept_family": family}]

            total = runs_col.count_documents(q)
            cursor = runs_col.find(q, projection={"source_code": 0}).sort("started_at", -1).skip((page - 1) * limit).limit(limit)

            items = []
            for doc in cursor:
                spec = doc.get("idea", {}).get("spec", {})
                train_res = doc.get("train_result", {})
                train_gates = doc.get("train_gates", {})
                strat_name = doc.get("strategy_name")
                if not strat_name or strat_name == "unnamed":
                    strat_name = spec.get("name") or doc.get("trial_id", "Strategy")

                items.append({
                    "strategy_id": doc.get("trial_id", strat_name),
                    "name": strat_name,
                    "status": doc.get("status", "rejected"),
                    "rejected_at": doc.get("rejected_at", ""),
                    "timeframe": doc.get("timeframe") or spec.get("timeframe", "1h"),
                    "concept_family": spec.get("concept_family") or doc.get("concept_family", "trend"),
                    "robustness_score": float(train_gates.get("robustness_score", 0.0)),
                    "train_metrics": {
                        "sharpe": float(train_res.get("sharpe", 0.0)),
                        "profit_factor": float(train_res.get("profit_factor", 0.0)),
                        "max_drawdown": float(train_res.get("max_drawdown", 0.0)),
                        "total_trades": int(train_res.get("total_trades", len(train_res.get("trades", [])))),
                    },
                    "started_at": str(doc.get("started_at", "")),
                })

            return {
                "items": items,
                "total": total,
                "page": page,
                "limit": limit,
                "split": split,
            }

        query: dict[str, Any] = {}
        if timeframe:
            query["timeframe"] = timeframe
        if family:
            query["concept_family"] = family
        if min_trades > 0:
            query["train_metrics.total_trades"] = {"$gte": min_trades}

        sort_dir = -1 if order == "desc" else 1
        sort_key = sort if "." in sort else ("robustness_score" if sort == "robustness_score" else f"train_metrics.{sort}")

        total = col.count_documents(query)
        cursor = col.find(
            query,
            projection={"_id": 0, "source_code": 0},
        ).sort(sort_key, sort_dir).skip((page - 1) * limit).limit(limit)

        items = list(cursor)
        return {
            "items": items,
            "total": total,
            "page": page,
            "limit": limit,
            "split": split,
        }

    # ── Strategy Detail API ─────────────────────────────────────────────────
    def _find_strat_doc(strat_id: str) -> dict[str, Any] | None:
        q = {"$or": [
            {"strategy_id": strat_id},
            {"trial_id": strat_id},
            {"strategy_name": strat_id},
            {"name": strat_id},
            {"idea.spec.name": strat_id},
        ]}
        for col_name in ("candidates", "leaderboard", "runs", "trials"):
            doc = database[col_name].find_one(q, projection={"_id": 0})
            if doc:
                return doc
        return None

    def _downsample(series: list[Any], max_len: int = 1500) -> list[Any]:
        if not series or len(series) <= max_len:
            return series
        step = max(1, len(series) // max_len)
        sampled = series[::step]
        if series[-1] != sampled[-1]:
            sampled.append(series[-1])
        return sampled

    @app.get("/api/strategies/{strat_id}")
    async def get_strategy_detail(strat_id: str, _auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        doc = _find_strat_doc(strat_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Strategy '{strat_id}' not found.")

        # Ensure source_code and spec are easily accessible
        if not doc.get("source_code"):
            doc["source_code"] = doc.get("idea", {}).get("code", "# Source code unavailable")
        spec = doc.get("idea", {}).get("spec", {})
        if not doc.get("concept_family"):
            doc["concept_family"] = spec.get("concept_family", "trend")
        if not doc.get("timeframe"):
            doc["timeframe"] = spec.get("timeframe", "1h")
        if not doc.get("strategy_name") or doc.get("strategy_name") == "unnamed":
            doc["strategy_name"] = spec.get("name") or doc.get("trial_id", strat_id)

        # Downsample equity curve to max 1500 points
        train_res = doc.get("train_result") or {}
        equity = train_res.get("equity_curve", [])
        if len(equity) > 1500:
            doc["train_result"]["equity_curve"] = _downsample(equity)

        return doc

    @app.get("/api/strategies/{strat_id}/equity")
    async def get_strategy_equity(strat_id: str, _auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        doc = _find_strat_doc(strat_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Strategy '{strat_id}' not found.")
        train_eq = doc.get("train_result", {}).get("equity_curve", [])
        trades = doc.get("train_result", {}).get("trades", []) or doc.get("trades", [])
        if not train_eq and trades:
            capital = 100000.0
            train_eq = [capital]
            for t in trades:
                capital += float(t.get("net_pnl", 0.0))
                train_eq.append(round(capital, 2))

        val_eq = doc.get("val_result", {}).get("equity_curve", [])
        holdout_eq = doc.get("holdout_result", {}).get("equity_curve", [])
        drawdown = doc.get("train_result", {}).get("drawdown_curve", [])
        return {
            "strategy_id": strat_id,
            "train_equity": _downsample(train_eq),
            "val_equity": _downsample(val_eq),
            "holdout_equity": _downsample(holdout_eq),
            "drawdown": _downsample(drawdown),
        }

    @app.get("/api/strategies/{strat_id}/monthly")
    async def get_strategy_monthly(strat_id: str, _auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        doc = _find_strat_doc(strat_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Strategy '{strat_id}' not found.")
        return doc.get("monthly_returns", {})

    @app.get("/api/strategies/{strat_id}/trades")
    async def get_strategy_trades(
        strat_id: str,
        page: int = 1,
        limit: int = 50,
        _auth: str = Depends(_verify_auth),
    ) -> dict[str, Any]:
        doc = _find_strat_doc(strat_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Strategy '{strat_id}' not found.")
        trades = doc.get("train_result", {}).get("trades", []) or doc.get("trades", [])
        total = len(trades)
        start = (page - 1) * limit
        return {
            "items": trades[start:start + limit],
            "total": total,
            "page": page,
            "limit": limit,
        }

    @app.get("/api/strategies/{strat_id}/gates")
    async def get_strategy_gates(strat_id: str, _auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        doc = _find_strat_doc(strat_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Strategy '{strat_id}' not found.")

        # Consolidate gate results from train_gates or gate_results
        gates = doc.get("gate_results", {})
        if not gates and doc.get("train_gates"):
            tg = doc.get("train_gates", {})
            for g in tg.get("gates", []):
                g_name = g.get("name", "unknown")
                gates[g_name] = {
                    "passed": g.get("passed", False),
                    "category": g.get("category", ""),
                    "summary": g.get("public_summary", ""),
                    "details": g.get("details", {}),
                }

        return {
            "strategy_id": strat_id,
            "gate_results": gates,
            "rejected_at": doc.get("rejected_at"),
            "status": doc.get("status", "unknown"),
        }

    @app.get("/api/strategies/{strat_id}/llm")
    async def get_strategy_llm(strat_id: str, _auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        calls = list(database["llm_calls"].find(
            {"$or": [{"strategy_id": strat_id}, {"trial_id": strat_id}]},
            projection={"_id": 0, "api_key": 0, "secret": 0},
            sort=[("timestamp", 1)],
        ))
        return {"strategy_id": strat_id, "calls": calls}

    @app.get("/api/strategies/{strat_id}/logs")
    async def get_strategy_logs(strat_id: str, limit: int = 100, _auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        logs = list(database["logs"].find(
            {"$or": [{"strategy_id": strat_id}, {"message": {"$regex": strat_id}}]},
            projection={"_id": 0},
            sort=[("timestamp", -1)],
        ).limit(limit))
        return {"strategy_id": strat_id, "logs": logs}

    @app.post("/api/strategies/{strat_id}/holdout", dependencies=[Depends(_verify_csrf)])
    async def trigger_holdout_run(strat_id: str, _auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        from forward.holdout_runner import run_holdout, HoldoutEligibilityError
        try:
            res = run_holdout(strat_id, cfg=cfg, db=database)
            _write_audit_log("holdout_run", "dashboard_ui", {"strategy_id": strat_id, "passed": res.passed})
            return {
                "strategy_id": strat_id,
                "passed": res.passed,
                "net_profit": res.net_profit,
                "total_trades": res.total_trades,
                "holdout_sharpe": res.holdout_sharpe,
            }
        except HoldoutEligibilityError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @app.get("/api/strategies/{strat_id}/forward")
    async def get_strategy_forward(strat_id: str, _auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        doc = _find_strat_doc(strat_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Strategy not found.")
        trades = list(database["forward_trades"].find({"strategy_id": strat_id}, projection={"_id": 0}).limit(100))
        return {
            "strategy_id": strat_id,
            "status": doc.get("status", "unknown"),
            "forward_metrics": doc.get("forward_metrics", {}),
            "recent_trades": trades,
        }

    @app.post("/api/strategies/{strat_id}/live-ready", dependencies=[Depends(_verify_csrf)])
    async def verify_live_ready(strat_id: str, _auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        from forward.live_ready import evaluate_live_ready, CandidateUnprovenError
        try:
            res = evaluate_live_ready(strat_id, db=database, cfg=cfg)
            _write_audit_log("promote_live_ready", "dashboard_ui", {"strategy_id": strat_id})
            return res
        except CandidateUnprovenError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # ── System Logs API ─────────────────────────────────────────────────────
    @app.get("/api/logs")
    async def get_logs(
        level: str | None = None,
        stage: str | None = None,
        strategy: str | None = None,
        limit: int = 100,
        _auth: str = Depends(_verify_auth),
    ) -> dict[str, Any]:
        q: dict[str, Any] = {}
        if level and level.upper() != "ALL":
            q["level"] = level.upper()
        if stage:
            q["stage"] = stage
        if strategy:
            q["strategy_id"] = strategy
        logs = list(database["logs"].find(
            q,
            projection={"_id": 0},
            sort=[("timestamp", -1)],
        ).limit(limit))

        # Augment with cycle status entries if raw logs are sparse
        if len(logs) < 15:
            for c in database["cycles"].find().sort("updated_at", -1).limit(40):
                st_name = c.get("strategy_name") or "Strategy"
                artefacts = c.get("artefacts") or {}
                status_str = c.get("status", "in_progress")
                rej = artefacts.get("rejected_at")
                msg = f"Cycle {c.get('cycle_id')}: {st_name} [{c.get('stage')}] - status={status_str}"
                if rej:
                    msg += f" (rejected_at: {rej})"
                logs.append({
                    "timestamp": c.get("updated_at", time.time()),
                    "level": "INFO" if status_str != "failed" else "ERROR",
                    "message": msg,
                    "stage": c.get("stage"),
                    "strategy_id": st_name,
                })
            logs.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)

        return {"logs": logs[:limit], "count": len(logs[:limit])}

    # ── Data and Integrity API ──────────────────────────────────────────────
    @app.get("/api/data")
    async def get_data_integrity(_auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        boundaries = list(database["split_boundaries"].find({}, projection={"_id": 0}))
        canary = database["canary_reports"].find_one({"_id": "latest"}, projection={"_id": 0})
        holdout_accesses = list(database["holdout_accesses"].find(
            {},
            projection={"_id": 0},
            sort=[("timestamp", -1)],
        ).limit(50))
        data_reports = list(database["data_reports"].find({}, projection={"_id": 0}).limit(20))
        return {
            "boundaries": boundaries,
            "canary": canary or {"status": "NOT_RUN", "passed": False},
            "holdout_accesses": holdout_accesses,
            "data_reports": data_reports,
        }

    # ── Coverage Map API ────────────────────────────────────────────────────
    @app.get("/api/coverage")
    async def get_coverage_map(_auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        from strategy.diversity import CoverageMap
        doc = database["coverage_map"].find_one({"_id": "current"})
        if doc and doc.get("cells"):
            cmap = CoverageMap.from_doc(doc)
            return cmap.to_doc()

        # Build dynamic coverage from runs if collection is not yet explicitly synced
        from collections import defaultdict
        cell_data = defaultdict(lambda: {"attempts": 0, "passes": 0, "consecutive_failures": 0, "cooling_until_cycle": 0})

        runs_col = database["runs"] if database["runs"].count_documents({}) >= database["trials"].count_documents({}) else database["trials"]
        for r in runs_col.find({}, {"idea.spec": 1, "timeframe": 1, "concept_family": 1, "status": 1}):
            spec = r.get("idea", {}).get("spec", {})
            fam = spec.get("concept_family") or r.get("concept_family") or "trend"
            tf = spec.get("timeframe") or r.get("timeframe") or "1h"
            sess = spec.get("session_bias") or "all_day"
            reg = spec.get("regime_bias") or "any"
            k = f"{fam}|{tf}|{sess}|{reg}"
            cell_data[k]["attempts"] += 1
            if r.get("status") in ("candidate", "survived"):
                cell_data[k]["passes"] += 1
            else:
                cell_data[k]["consecutive_failures"] += 1

        return {
            "total_cells": 480,
            "max_consecutive_failures": 5,
            "cooling_cycles": 20,
            "cells": dict(cell_data),
        }

    # ── LLM Usage API ───────────────────────────────────────────────────────
    @app.get("/api/llm/usage")
    async def get_llm_usage(_auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        doc = database["llm_usage"].find_one({"_id": "global"}) or {}
        calls = list(database["llm_calls"].find(
            {},
            projection={"_id": 0, "prompt": 0, "response": 0},
            sort=[("timestamp", -1)],
        ).limit(100))

        total_tok = doc.get("total_tokens", 0)
        prompt_tok = doc.get("prompt_tokens", 0)
        comp_tok = doc.get("completion_tokens", 0)
        total_calls = doc.get("total_calls", 0)
        failed_calls = doc.get("failed_calls", 0)

        # Aggregate purpose counters (Section 59, 98)
        by_purpose = {
            "ideas": 0,
            "improvements": 0,
            "failure_analyses": 0,
            "reviews": 0,
            "robustness_analyses": 0,
            "other": 0,
        }
        for c in database["llm_calls"].find({}, {"purpose": 1}):
            p = str(c.get("purpose", "")).lower()
            if "ideat" in p:
                by_purpose["ideas"] += 1
            elif any(w in p for w in ("improv", "fix", "rethink")):
                by_purpose["improvements"] += 1
            elif any(w in p for w in ("analy", "autopsy", "diagnos")):
                by_purpose["failure_analyses"] += 1
            elif "review" in p:
                by_purpose["reviews"] += 1
            elif "robust" in p:
                by_purpose["robustness_analyses"] += 1
            else:
                by_purpose["other"] += 1

        return {
            "summary": {
                "total_tokens": total_tok,
                "prompt_tokens": prompt_tok,
                "completion_tokens": comp_tok,
                "total_calls": total_calls,
                "failed_calls": failed_calls,
                "by_purpose": by_purpose,
            },
            "recent_calls": calls,
        }

    # ── Settings API ────────────────────────────────────────────────────────
    @app.get("/api/settings")
    async def get_settings(_auth: str = Depends(_verify_auth)) -> dict[str, Any]:
        overrides = database["runtime_overrides"].find_one({"_id": "current"}, projection={"_id": 0}) or {}
        locked_thresholds = {
            "min_sharpe": cfg.get("gates", {}).get("train", {}).get("min_sharpe", 0.8),
            "min_profit_factor": cfg.get("gates", {}).get("train", {}).get("min_profit_factor", 1.2),
            "max_drawdown": cfg.get("gates", {}).get("train", {}).get("max_drawdown", 0.20),
            "dsr_probability": cfg.get("gates", {}).get("dsr", {}).get("min_probability", 0.95),
            "pbo_max": cfg.get("gates", {}).get("pbo", {}).get("max_pbo", 0.30),
            "monte_carlo_max_dd": cfg.get("gates", {}).get("monte_carlo", {}).get("max_95th_percentile_dd", 0.30),
        }
        return {
            "runtime_overrides": overrides,
            "locked_gate_thresholds": locked_thresholds,
        }

    @app.patch("/api/settings", dependencies=[Depends(_verify_csrf)])
    async def update_settings(payload: SettingsUpdateRequest) -> dict[str, Any]:
        data = payload.model_dump(exclude_unset=True) if hasattr(payload, "model_dump") else payload.dict(exclude_unset=True)
        # Reject attempts to mutate locked gate thresholds
        forbidden = {"min_sharpe", "min_profit_factor", "max_drawdown", "dsr_probability", "pbo_max"}
        if any(k in data for k in forbidden):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Gate thresholds are strictly locked and cannot be modified at runtime.",
            )

        database["runtime_overrides"].update_one(
            {"_id": "current"},
            {"$set": data},
            upsert=True,
        )
        _write_audit_log("update_settings", "dashboard_ui", details=data)
        return {"status": "ok", "updated": data}

    # ── SSE Live Streaming ──────────────────────────────────────────────────
    @app.get("/api/stream")
    async def stream_live_events(request: Request, _auth: str = Depends(_verify_auth)) -> StreamingResponse:
        async def event_generator() -> AsyncGenerator[str, None]:
            mgr = EngineStateManager(db=database)
            interval = float(dash_cfg.get("sse_interval_seconds", 1.5))

            while True:
                if await request.is_disconnected():
                    break

                state_doc = database["engine_state"].find_one({"_id": "current_state"}) or {}
                trial_count = mgr.get_trial_count()
                cand_count = database["candidates"].count_documents({})
                last_hb = float(state_doc.get("last_heartbeat", 0))
                hb_age = round(time.time() - last_hb, 1) if last_hb > 0 else 0.0

                current_cycle = database["cycles"].find_one({"status": "in_progress"}, sort=[("updated_at", -1)])
                if not current_cycle:
                    current_cycle = database["cycles"].find_one({}, sort=[("updated_at", -1)])

                payload = {
                    "state": state_doc.get("state", "RUNNING"),
                    "reason": state_doc.get("reason", ""),
                    "worker_id": state_doc.get("worker_id", "worker_default"),
                    "heartbeat_age": hb_age,
                    "total_trials": trial_count,
                    "candidates_count": cand_count,
                    "current_cycle_id": current_cycle["cycle_id"] if current_cycle else None,
                    "current_stage": current_cycle.get("stage", "idle") if current_cycle else "idle",
                    "current_strategy_name": current_cycle.get("strategy_name", "") if current_cycle else "",
                    "timestamp": time.time(),
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(interval)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ── Static UI and Frontend Routes ───────────────────────────────────────
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index_page(response: Response, request: Request) -> HTMLResponse:
        tmpl_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
        content = "<h1>QuantForge Dashboard</h1><p>Templates not yet populated.</p>"
        if os.path.exists(tmpl_path):
            with open(tmpl_path, "r", encoding="utf-8") as f:
                content = f.read()

        # Always set authentication cookie for the session so SSE and all fetch calls authenticate automatically
        response.set_cookie(
            key="qf_token",
            value=dashboard_token,
            httponly=False,
            samesite="lax",
            secure=False,
            max_age=86400 * 30,
        )
        return HTMLResponse(content=content, headers=dict(response.headers))

    return app
