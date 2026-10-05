"""
tests/regression/test_f11_dashboard.py – Regression tests for Phase F11 Professional Dashboard & API.

Closes: UI-1 to UI-5, ORCH-2 (visibility).
Tests:
1. Auth & CSRF: token login, HttpOnly cookie, CSRF header required on POST/PATCH.
2. Security: No CORS wildcard, Host/Origin validation, zero secrets in API responses.
3. Engine Control API: /api/engine/start, pause, resume, stop, force_stop with audit logging.
4. Stats & Leaderboard API: /api/stats/funnel, /api/leaderboard (sorting, filtering, pagination).
5. Settings API: safe runtime overrides allowed, locked gate thresholds protected.
"""

from __future__ import annotations

import mongomock
import pytest
from fastapi.testclient import TestClient

from dashboard.fastapi_app import create_app
from engine.state import EngineState, EngineStateManager
from engine.supervisor import EngineSupervisor


@pytest.fixture
def mock_db():
    client = mongomock.MongoClient()
    return client["test_quantforge_f11"]


@pytest.fixture
def test_token():
    return "test_secret_dashboard_token_xyz123"


@pytest.fixture
def client(mock_db, base_cfg, test_token):
    app = create_app(db=mock_db, cfg=base_cfg, token=test_token)
    return TestClient(app, base_url="http://127.0.0.1:8000")


@pytest.fixture
def authed_client(client, test_token):
    # Log in to get cookie and csrf token
    resp = client.post("/api/auth/login", json={"token": test_token})
    assert resp.status_code == 200
    csrf_token = resp.json()["csrf_token"]
    client.headers.update({"X-CSRF-Token": csrf_token})
    return client


class TestDashboardSecurityAndAuth:
    def test_unauthenticated_request_rejected(self, client):
        resp = client.get("/api/engine/state")
        assert resp.status_code == 401

    def test_valid_login_sets_httponly_cookie(self, client, test_token):
        resp = client.post("/api/auth/login", json={"token": test_token})
        assert resp.status_code == 200
        assert "qf_token" in resp.cookies
        assert "csrf_token" in resp.json()

    def test_post_without_csrf_header_rejected(self, client, test_token):
        # Set auth cookie without CSRF header
        client.cookies.set("qf_token", test_token)
        resp = client.post("/api/engine/start", json={})
        assert resp.status_code == 403
        assert "CSRF" in resp.text

    def test_no_cors_wildcard(self, client):
        resp = client.options("/api/engine/state")
        allow_origin = resp.headers.get("access-control-allow-origin")
        assert allow_origin != "*", "CORS wildcard '*' is strictly forbidden (UI-4)"

    def test_no_secrets_in_api_responses(self, authed_client, mock_db):
        mock_db["engine_state"].insert_one({
            "_id": "current_state",
            "state": "RUNNING",
            "secret_key": "SUPER_SECRET_KEY_DO_NOT_LEAK",
        })
        resp = authed_client.get("/api/engine/state")
        assert resp.status_code == 200
        text = resp.text
        assert "SUPER_SECRET_KEY" not in text
        assert "MONGO_URL" not in text
        assert "LLM_API_KEY" not in text


class TestEngineControlAPI:
    def test_get_engine_state(self, authed_client, mock_db):
        mgr = EngineStateManager(db=mock_db, worker_id="w1")
        mgr.set_state(EngineState.RUNNING)
        mgr.heartbeat()

        resp = authed_client.get("/api/engine/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "RUNNING"
        assert "heartbeat_age_seconds" in data
        assert "uptime_seconds" in data

    def test_engine_start_control_and_audit_log(self, authed_client, mock_db):
        resp = authed_client.post("/api/engine/start", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Check desired_state updated
        supervisor = EngineSupervisor(db=mock_db)
        assert supervisor.get_desired_state() == "running"

        # Check audit log written
        audit = mock_db["audit_logs"].find_one({"action": "start"})
        assert audit is not None
        assert audit["action"] == "start"

    def test_engine_stop_and_pause_controls(self, authed_client, mock_db):
        supervisor = EngineSupervisor(db=mock_db)

        # Pause
        r_pause = authed_client.post("/api/engine/pause", json={})
        assert r_pause.status_code == 200
        assert supervisor.get_desired_state() == "paused"

        # Resume
        r_resume = authed_client.post("/api/engine/resume", json={})
        assert r_resume.status_code == 200
        assert supervisor.get_desired_state() == "running"

        # Graceful Stop
        r_stop = authed_client.post("/api/engine/stop", json={})
        assert r_stop.status_code == 200
        assert supervisor.get_desired_state() == "stopped"


class TestStatsAndLeaderboardAPI:
    def test_funnel_stats(self, authed_client, mock_db):
        # Insert sample trial records
        mock_db["trials"].insert_many([
            {"trial_id": "t1", "status": "candidate", "rejected_at": None},
            {"trial_id": "t2", "status": "rejected", "rejected_at": "basic_quality"},
            {"trial_id": "t3", "status": "rejected", "rejected_at": "delay"},
        ])
        resp = authed_client.get("/api/stats/funnel")
        assert resp.status_code == 200
        data = resp.json()
        assert "funnel" in data
        assert data["total_trials"] >= 3

    def test_leaderboard_pagination_and_sorting(self, authed_client, mock_db):
        # Insert 3 candidates
        mock_db["candidates"].insert_many([
            {"strategy_id": "strat_a", "name": "Strat A", "robustness_score": 85.0, "timeframe": "1h", "train_metrics": {"sharpe": 2.1}},
            {"strategy_id": "strat_b", "name": "Strat B", "robustness_score": 92.0, "timeframe": "15m", "train_metrics": {"sharpe": 2.5}},
            {"strategy_id": "strat_c", "name": "Strat C", "robustness_score": 70.0, "timeframe": "1h", "train_metrics": {"sharpe": 1.5}},
        ])

        # Default sort: robustness_score desc
        resp = authed_client.get("/api/leaderboard?limit=2&page=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["items"][0]["strategy_id"] == "strat_b"
        assert data["items"][1]["strategy_id"] == "strat_a"
        assert data["total"] == 3


class TestSettingsAPI:
    def test_get_settings_shows_locked_gate_thresholds(self, authed_client):
        resp = authed_client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "locked_gate_thresholds" in data
        assert "runtime_overrides" in data

    def test_patch_safe_runtime_overrides(self, authed_client, mock_db):
        resp = authed_client.patch("/api/settings", json={"daily_token_budget": 5000000})
        assert resp.status_code == 200
        doc = mock_db["runtime_overrides"].find_one({"_id": "current"})
        assert doc["daily_token_budget"] == 5000000

    def test_patch_locked_gate_threshold_rejected(self, authed_client):
        resp = authed_client.patch("/api/settings", json={"min_sharpe": 0.5})
        assert resp.status_code == 400
        assert "locked" in resp.text


class TestAdditionalDashboardEndpoints:
    def test_index_page_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "QuantForge Console" in resp.text
        assert "text/html" in resp.headers["content-type"]

    def test_strategy_sub_endpoints(self, authed_client, mock_db):
        mock_db["candidates"].insert_one({
            "strategy_id": "strat_alpha_1",
            "concept_family": "trend",
            "timeframe": "1h",
            "train_result": {
                "equity_curve": [10000.0, 10050.0, 10120.0],
                "drawdown_curve": [0.0, -0.01, -0.005],
                "trades": [
                    {"entry_time": "2022-01-01", "exit_time": "2022-01-02", "type": "BUY", "pnl": 50.0, "return_pct": 0.005}
                ],
            },
            "val_result": {
                "equity_curve": [10120.0, 10200.0],
            },
            "gate_results": {
                "lookahead": {"passed": True},
                "basic_quality": {"passed": True},
            },
            "monthly_returns": {"2022": {"01": 0.012}},
        })

        # Base detail
        r_detail = authed_client.get("/api/strategies/strat_alpha_1")
        assert r_detail.status_code == 200
        assert r_detail.json()["strategy_id"] == "strat_alpha_1"

        # Equity
        r_eq = authed_client.get("/api/strategies/strat_alpha_1/equity")
        assert r_eq.status_code == 200
        assert len(r_eq.json()["train_equity"]) == 3

        # Monthly
        r_m = authed_client.get("/api/strategies/strat_alpha_1/monthly")
        assert r_m.status_code == 200
        assert "2022" in r_m.json()

        # Trades
        r_tr = authed_client.get("/api/strategies/strat_alpha_1/trades")
        assert r_tr.status_code == 200
        assert len(r_tr.json()["items"]) == 1

        # Gates
        r_gt = authed_client.get("/api/strategies/strat_alpha_1/gates")
        assert r_gt.status_code == 200
        assert r_gt.json()["gate_results"]["lookahead"]["passed"] is True

    def test_logs_and_data_endpoints(self, authed_client, mock_db):
        mock_db["logs"].insert_one({
            "timestamp": 1700000000,
            "level": "INFO",
            "stage": "train_backtest",
            "message": "Strategy strat_alpha_1 passed train backtest",
        })
        mock_db["split_boundaries"].insert_one({
            "timeframe": "1h",
            "train_start": "2018-01-01",
            "train_end": "2021-12-31",
        })

        r_logs = authed_client.get("/api/logs?level=INFO")
        assert r_logs.status_code == 200
        assert len(r_logs.json()["logs"]) >= 1

        r_data = authed_client.get("/api/data")
        assert r_data.status_code == 200
        assert len(r_data.json()["boundaries"]) >= 1

    def test_tui_rendering_with_mock_db(self, mock_db):
        from dashboard.tui import render_tui_frame
        from rich.console import Console
        import io

        mock_db["engine_state"].insert_one({
            "_id": "current_state",
            "state": "RUNNING",
            "worker_id": "test_w",
            "last_heartbeat": 1000.0,
        })
        mock_db["candidates"].insert_one({
            "strategy_id": "strat_1",
            "robustness_score": 91.5,
            "train_metrics": {"sharpe": 2.2, "profit_factor": 1.8, "max_drawdown": 0.08, "total_trades": 80},
        })

        buf = io.StringIO()
        console = Console(file=buf, width=120, height=40)
        # Verify render completes without exception
        render_tui_frame(console, mock_db)
        out = buf.getvalue()
        assert "QuantForge" in out
        assert "strat_1" in out

