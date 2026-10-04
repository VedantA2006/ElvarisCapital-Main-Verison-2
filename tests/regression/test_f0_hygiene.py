"""
Phase F0 regression tests: repo hygiene and test foundation.

Defects covered (see docs/AUDIT.md):
    DB-1   only the first trial document is stored; trial counter never advances
    DB-2   `main.py leaderboard` crashes (dict passed to get_db)
    DB-3   orchestrator / dashboard bypass the test-database override
    ORCH-3 database errors swallowed with `except: pass`
    LLM-6  no requirements.txt
    UI-4   dashboard sends `Access-Control-Allow-Origin: *`
    F0     strict config schema (unknown keys, duplicate keys, bad ranges)

Every test here was written BEFORE the fix and was shown to FAIL on the old code.
"""

from __future__ import annotations

import ast
import copy
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("core", "llm", "storage", "dashboard", "scripts", "sandbox")


# ─── helpers ────────────────────────────────────────────────────────────────

def _stored_run_count(db) -> int:
    """Count run documents no matter which collection the engine uses for them."""
    return sum(db[name].count_documents({"trial_id": {"$exists": True}})
               for name in ("runs", "trials"))


class _StubLLM:
    """Stands in for LLMClient: returns an idea, then code that cannot load."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, temperature=None, **kwargs):
        self.calls += 1
        return SimpleNamespace(content="stub", usage={"total_tokens": 0}, model="stub")

    def extract_json(self, text):
        return {"name": f"stub_idea_{self.calls}", "concept_family": "test",
                "hypothesis": "stub", "entry_logic": "stub", "exit_logic": "stub"}

    def extract_code(self, text):
        return "def broken(:\n    pass\n"   # SyntaxError -> rejected, never backtested


def _make_orchestrator(cfg, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY_1", os.environ.get("LLM_API_KEY_1") or "dummy-key")
    from llm.orchestrator import Orchestrator
    orch = Orchestrator(cfg)
    orch._llm = _StubLLM()
    return orch


def _blank_record(i: int):
    from llm.orchestrator import TrialRecord
    return TrialRecord(trial_id=f"trial-{i:06d}", strategy_name=f"s{i}", timeframe="1h",
                       idea={}, source_code="", source_hash="", status="rejected")


# ─── DB-1 ───────────────────────────────────────────────────────────────────

class TestDB1RunsArePersisted:

    def test_orchestrator_uses_the_test_database(self, base_cfg, fresh_db, monkeypatch):
        """DB-3: the orchestrator must honour QF_MONGO_DB, never write to production."""
        orch = _make_orchestrator(base_cfg, monkeypatch)
        assert orch._db.name == fresh_db.name, \
            f"orchestrator writes to {orch._db.name!r}, tests expect {fresh_db.name!r}"

    def test_1000_runs_are_all_stored(self, base_cfg, fresh_db, monkeypatch):
        from storage import mongo
        orch = _make_orchestrator(base_cfg, monkeypatch)
        # Use the database that has the production indexes (ensure_indexes ran on it).
        # Without this, the old DB-3 bug sends writes to an unindexed DB and hides DB-1.
        orch._db = fresh_db
        mongo.increment_trial_counter()          # the counter exists, as in production
        for i in range(1000):
            orch._finish(_blank_record(i), time.perf_counter())
        stored = _stored_run_count(fresh_db)
        assert stored == 1000, f"only {stored} of 1000 run documents were stored"

    def test_trial_counter_counts_every_attempted_idea(self, base_cfg, fresh_db, monkeypatch):
        from storage import mongo
        orch = _make_orchestrator(base_cfg, monkeypatch)
        for _ in range(3):
            rec = orch.run_trial("1h")
            assert rec.status in ("rejected", "error")
        assert mongo.get_trial_count() == 3, \
            f"trial counter is {mongo.get_trial_count()} after 3 attempted ideas"
        assert _stored_run_count(orch._db) == 3

    def test_persistence_failure_is_not_silent(self, base_cfg, fresh_db, monkeypatch):
        """ORCH-3: a failed write must raise, never vanish."""
        orch = _make_orchestrator(base_cfg, monkeypatch)

        class _Boom:
            def __getattr__(self, name):
                def fail(*a, **k):
                    raise RuntimeError("simulated database outage")
                return fail

        class _FailingDB:
            name = fresh_db.name
            def __getitem__(self, key):
                return _Boom()

        orch._db = _FailingDB()
        with pytest.raises(Exception) as exc:
            orch._finish(_blank_record(1), time.perf_counter())
        assert "simulated database outage" in str(exc.value) or exc.value.__cause__ is not None


class TestLegacyMigration:

    def test_legacy_trials_collection_is_migrated(self, fresh_db):
        """Old databases have the counter and one run in `trials` with a unique index."""
        from pymongo import IndexModel, ASCENDING
        from storage import mongo
        legacy = fresh_db["trials"]
        legacy.drop_indexes()
        legacy.create_indexes([IndexModel([("counter_id", ASCENDING)], unique=True)])
        legacy.insert_one({"counter_id": "global", "count": 7})
        legacy.insert_one({"trial_id": "trial-legacy-1", "status": "rejected"})

        mongo.ensure_indexes()     # runs the idempotent migration
        mongo.ensure_indexes()     # twice: must stay idempotent

        assert mongo.get_trial_count() == 7
        assert fresh_db["runs"].count_documents({"trial_id": "trial-legacy-1"}) == 1
        # new runs can now be inserted freely
        for i in range(5):
            fresh_db["runs"].insert_one({"trial_id": f"t{i}", "run_id": f"t{i}"})
        assert fresh_db["runs"].count_documents({}) == 6


# ─── DB-2 + CLI smoke ───────────────────────────────────────────────────────

class TestDB2Cli:

    def test_leaderboard_command_in_process(self, fresh_db, capsys):
        import main
        rc = main.main(["leaderboard"])
        assert rc == 0
        assert "Leaderboard is empty" in capsys.readouterr().out

    @pytest.mark.parametrize("argv", [
        ["--help"],
        ["verify-data", "--help"],
        ["run", "--help"],
        ["leaderboard", "--help"],
        ["dashboard", "--help"],
        ["leaderboard"],
        ["run", "--trials", "0"],
    ])
    def test_cli_command_subprocess(self, argv):
        env = dict(os.environ)
        env["QF_MONGO_BACKEND"] = "mock"          # offline, in-memory Mongo
        env["QF_MONGO_DB"] = "quantforge_test"
        env.setdefault("LLM_API_KEY_1", "dummy-key")
        p = subprocess.run([sys.executable, str(ROOT / "main.py"), *argv],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, f"{argv} exited {p.returncode}\nSTDERR:\n{p.stderr[-2000:]}"


# ─── Strict config ──────────────────────────────────────────────────────────

class TestStrictConfig:

    def _write(self, tmp_path, mutate=None, raw_append: str = "") -> Path:
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        if mutate:
            mutate(cfg)
        p = tmp_path / "config.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False) + raw_append, encoding="utf-8")
        return p

    def test_real_config_is_valid(self):
        from core.config import load_config
        load_config()

    def test_unknown_key_is_rejected(self, tmp_path):
        from core.config import load_config
        p = self._write(tmp_path, lambda c: c["gates"]["train"].update({"min_sharp": 0.8}))
        with pytest.raises(ValueError, match="min_sharp"):
            load_config(p)

    def test_unknown_top_level_section_is_rejected(self, tmp_path):
        from core.config import load_config
        p = self._write(tmp_path, lambda c: c.update({"gatez": {"x": 1}}))
        with pytest.raises(ValueError, match="gatez"):
            load_config(p)

    def test_duplicate_key_is_rejected(self, tmp_path):
        from core.config import load_config
        p = self._write(tmp_path, raw_append="\nmongo:\n  database: other\n")
        with pytest.raises(ValueError, match="(?i)duplicate"):
            load_config(p)

    @pytest.mark.parametrize("path,value", [
        (("sizing", "risk_per_trade"), 5.0),          # 500 % risk per trade
        (("sizing", "initial_equity"), -1.0),
        (("costs", "spread", "default"), -0.25),
        (("gates", "dsr", "min_probability"), 1.5),
        (("sandbox", "wall_clock_timeout"), 0),
        (("splits", "embargo_days"), -3),
    ])
    def test_out_of_range_value_is_rejected(self, tmp_path, path, value):
        from core.config import load_config

        def mutate(c):
            node = c
            for k in path[:-1]:
                node = node[k]
            node[path[-1]] = value

        p = self._write(tmp_path, mutate)
        with pytest.raises(ValueError):
            load_config(p)

    def test_wrong_type_is_rejected(self, tmp_path):
        from core.config import load_config
        p = self._write(tmp_path, lambda c: c["gates"]["train"].update({"min_sharpe": "high"}))
        with pytest.raises(ValueError):
            load_config(p)


# ─── Logging: no silently dropped writes ────────────────────────────────────

class TestLoggerDroppedWrites:

    def test_failed_mongo_log_write_is_counted(self, tmp_path, monkeypatch):
        from storage import logger as logger_mod

        class _BadCol:
            def insert_one(self, doc):
                raise RuntimeError("mongo down")

        monkeypatch.setattr(logger_mod, "col_logs", lambda: _BadCol())
        lg = logger_mod.QuantForgeLogger(jsonl_dir=str(tmp_path))
        lg.info("hello")
        lg.info("again")
        assert lg.dropped_writes == 2
        text = (tmp_path / "quantforge.jsonl").read_text(encoding="utf-8")
        assert "hello" in text, "the JSONL fallback must still contain the entry"


# ─── Source hygiene: no swallowed exceptions ────────────────────────────────

_LOGGY = ("log", "warn", "error", "exception", "critical", "debug", "info", "send_error")


def _is_broad(handler: ast.ExceptHandler) -> bool:
    t = handler.type
    if t is None:
        return True
    names = [t] if not isinstance(t, ast.Tuple) else list(t.elts)
    return any(isinstance(n, ast.Name) and n.id in ("Exception", "BaseException") for n in names)


def _handler_is_silent(handler: ast.ExceptHandler) -> bool:
    body = handler.body
    if all(isinstance(s, ast.Pass) for s in body):
        return True
    if not _is_broad(handler):
        return False
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Raise):
            return False
        if handler.name and isinstance(node, ast.Name) and node.id == handler.name:
            return False                      # the exception is reported / stored
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if any(k in name.lower() for k in _LOGGY):
                return False
    return True


def test_no_silently_swallowed_exceptions():
    offenders = []
    for d in SOURCE_DIRS:
        for f in (ROOT / d).rglob("*.py") if (ROOT / d).exists() else []:
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and _handler_is_silent(node):
                    offenders.append(f"{f.relative_to(ROOT)}:{node.lineno}")
    for f in (ROOT / "main.py",):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and _handler_is_silent(node):
                offenders.append(f"{f.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, "exceptions swallowed without logging or re-raising:\n  " + "\n  ".join(offenders)


# ─── requirements.txt / repo layout ─────────────────────────────────────────

_IMPORT_TO_DIST = {"yaml": "pyyaml", "dotenv": "python-dotenv", "sklearn": "scikit-learn"}


def _third_party_imports() -> set[str]:
    stdlib = set(sys.stdlib_module_names)
    local = {"core", "llm", "storage", "dashboard", "scripts", "main", "tests", "sandbox"}
    found = set()
    files = [ROOT / "main.py"] + [f for d in SOURCE_DIRS if (ROOT / d).exists()
                                  for f in (ROOT / d).rglob("*.py")]
    for f in files:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            else:
                continue
            for m in mods:
                top = m.split(".")[0]
                if top not in stdlib and top not in local and top != "__future__":
                    found.add(top)
    return found


def test_requirements_file_covers_every_import():
    req = ROOT / "requirements.txt"
    assert req.exists(), "requirements.txt is missing (LLM-6)"
    listed = set()
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            name = line.split(";")[0]
            for sep in ("==", ">=", "<=", "~=", "<", ">", "["):
                name = name.split(sep)[0]
            listed.add(name.strip().lower())
    missing = {m for m in _third_party_imports()
               if _IMPORT_TO_DIST.get(m, m).lower() not in listed}
    assert not missing, f"imported but not in requirements.txt: {sorted(missing)}"


def test_repo_layout():
    assert (ROOT / "README.md").exists(), "README.md is missing"
    assert not (ROOT / "patch_hole.py").exists(), "one-off scripts belong in scripts/"
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".env", "__pycache__", "logs/"):
        assert pattern in gi, f".gitignore does not cover {pattern}"


# ─── UI-4: no CORS wildcard ─────────────────────────────────────────────────

def test_dashboard_sends_no_cors_wildcard(fresh_db):
    import threading
    import urllib.request
    from http.server import HTTPServer
    from dashboard import app as dash

    dash.DashboardHandler.cfg = {}
    dash.DashboardHandler.db = fresh_db
    server = HTTPServer(("127.0.0.1", 0), dash.DashboardHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/api/stats"
        with urllib.request.urlopen(url, timeout=10) as r:
            assert r.status == 200
            assert r.headers.get("Access-Control-Allow-Origin") != "*"
    finally:
        server.shutdown()
        server.server_close()
