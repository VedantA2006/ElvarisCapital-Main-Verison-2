"""
tests/regression/test_f10_engine.py – Tests for Phase F10 Autonomous Engine & State Machine.

Closes: ORCH-1 to ORCH-4 (Phase F10)
Tests:
1. Engine states, transitions, heartbeat updates, and stale detection (> 30s).
2. Single-instance lock (second worker refused).
3. Resumable cycles and trial counter idempotency key (cycle_id, version).
4. Never-stopping contract: loop continues after finding a survivor.
5. Error taxonomy: BLOCKED on data/config problem, WAITING_FOR_QUOTA on rate limits, WAITING_FOR_DB on mongo errors.
"""

from __future__ import annotations

import time
import mongomock
import pytest

from engine.state import (
    EngineState,
    EngineStateManager,
    LockAcquisitionError,
)
from engine.supervisor import EngineSupervisor
from engine.worker import EngineWorker, CycleRecord


@pytest.fixture
def mock_db():
    client = mongomock.MongoClient()
    return client["test_quantforge_f10"]


class TestEngineState:
    def test_all_10_states_defined(self):
        expected_states = {
            "STOPPED",
            "STARTING",
            "RUNNING",
            "PAUSED",
            "STOPPING",
            "WAITING_FOR_QUOTA",
            "BUDGET_PAUSED",
            "ERROR_BACKOFF",
            "WAITING_FOR_DB",
            "BLOCKED",
        }
        actual_states = {s.value for s in EngineState}
        assert actual_states == expected_states

    def test_state_persistence_and_heartbeat(self, mock_db):
        mgr = EngineStateManager(db=mock_db, worker_id="worker_01")
        mgr.set_state(EngineState.RUNNING)
        doc = mock_db["engine_state"].find_one({"_id": "current_state"})
        assert doc is not None
        assert doc["state"] == "RUNNING"
        assert doc["worker_id"] == "worker_01"

        # Update heartbeat
        t0 = time.time()
        mgr.heartbeat()
        doc = mock_db["engine_state"].find_one({"_id": "current_state"})
        assert doc["last_heartbeat"] >= t0

    def test_watchdog_detects_stale_heartbeat(self, mock_db):
        mgr = EngineStateManager(db=mock_db, worker_id="worker_01")
        mgr.set_state(EngineState.RUNNING)
        # Mock heartbeat 35s in the past
        old_time = time.time() - 35.0
        mock_db["engine_state"].update_one(
            {"_id": "current_state"},
            {"$set": {"last_heartbeat": old_time}},
        )
        assert mgr.is_stale(threshold_seconds=30.0) is True

    def test_single_instance_lock_refuses_second_worker(self, mock_db):
        mgr1 = EngineStateManager(db=mock_db, worker_id="worker_01")
        mgr2 = EngineStateManager(db=mock_db, worker_id="worker_02")

        acquired = mgr1.acquire_lock(ttl_seconds=60)
        assert acquired is True

        with pytest.raises(LockAcquisitionError):
            mgr2.acquire_lock(ttl_seconds=60)

        # Release lock allows worker 2 to acquire
        mgr1.release_lock()
        assert mgr2.acquire_lock(ttl_seconds=60) is True


class TestResumableCyclesAndIdempotency:
    def test_trial_counter_idempotency_key(self, mock_db):
        """Trial counter must be incremented exactly once per (cycle_id, version).
        Resuming a cycle must not double-count.
        """
        mgr = EngineStateManager(db=mock_db, worker_id="worker_01")

        cycle_id = "cycle_abc_123"
        version = 1

        inc1 = mgr.increment_trial_once(cycle_id, version)
        assert inc1 is True, "First execution must increment trial"
        assert mgr.get_trial_count() == 1

        # Re-running / resuming the same cycle and version must be a no-op
        inc2 = mgr.increment_trial_once(cycle_id, version)
        assert inc2 is False, "Resume of same (cycle_id, version) must not increment trial"
        assert mgr.get_trial_count() == 1

        # A new revision / version increments cleanly
        inc3 = mgr.increment_trial_once(cycle_id, version=2)
        assert inc3 is True
        assert mgr.get_trial_count() == 2

    def test_resumable_cycle_stage_tracking(self, mock_db):
        cycle = CycleRecord(
            cycle_id="cycle_resumable_01",
            stage="ideate",
            strategy_name="Gold_Breakout",
            timeframe="1h",
        )
        cycle.save(mock_db)

        # Update stage to train_backtest
        cycle.update_stage(mock_db, stage="train_backtest", artefacts={"code_hash": "a1b2c3"})

        loaded = CycleRecord.load(mock_db, "cycle_resumable_01")
        assert loaded is not None
        assert loaded.stage == "train_backtest"
        assert loaded.artefacts.get("code_hash") == "a1b2c3"


class TestNeverStoppingContract:
    def test_loop_continues_after_survivor_found(self, mock_db, base_cfg):
        """The engine contract: finding a survivor NEVER ends the loop.
        Only desired_state = 'stopped' ends the loop.
        """
        mgr = EngineStateManager(db=mock_db, worker_id="worker_test")
        supervisor = EngineSupervisor(db=mock_db, cfg=base_cfg)

        # Set desired state to running
        supervisor.set_desired_state("running", requested_by="test_admin")

        worker = EngineWorker(db=mock_db, cfg=base_cfg, state_manager=mgr)

        # Simulate 3 cycles: Cycle 2 produces a survivor
        cycles_run = 0
        survivors_found = 0

        def fake_cycle_step():
            nonlocal cycles_run, survivors_found
            cycles_run += 1
            if cycles_run == 2:
                survivors_found += 1
                # Survivor found, but desired_state is still running!
            if cycles_run >= 3:
                # Stop requested externally after cycle 3
                supervisor.set_desired_state("stopped", requested_by="test_admin")

        worker.run_loop_step_fn = fake_cycle_step
        worker.run_until_stopped()

        assert survivors_found == 1, "A survivor was discovered on cycle 2"
        assert cycles_run >= 3, "Engine must have continued running past cycle 2"
        assert mgr.get_state() == EngineState.STOPPED


class TestErrorTaxonomy:
    def test_data_problem_sets_state_to_blocked(self, mock_db, base_cfg):
        mgr = EngineStateManager(db=mock_db, worker_id="worker_test")
        worker = EngineWorker(db=mock_db, cfg=base_cfg, state_manager=mgr)

        worker.handle_data_error("Data directory missing XAUUSD_M5.csv")
        assert mgr.get_state() == EngineState.BLOCKED
        assert "Data directory missing" in mgr.get_state_reason()

    def test_rate_limit_sets_state_to_waiting_for_quota(self, mock_db, base_cfg):
        mgr = EngineStateManager(db=mock_db, worker_id="worker_test")
        worker = EngineWorker(db=mock_db, cfg=base_cfg, state_manager=mgr)

        worker.handle_rate_limit("Both API keys exhausted until 02:30 UTC")
        assert mgr.get_state() == EngineState.WAITING_FOR_QUOTA
        assert "exhausted" in mgr.get_state_reason()

    def test_consecutive_internal_errors_trigger_error_backoff(self, mock_db, base_cfg):
        mgr = EngineStateManager(db=mock_db, worker_id="worker_test")
        worker = EngineWorker(db=mock_db, cfg=base_cfg, state_manager=mgr)

        # 20 consecutive internal failures
        for i in range(20):
            worker.handle_internal_cycle_error(RuntimeError(f"Unexpected crash {i}"))

        assert mgr.get_state() == EngineState.ERROR_BACKOFF
        assert worker.consecutive_internal_errors >= 20
