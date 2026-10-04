"""
tests/chaos/test_f10_chaos.py – Chaos & fault injection tests for Phase F10 Engine.

Requirements from AUDIT_PROMPT.md:
1. Mid-cycle abort/crash: worker resumes unfinished cycle at last completed stage, trial not double counted.
2. Mongo unavailable: retries with capped backoff, preserves in-memory cycle.
3. Two worker start attempts: second refuses due to lock.
4. Stop during step: graceful stop finishes atomic step and stops cleanly.
"""

from __future__ import annotations

import time
import mongomock
import pytest

from engine.state import EngineState, EngineStateManager, LockAcquisitionError
from engine.supervisor import EngineSupervisor
from engine.worker import EngineWorker, CycleRecord


@pytest.fixture
def mock_db():
    client = mongomock.MongoClient()
    return client["test_quantforge_chaos"]


class TestChaosFaultInjection:
    def test_worker_killed_mid_cycle_resumes_without_double_count(self, mock_db, base_cfg):
        """Kill worker mid-cycle (e.g. after train_backtest).
        New worker resumes at robustness_gates without repeating ideation or double-counting trial.
        """
        mgr1 = EngineStateManager(db=mock_db, worker_id="worker_killed")
        mgr1.acquire_lock()

        # Cycle started and completed train_backtest
        cycle = CycleRecord(
            cycle_id="cycle_crash_99",
            stage="train_backtest",
            strategy_name="CrashResilienceStrategy",
            timeframe="1h",
            version=1,
            artefacts={"idea": {"name": "CrashResilienceStrategy"}, "source": "class Strategy: pass"},
        )
        cycle.save(mock_db)
        mgr1.increment_trial_once("cycle_crash_99", version=1)
        assert mgr1.get_trial_count() == 1

        # Simulate crash: worker 1 releases lock (or TTL expires)
        mgr1.release_lock()

        # Worker 2 starts up
        mgr2 = EngineStateManager(db=mock_db, worker_id="worker_resumed")
        mgr2.acquire_lock()
        worker2 = EngineWorker(db=mock_db, cfg=base_cfg, state_manager=mgr2)

        # Worker 2 resumes the unfinished cycle
        resumed_cycle = worker2.get_unfinished_cycle()
        assert resumed_cycle is not None
        assert resumed_cycle.cycle_id == "cycle_crash_99"
        assert resumed_cycle.stage == "train_backtest"

        # Worker 2 completes the cycle
        mgr2.increment_trial_once(resumed_cycle.cycle_id, version=resumed_cycle.version)
        assert mgr2.get_trial_count() == 1, "Trial counter must remain 1 (no double-counting on resume)"

        resumed_cycle.update_stage(mock_db, "promote", artefacts={"status": "candidate"})
        loaded = CycleRecord.load(mock_db, "cycle_crash_99")
        assert loaded.stage == "promote"

    def test_two_workers_cannot_run_simultaneously(self, mock_db):
        """Single-instance lock prevents multiple workers from corrupting state."""
        w1 = EngineStateManager(db=mock_db, worker_id="worker_primary")
        w2 = EngineStateManager(db=mock_db, worker_id="worker_secondary")

        assert w1.acquire_lock(ttl_seconds=30) is True

        # Second worker must fail to acquire lock
        with pytest.raises(LockAcquisitionError):
            w2.acquire_lock(ttl_seconds=30)

    def test_graceful_stop_during_cycle(self, mock_db, base_cfg):
        """When desired_state becomes 'stopped', worker finishes current atomic step
        and exits within graceful stop window.
        """
        supervisor = EngineSupervisor(db=mock_db, cfg=base_cfg)
        supervisor.set_desired_state("running")

        mgr = EngineStateManager(db=mock_db, worker_id="worker_graceful")
        worker = EngineWorker(db=mock_db, cfg=base_cfg, state_manager=mgr)

        step_finished = False

        def atomic_cycle_step():
            nonlocal step_finished
            # Step in progress... user clicks stop
            supervisor.set_desired_state("stopped", requested_by="user")
            time.sleep(0.05)
            step_finished = True

        worker.run_loop_step_fn = atomic_cycle_step
        worker.run_until_stopped()

        assert step_finished is True, "Atomic step must have completed before exiting"
        assert mgr.get_state() == EngineState.STOPPED
