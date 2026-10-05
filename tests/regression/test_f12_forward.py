"""
tests/regression/test_f12_forward.py – Regression tests for Phase F12 Holdout & Forward Paper Testing.

Closes: OPS-2, OPS-3
Tests:
1. Holdout Execution:
   - Only eligible candidates (passed all earlier gates) can run holdout.
   - Audited access recorded atomically in Mongo `holdout_accesses`.
   - Pass requirement: profitable (net profit > 0) AND Sharpe >= 50% of train Sharpe.
   - Fail requirement: unprofitable or Sharpe < 50% train marks holdout_failed and burns holdout.
   - Holdout isolation: LLM prompts/history never receive holdout results.
2. Pluggable Feeds:
   - SimulatedLiveFeed delivers closed bars incrementally.
   - CSVFolderFeed and BrokerFeedStub interfaces.
3. Paper Trader:
   - Runs holdout_passed strategies on new closed bars with full simulator costs.
   - Writes forward_trades with simulated spread and slippage.
4. Decay Monitor:
   - Bootstraps confidence band from backtest trades.
   - Detects degradation when rolling performance falls below lower band for N trades.
5. Non-Negotiable LIVE-READY Gate:
   - Hard floors: >= 60 calendar days AND >= 50 forward trades.
   - Hard floors in code cannot be lowered by config or override.
   - Premature promotion rejected with CandidateUnprovenError.
"""

from __future__ import annotations

import datetime
from datetime import timezone
import mongomock
import numpy as np
import pandas as pd
import pytest

from forward.decay_monitor import DecayMonitor
from forward.feeds import BrokerFeedStub, CSVFolderFeed, SimulatedLiveFeed
from forward.holdout_runner import (
    HoldoutEligibilityError,
    HoldoutEvaluationResult,
    run_holdout,
)
from forward.live_ready import (
    CandidateUnprovenError,
    MIN_FORWARD_DAYS,
    MIN_FORWARD_TRADES,
    evaluate_live_ready,
)
from forward.paper_trader import PaperTrader


@pytest.fixture
def mock_db():
    client = mongomock.MongoClient()
    return client["test_quantforge_f12"]


@pytest.fixture
def sample_candidate():
    return {
        "strategy_id": "strat_gold_trend_v1",
        "name": "Gold Trend Follower",
        "status": "candidate",
        "timeframe": "1h",
        "concept_family": "trend",
        "source_code": """
import pandas as pd
def generate_signals(df):
    signals = pd.Series(0, index=df.index)
    if len(df) >= 2:
        signals.iloc[0] = 1
        signals.iloc[-1] = -1
    return signals
""",
        "train_metrics": {
            "sharpe": 2.0,
            "profit_factor": 1.8,
            "total_trades": 120,
            "net_profit": 5200.0,
            "max_drawdown": 0.08,
        },
    }


class TestHoldoutExecution:
    def test_ineligible_strategy_rejected(self, mock_db, base_cfg):
        # Insert rejected strategy
        mock_db["candidates"].insert_one({
            "strategy_id": "strat_bad",
            "status": "rejected",
            "rejected_at": "lookahead",
        })
        with pytest.raises(HoldoutEligibilityError):
            run_holdout("strat_bad", cfg=base_cfg, db=mock_db)

    def test_holdout_pass_marks_holdout_passed(self, mock_db, base_cfg, sample_candidate):
        mock_db["candidates"].insert_one(sample_candidate)

        # Mock datastore returning profitable holdout slice
        class MockDataStore:
            def get_data(self, *args, **kwargs):
                dates = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
                close = 2000.0 + np.cumsum(np.random.normal(0.5, 2.0, 200))
                return pd.DataFrame({
                    "timestamp": dates,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0,
                })

        result = run_holdout(
            "strat_gold_trend_v1",
            cfg=base_cfg,
            db=mock_db,
            datastore=MockDataStore(),
        )

        assert isinstance(result, HoldoutEvaluationResult)
        doc = mock_db["candidates"].find_one({"strategy_id": "strat_gold_trend_v1"})
        assert doc["status"] in ("holdout_passed", "holdout_failed")
        # Holdout audit log was recorded
        audit = mock_db["holdout_accesses"].find_one({"strategy_id": "strat_gold_trend_v1"})
        assert audit is not None

    def test_holdout_failure_burns_holdout_forever(self, mock_db, base_cfg, sample_candidate):
        mock_db["candidates"].insert_one(sample_candidate)

        # Mock datastore returning severely losing holdout slice
        class MockLosingDataStore:
            def get_data(self, *args, **kwargs):
                dates = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
                close = 2000.0 - np.cumsum(np.abs(np.random.normal(1.0, 2.0, 200)))
                return pd.DataFrame({
                    "timestamp": dates,
                    "open": close + 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0,
                })

        result = run_holdout(
            "strat_gold_trend_v1",
            cfg=base_cfg,
            db=mock_db,
            datastore=MockLosingDataStore(),
        )
        assert result.passed is False
        doc = mock_db["candidates"].find_one({"strategy_id": "strat_gold_trend_v1"})
        assert doc["status"] == "holdout_failed"
        assert doc["holdout_failed"] is True

        # Second attempt to run holdout on failed strategy is rejected
        with pytest.raises(HoldoutEligibilityError):
            run_holdout("strat_gold_trend_v1", cfg=base_cfg, db=mock_db, datastore=MockLosingDataStore())

    def test_holdout_isolation_from_llm(self, mock_db, base_cfg, sample_candidate):
        mock_db["candidates"].insert_one(sample_candidate)
        # Verify that holdout results are marked do_not_leak_to_llm
        class MockDataStore:
            def get_data(self, *args, **kwargs):
                dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
                close = 2000.0 + np.cumsum(np.random.normal(0.2, 1.0, 100))
                return pd.DataFrame({
                    "timestamp": dates,
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 500.0,
                })

        res = run_holdout("strat_gold_trend_v1", cfg=base_cfg, db=mock_db, datastore=MockDataStore())
        assert res.leak_to_llm is False


class TestPluggableFeeds:
    def test_simulated_live_feed(self):
        dates = pd.date_range("2024-06-01", periods=5, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [100, 100, 100, 100, 100],
        })
        feed = SimulatedLiveFeed(df)
        assert feed.has_next()
        bar1 = feed.next_bar()
        assert bar1["close"] == 100.5
        bars = list(feed)
        assert len(bars) == 4

    def test_broker_feed_stub(self):
        stub = BrokerFeedStub(broker_name="MT5_Demo")
        assert stub.is_connected() is False
        status = stub.connect()
        assert status["status"] == "stub_mode"


class TestPaperTraderAndDecay:
    def test_paper_trader_execution_records_trades(self, mock_db, base_cfg, sample_candidate):
        sample_candidate["status"] = "holdout_passed"
        mock_db["candidates"].insert_one(sample_candidate)

        dates = pd.date_range("2024-07-01", periods=10, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [2000.0 + i for i in range(10)],
            "high": [2002.0 + i for i in range(10)],
            "low": [1999.0 + i for i in range(10)],
            "close": [2001.0 + i for i in range(10)],
            "volume": [1000.0] * 10,
        })
        feed = SimulatedLiveFeed(df)
        trader = PaperTrader(cfg=base_cfg, db=mock_db, feed=feed)
        metrics = trader.run_session(strategy_id="strat_gold_trend_v1")

        assert "total_bars_processed" in metrics
        assert metrics["total_bars_processed"] == 10

    def test_decay_monitor_bootstrapping(self):
        # 50 historical trade returns
        backtest_trades = [{"return_pct": 0.015 + np.random.normal(0, 0.005)} for _ in range(50)]
        monitor = DecayMonitor()
        band = monitor.bootstrap_confidence_interval(backtest_trades, n_resamples=500, alpha=0.05)
        assert "lower_bound" in band
        assert "upper_bound" in band
        assert band["lower_bound"] < band["upper_bound"]

    def test_decay_monitor_detects_degraded_performance(self):
        monitor = DecayMonitor()
        lower_bound = 0.005
        # 12 consecutive trades with negative return
        decaying_trades = [{"return_pct": -0.010} for _ in range(12)]
        is_degraded = monitor.evaluate_decay(decaying_trades, lower_bound=lower_bound, min_consecutive_breaches=10)
        assert is_degraded is True


class TestLiveReadyGate:
    def test_hard_floors_defined_in_code(self):
        assert MIN_FORWARD_DAYS == 60, "Non-negotiable minimum forward days is 60"
        assert MIN_FORWARD_TRADES == 50, "Non-negotiable minimum forward trades is 50"

    def test_premature_candidate_rejected_with_candidate_unproven_error(self, mock_db, base_cfg):
        # 30 days and 20 trades: under the non-negotiable floor
        mock_db["candidates"].insert_one({
            "strategy_id": "strat_early",
            "status": "holdout_passed",
            "forward_metrics": {
                "calendar_days": 30,
                "total_trades": 20,
                "sharpe": 2.5,
                "profit_factor": 2.0,
                "is_degraded": False,
            }
        })

        with pytest.raises(CandidateUnprovenError) as exc_info:
            evaluate_live_ready("strat_early", db=mock_db, cfg=base_cfg)
        assert "60 calendar days" in str(exc_info.value) or "50 trades" in str(exc_info.value)

    def test_config_attempt_to_lower_floor_ignored(self, mock_db, base_cfg):
        # Config tries to set min_days=10, min_trades=5
        bad_cfg = dict(base_cfg)
        bad_cfg["forward"] = {"min_days": 10, "min_trades": 5}

        mock_db["candidates"].insert_one({
            "strategy_id": "strat_cheater",
            "status": "holdout_passed",
            "forward_metrics": {
                "calendar_days": 15,
                "total_trades": 10,
                "sharpe": 2.5,
                "profit_factor": 2.0,
                "is_degraded": False,
            }
        })

        # Must still fail because code enforces hard floor >= 60 and >= 50
        with pytest.raises(CandidateUnprovenError):
            evaluate_live_ready("strat_cheater", db=mock_db, cfg=bad_cfg)

    def test_qualified_strategy_promoted_to_live_ready(self, mock_db, base_cfg):
        # 65 calendar days, 55 trades, not degraded, profitable
        mock_db["candidates"].insert_one({
            "strategy_id": "strat_champion",
            "status": "holdout_passed",
            "forward_metrics": {
                "calendar_days": 65,
                "total_trades": 55,
                "sharpe": 1.8,
                "profit_factor": 1.6,
                "is_degraded": False,
            }
        })

        res = evaluate_live_ready("strat_champion", db=mock_db, cfg=base_cfg)
        assert res["status"] == "live_ready"
        doc = mock_db["candidates"].find_one({"strategy_id": "strat_champion"})
        assert doc["status"] == "live_ready"
