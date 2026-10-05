"""
forward/ – Phase F12: Holdout execution, paper trading, decay monitoring, and live-ready verification.
"""

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

__all__ = [
    "DecayMonitor",
    "BrokerFeedStub",
    "CSVFolderFeed",
    "SimulatedLiveFeed",
    "HoldoutEligibilityError",
    "HoldoutEvaluationResult",
    "run_holdout",
    "CandidateUnprovenError",
    "MIN_FORWARD_DAYS",
    "MIN_FORWARD_TRADES",
    "evaluate_live_ready",
    "PaperTrader",
]
