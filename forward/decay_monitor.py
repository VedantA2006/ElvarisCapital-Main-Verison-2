"""
forward/decay_monitor.py – Strategy performance decay monitor using bootstrapped confidence bands.

Closes: OPS-2 (decay monitor part)
Methodology:
1. Bootstraps trade return distributions from the strategy's backtest trades to construct an expected confidence band.
2. Compares rolling forward trade results against the lower confidence bound.
3. If rolling performance falls below the lower band for N consecutive trades, flags strategy as degraded.
"""

from __future__ import annotations

from typing import Any
import numpy as np


class DecayMonitor:
    """Detects post-deployment statistical decay in forward-trading strategies."""

    def bootstrap_confidence_interval(
        self,
        backtest_trades: list[dict[str, Any]],
        n_resamples: int = 1000,
        alpha: float = 0.05,
    ) -> dict[str, float]:
        """Bootstrap the expected mean trade return distribution from backtest trades."""
        if not backtest_trades:
            return {"lower_bound": 0.0, "upper_bound": 0.0, "mean": 0.0}

        returns = [
            float(t.get("return_pct", t.get("pnl", 0.0)))
            for t in backtest_trades
        ]
        arr = np.array(returns)
        n = len(arr)

        bootstrapped_means = np.empty(n_resamples)
        rng = np.random.default_rng(42)
        for i in range(n_resamples):
            sample = rng.choice(arr, size=n, replace=True)
            bootstrapped_means[i] = np.mean(sample)

        lower_bound = float(np.percentile(bootstrapped_means, alpha * 100))
        upper_bound = float(np.percentile(bootstrapped_means, (1.0 - alpha) * 100))
        mean_val = float(np.mean(bootstrapped_means))

        return {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "mean": mean_val,
        }

    def evaluate_decay(
        self,
        forward_trades: list[dict[str, Any]],
        lower_bound: float,
        min_consecutive_breaches: int = 10,
        window_size: int = 5,
    ) -> bool:
        """Check if rolling performance breaches lower bound for min_consecutive_breaches trades."""
        if len(forward_trades) < min_consecutive_breaches:
            return False

        returns = [
            float(t.get("return_pct", t.get("pnl", 0.0)))
            for t in forward_trades
        ]

        breach_count = 0
        for i in range(len(returns)):
            start = max(0, i - window_size + 1)
            window = returns[start: i + 1]
            rolling_mean = np.mean(window)
            if rolling_mean < lower_bound:
                breach_count += 1
                if breach_count >= min_consecutive_breaches:
                    return True
            else:
                breach_count = 0

        return False
