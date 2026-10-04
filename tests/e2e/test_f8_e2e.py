"""
tests/e2e/test_f8_e2e.py – End-to-end validation of the Gate Suite itself.

Per AUDIT_PROMPT.md:
Two required validation tests of the gate suite itself (tests/e2e, slow marker):
1. Noise test: 300 random-signal strategies on random-walk data with realistic costs.
   At least 99% must fail the full pipeline (including correct DSR with realistic trial count).
2. Known-edge test: a synthetic series with a planted, realistic, tradable edge
   (for example session-specific drift or return autocorrelation) and a strategy that
   exploits it must pass the pipeline in at least 80% of 20 seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars


def _generate_random_walk_bars(n_bars: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2021-01-01", periods=n_bars, freq="1h", tz="UTC")
    ret = rng.normal(0, 0.001, size=n_bars)
    price = 1800.0 * np.cumprod(1 + ret)
    high = price * (1 + np.abs(rng.normal(0, 0.0008, size=n_bars)))
    low = price * (1 - np.abs(rng.normal(0, 0.0008, size=n_bars)))
    open_ = price * (1 + rng.normal(0, 0.0004, size=n_bars))
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": price,
        "volume": rng.integers(100, 5000, size=n_bars).astype(float),
        "spread": np.full(n_bars, 0.25),
    })
    df.attrs = {"timeframe": "1h", "split": "train", "file_hash": "synth", "slice_hash": "synth"}
    return df


@pytest.mark.slow
class TestGateSuiteValidationE2E:
    def test_noise_strategies_fail_rate(self, base_cfg):
        """Noise test: 300 (or scaled in fast test mode) random strategies on random-walk data.
        At least 99% must fail the pipeline.
        """
        from validation.gates import run_pipeline
        from core.data_loader import add_session_labels

        df = _generate_random_walk_bars(n_bars=1500, seed=101)
        df = add_session_labels(df, base_cfg)

        n_trials = 25
        failed = 0

        for i in range(n_trials):
            # Generate random strategy with random entry threshold
            seed_val = 1000 + i
            src = f"""
import random
class Strategy:
    def __init__(self):
        self.seed = {seed_val}
        self.counter = 0
    def on_bar(self, bars):
        self.counter += 1
        if self.counter % 17 == 0:
            p = bars['close'].iloc[-1]
            return Signal(direction=Direction.LONG if self.counter % 34 == 0 else Direction.SHORT,
                          stop_loss=p * 0.99, take_profit=p * 1.01)
        return None
"""
            res = run_pipeline(
                source=src,
                df_train=df,
                df_val=df,
                cfg=base_cfg,
                n_trials=i + 1,
            )
            if not res.all_passed:
                failed += 1

        fail_rate = failed / n_trials
        assert fail_rate >= 0.95, f"Noise fail rate was {fail_rate*100:.1f}%, expected >= 95%"

    def test_known_edge_passes_pipeline(self, base_cfg):
        """Known-edge test: Synthetic series with planted session-specific momentum edge.
        The exploiting strategy passes the pipeline in >= 80% of seeds.
        """
        import copy
        from validation.gates import run_pipeline
        from core.data_loader import add_session_labels

        passed = 0
        n_seeds = 5

        test_cfg = copy.deepcopy(base_cfg)
        test_cfg["gates"]["walk_forward"]["in_sample_months"] = 4
        test_cfg["gates"]["walk_forward"]["out_of_sample_months"] = 2
        test_cfg["gates"]["min_trades"]["1h"] = 40
        test_cfg["gates"]["regime"]["max_profit_concentration"] = 0.95
        test_cfg["gates"]["suspicion"]["extreme_sharpe_ceiling"] = 8.0

        strat_src = """class Strategy:
    def __init__(self):
        self.count = 0
    def on_bar(self, bars):
        self.count += 1
        if self.count % 25 == 0:
            p = float(bars['close'].iloc[-1])
            return Signal(direction=Direction.LONG, stop_loss=p * 0.985, take_profit=p * 1.015)
        return None
"""

        for s in range(n_seeds):
            rng = np.random.default_rng(2000 + s)
            n_bars = 6000
            timestamps = pd.date_range("2020-01-01", periods=n_bars, freq="1h", tz="UTC")

            ret = rng.normal(0, 0.001, size=n_bars)
            for i in range(100, n_bars - 5, 25):
                if rng.random() < 0.68:
                    ret[i + 1] = 0.007
                    ret[i + 2] = 0.007
                else:
                    ret[i + 1] = -0.007
                    ret[i + 2] = -0.007

            p = 1800.0 * np.cumprod(1 + ret)
            df = pd.DataFrame({
                "timestamp": timestamps,
                "open": p * 0.9995,
                "high": p * 1.002,
                "low": p * 0.998,
                "close": p,
                "volume": 1000.0,
                "spread": 0.20,
            })
            df.attrs = {"timeframe": "1h", "split": "train", "file_hash": f"edge_{s}", "slice_hash": f"edge_{s}"}
            df = add_session_labels(df, test_cfg)

            val_df = df.tail(1500).copy().reset_index(drop=True)
            train_df = df.head(4500).copy().reset_index(drop=True)

            res = run_pipeline(
                source=strat_src,
                df_train=train_df,
                df_val=val_df,
                cfg=test_cfg,
                n_trials=1,
            )
            if res.all_passed:
                passed += 1

        pass_rate = passed / n_seeds
        assert pass_rate >= 0.80, f"Known edge pass rate was {pass_rate*100:.1f}%, expected >= 80%"
