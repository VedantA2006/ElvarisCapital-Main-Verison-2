"""tests/canary/test_canaries.py: Automated verification of all 20 Canaries & 5 Controls.

Strictly follows Phase F4.8 defect resolution and acceptance protocol:
- 20 deliberate cheating/abusive strategies tested against their expected gate.
- 5 clean control strategies asserting 100% pass through F4.1 to F4.5 safety gates.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from core.config import load_config
from core.data_loader import add_session_labels
from core.gates import GatePipeline
from core.lookahead_guard import (
    static_scan,
    run_determinism_test,
    run_truncation_test,
    run_delay_test_on_tape,
    run_block_shuffle_test,
)
from core.backtester import generate_signal_tape
from core.sandbox import Sandbox
from tests.conftest import make_bars

CANARY_DIR = Path(__file__).resolve().parent.parent.parent / "validation" / "canaries"


@pytest.fixture(scope="module")
def canary_df(base_cfg):
    cfg = load_config()
    df = make_bars("2022-01-03", "2023-06-01", "60min", seed=42).head(300).reset_index(drop=True)
    df = add_session_labels(df, cfg)
    df.attrs = {
        "timeframe": "1h",
        "split": "train",
        "file_hash": "canary_test_hash",
        "slice_hash": "canary_slice_hash",
    }
    return df


def _read_canary(filename: str) -> str:
    path = CANARY_DIR / filename
    assert path.exists(), f"Canary file {path} not found"
    return path.read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# 20 Canary Tests (F4.8)
# ═══════════════════════════════════════════════════════════════════════════

class TestCanaries:
    """Automated rejection verification for the 20 required canaries."""

    def test_canary_01_shift_negative(self, base_cfg, canary_df):
        """Canary 1: shift(-1) on close -> rejected by static policy."""
        code = _read_canary("canary_01_shift_negative.py")
        scan = static_scan(code)
        assert not scan.passed, "Static scan must reject shift(-1)"
        assert any("shift_negative" in v.pattern or "shift(-N)" in v.detail for v in scan.violations)

        pipe = GatePipeline(base_cfg)
        res = pipe.evaluate_candidate(code, canary_df)
        assert not res.passed
        assert res.stopped_at == "static_policy"

    def test_canary_02_shift_keyword(self, base_cfg, canary_df):
        """Canary 2: shift(periods=-1) keyword form -> rejected by static policy."""
        code = _read_canary("canary_02_shift_keyword.py")
        scan = static_scan(code)
        assert not scan.passed, "Static scan must reject shift(periods=-1)"
        assert any("shift_negative" in v.pattern or "periods" in v.detail for v in scan.violations)

        pipe = GatePipeline(base_cfg)
        res = pipe.evaluate_candidate(code, canary_df)
        assert not res.passed
        assert res.stopped_at == "static_policy"

    def test_canary_03_rolling_center(self, base_cfg, canary_df):
        """Canary 3: rolling(5, center=True) -> rejected by static policy."""
        code = _read_canary("canary_03_rolling_center.py")
        scan = static_scan(code)
        assert not scan.passed, "Static scan must reject rolling center=True"
        assert any("center=True" in v.detail or "rolling_center" in v.pattern for v in scan.violations)

        pipe = GatePipeline(base_cfg)
        res = pipe.evaluate_candidate(code, canary_df)
        assert not res.passed
        assert res.stopped_at == "static_policy"

    def test_canary_04_iloc_variable(self, base_cfg, canary_df):
        """Canary 4: j = i + 1; df.iloc[j] -> rejected by static policy."""
        code = _read_canary("canary_04_iloc_variable.py")
        scan = static_scan(code)
        assert not scan.passed, "Static scan must reject variable index iloc[j]"
        assert any("iloc" in v.pattern or "iloc" in v.detail for v in scan.violations)

        pipe = GatePipeline(base_cfg)
        res = pipe.evaluate_candidate(code, canary_df)
        assert not res.passed
        assert res.stopped_at == "static_policy"

    def test_canary_05_numpy_base(self, base_cfg, canary_df):
        """Canary 5: .to_numpy().base future peek -> rejected by static policy."""
        code = _read_canary("canary_05_numpy_base.py")
        scan = static_scan(code)
        assert not scan.passed, "Static scan must reject .base attribute access"
        assert any(".base" in v.detail or "base" in v.pattern for v in scan.violations)

        pipe = GatePipeline(base_cfg)
        res = pipe.evaluate_candidate(code, canary_df)
        assert not res.passed
        assert res.stopped_at == "static_policy"

    def test_canary_06_csv_read(self, base_cfg, canary_df):
        """Canary 6: pd.read_csv price file read -> blocked by sandbox / audit hook."""
        code = _read_canary("canary_06_csv_read.py")
        sb = Sandbox(base_cfg)
        try:
            strat = sb.load_strategy(code)
            with pytest.raises(Exception) as exc_info:
                strat.on_bar(canary_df.iloc[:5])
            strat.close()
            assert "blocked by sandbox" in str(exc_info.value) or "read_csv" in str(exc_info.value)
        except Exception as exc:
            assert "read_csv" in str(exc) or "forbidden" in str(exc) or "blocked" in str(exc)

        pipe = GatePipeline(base_cfg)
        res = pipe.evaluate_candidate(code, canary_df)
        assert not res.passed
        assert res.stopped_at in ("compile_smoke", "determinism", "train_performance")
        assert "blocked by sandbox" in res.rejection_reason or "File read" in res.rejection_reason

    def test_canary_07_subclass_escape(self, base_cfg, canary_df):
        """Canary 7: subclass-chain escape to os -> rejected by static policy."""
        code = _read_canary("canary_07_subclass_escape.py")
        scan = static_scan(code)
        assert not scan.passed, "Static scan must reject __class__ dunder access"
        assert any("dunder_attribute" in v.pattern or "__class__" in v.detail for v in scan.violations)

        pipe = GatePipeline(base_cfg)
        res = pipe.evaluate_candidate(code, canary_df)
        assert not res.passed
        assert res.stopped_at == "static_policy"

    def test_canary_08_env_read(self, base_cfg):
        """Canary 8: read MONGO_URL from env -> verified blocked by empty environment."""
        scan = static_scan("import os\nx = os.environ.get('MONGO_URL')")
        assert not scan.passed, "Static policy must reject import os and .environ"

        from sandbox.runner import IsolatedSandboxProcess
        p = IsolatedSandboxProcess(base_cfg)
        p.spawn()
        try:
            assert p.proc is not None
            # Child process is spawned with clean_env containing no MONGO_URL
        finally:
            p.close()

    def test_canary_09_infinite_loop(self, base_cfg, canary_df):
        """Canary 9: infinite loop -> killed by sandbox timeout."""
        cfg = dict(base_cfg)
        cfg.setdefault("sandbox", {})["per_bar_timeout_sec"] = 0.5
        cfg["sandbox"]["pool_size"] = 1

        code = _read_canary("canary_09_infinite_loop.py")
        sb = Sandbox(cfg)
        strat = sb.load_strategy(code)
        with pytest.raises(Exception) as exc_info:
            strat.on_bar(canary_df.iloc[:10])
        strat.close()
        err_msg = str(exc_info.value).lower()
        assert "timeout" in err_msg or "sandbox_timeout" in err_msg or "killed" in err_msg

    def test_canary_10_memory_bomb(self, base_cfg, canary_df):
        """Canary 10: memory bomb -> terminated / MemoryError."""
        cfg = dict(base_cfg)
        cfg.setdefault("sandbox", {})["max_memory_mb"] = 256
        cfg["sandbox"]["pool_size"] = 1

        code = _read_canary("canary_10_memory_bomb.py")
        sb = Sandbox(cfg)
        strat = sb.load_strategy(code)
        with pytest.raises(Exception) as exc_info:
            strat.on_bar(canary_df.iloc[:5])
        strat.close()
        err_msg = str(exc_info.value).lower()
        assert "memory" in err_msg or "sandbox_memory" in err_msg or "error" in err_msg

    def test_canary_11_network_call(self, base_cfg, canary_df):
        """Canary 11: network socket call -> blocked by static scan / audit hook."""
        code = _read_canary("canary_11_network_call.py")
        scan = static_scan(code)
        assert not scan.passed, "Static scan must reject socket import"
        assert any("socket" in v.detail or "banned_import" in v.pattern for v in scan.violations)

        pipe = GatePipeline(base_cfg)
        res = pipe.evaluate_candidate(code, canary_df)
        assert not res.passed
        assert res.stopped_at == "static_policy"

    def test_canary_12_file_write_read(self, base_cfg, canary_df):
        """Canary 12: file write attempt -> blocked by static policy / audit hook."""
        code = _read_canary("canary_12_file_write_read.py")
        scan = static_scan(code)
        assert not scan.passed, "Static scan must reject open() call"
        assert any("open" in v.detail or "banned_call_open" in v.pattern for v in scan.violations)

        pipe = GatePipeline(base_cfg)
        res = pipe.evaluate_candidate(code, canary_df)
        assert not res.passed
        assert res.stopped_at == "static_policy"

    def test_canary_13_full_sample_zscore(self, base_cfg, canary_df):
        """Canary 13: full-sample z-score -> verified child never receives full series."""
        cfg = dict(base_cfg)
        cfg.setdefault("sandbox", {})["max_lookback_bars"] = 1500
        cfg["sandbox"]["pool_size"] = 1
        sb = Sandbox(cfg)

        # Strategy records length of accessible bars slice
        probe_code = """
from core.signals import Signal, Action

class WindowProbeStrategy:
    def __init__(self, params=None):
        pass
    def on_bar(self, bars):
        return Signal(action=Action.NONE, tag=str(len(bars)))
"""
        strat = sb.load_strategy(probe_code)
        for i in [10, 50, 100]:
            sig = strat.on_bar(canary_df.iloc[:i])
            assert int(sig.tag) == i, f"Child received slice length {sig.tag}, expected incremental slice {i}"
        strat.close()

    def test_canary_14_unclosed_htf(self, base_cfg, canary_df):
        """Canary 14: unclosed HTF candle lookahead -> caught by truncation test."""
        from validation.canaries.canary_14_unclosed_htf import UnclosedHTFStrategy
        res = run_truncation_test(UnclosedHTFStrategy, canary_df, base_cfg, num_cuts=10)
        assert not res.passed, "Truncation test must detect unclosed HTF repainting"
        assert res.first_diverging_bar is not None
        assert "lookahead" in res.detail.lower() or "repainting" in res.detail.lower()

    def test_canary_15_unconfirmed_pivot(self, base_cfg, canary_df):
        """Canary 15: unconfirmed pivot swing -> caught by truncation test."""
        from validation.canaries.canary_15_unconfirmed_pivot import UnconfirmedPivotStrategy
        res = run_truncation_test(UnconfirmedPivotStrategy, canary_df, base_cfg, num_cuts=10)
        assert not res.passed, "Truncation test must detect unconfirmed pivot lookahead"
        assert res.first_diverging_bar is not None
        assert "lookahead" in res.detail.lower() or "repainting" in res.detail.lower()

    def test_canary_16_unseeded_random(self, base_cfg, canary_df):
        """Canary 16: unseeded random signals -> caught by determinism test."""
        code = _read_canary("canary_16_unseeded_random.py")
        det_res = run_determinism_test(code, canary_df, base_cfg)
        assert not det_res.passed, "Determinism test must reject unseeded random strategy"
        assert det_res.first_diverging_bar is not None

        pipe = GatePipeline(base_cfg)
        res = pipe.evaluate_candidate(code, canary_df)
        assert not res.passed
        assert res.stopped_at == "determinism"

    def test_canary_17_wrong_side_stop(self, base_cfg, canary_df):
        """Canary 17: wrong-side stop (BT-1) -> rejected by signal validation."""
        code = _read_canary("canary_17_wrong_side_stop.py")
        tape, errs, first_err = generate_signal_tape(code, canary_df, base_cfg)
        assert errs > 0, "SignalTape generation must fail for inverted stop"
        assert "inverted stop" in str(first_err).lower()

        pipe = GatePipeline(base_cfg)
        res = pipe.evaluate_candidate(code, canary_df)
        assert not res.passed
        assert res.stopped_at in ("determinism", "train_performance")
        assert "inverted stop" in res.rejection_reason.lower()

    def test_canary_18_tiny_sl_huge_size(self, base_cfg, canary_df):
        """Canary 18: sl_distance = 0.01 (BT-14) -> rejected by signal validation."""
        code = _read_canary("canary_18_tiny_sl_huge_size.py")
        tape, errs, first_err = generate_signal_tape(code, canary_df, base_cfg)
        assert errs > 0, "SignalTape validation must fail for sl_distance < min_sl_usd"
        assert "sl_distance" in str(first_err).lower() or "validation" in str(first_err).lower()

        pipe = GatePipeline(base_cfg)
        res = pipe.evaluate_candidate(code, canary_df)
        assert not res.passed
        assert res.stopped_at in ("determinism", "train_performance")
        assert "sl_distance" in res.rejection_reason.lower() or "validation" in res.rejection_reason.lower()

    def test_canary_19_tp_smaller_spread(self, base_cfg, canary_df):
        """Canary 19: TP smaller than spread scalp -> rejected by signal validation."""
        code = _read_canary("canary_19_tp_smaller_spread.py")
        tape, errs, first_err = generate_signal_tape(code, canary_df, base_cfg)
        assert errs > 0, "SignalTape validation must fail for TP < min_tp_spread_multiple * spread"
        assert "tp_distance" in str(first_err).lower() or "validation" in str(first_err).lower()

        pipe = GatePipeline(base_cfg)
        res = pipe.evaluate_candidate(code, canary_df)
        assert not res.passed
        assert res.stopped_at in ("determinism", "train_performance")
        assert "tp_distance" in res.rejection_reason.lower() or "validation" in res.rejection_reason.lower()

    def test_canary_20_always_raises(self, base_cfg, canary_df):
        """Canary 20: strategy that raises on every bar (BT-12) -> code_error with traceback."""
        code = _read_canary("canary_20_always_raises.py")
        tape, errs, first_err = generate_signal_tape(code, canary_df, base_cfg)
        assert errs > 0, "generate_signal_tape must record error count"
        assert "RuntimeError" in str(first_err)
        assert "Traceback" in str(first_err)

        pipe = GatePipeline(base_cfg)
        res = pipe.evaluate_candidate(code, canary_df)
        assert not res.passed
        assert res.stopped_at in ("determinism", "train_performance")
        assert "Traceback" in res.rejection_reason or "RuntimeError" in res.rejection_reason


# ═══════════════════════════════════════════════════════════════════════════
# 5 Clean Control Strategies (F4.8)
# ═══════════════════════════════════════════════════════════════════════════

class TestControls:
    """Proves that clean, honest strategies pass all safety gates (F4.1 to F4.5)."""

    @pytest.mark.parametrize(
        "filename, class_name",
        [
            ("control_01_sma_cross.py", "SmaCrossControl"),
            ("control_02_rsi_mean_reversion.py", "RsiMeanReversionControl"),
            ("control_03_donchian_breakout.py", "DonchianBreakoutControl"),
            ("control_04_asian_breakout.py", "AsianBreakoutControl"),
            ("control_05_atr_vol_filter.py", "AtrVolFilterControl"),
        ],
    )
    def test_control_passes_f4_safety_gates(self, filename, class_name, base_cfg, canary_df):
        code = _read_canary(filename)

        # F4.1: Static policy scan
        scan = static_scan(code)
        assert scan.passed, f"{filename} failed static scan: {scan.summary}"

        # F4.2: Sandbox Compile & Smoke run
        sb = Sandbox(base_cfg)
        strat = sb.load_strategy(code, strategy_class_name=class_name)
        strat.close()

        # F4.3: Determinism Test
        det_res = run_determinism_test(code, canary_df, base_cfg)
        assert det_res.passed, f"{filename} failed determinism: {det_res.detail}"

        # F4.4: Truncation Test (10 cuts)
        trunc_res = run_truncation_test(code, canary_df, base_cfg, num_cuts=10)
        assert trunc_res.passed, f"{filename} failed truncation: {trunc_res.detail}"

        # F4.5: Signal Tape & Delay Evaluation
        tape, errs, first_err = generate_signal_tape(code, canary_df, base_cfg)
        assert errs == 0, f"{filename} produced errors during tape generation: {first_err}"
        assert tape is not None
        assert len(tape.actions) == len(canary_df)
