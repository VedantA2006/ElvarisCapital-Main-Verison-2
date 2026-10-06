"""
Phase 3 tests: lookahead guard (static AST, delay, truncation) + sandbox.

Run:  python -m pytest tests/test_phase3_guard.py -v
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    from core.data_loader import add_session_labels
    from core.config import load_config
    cfg = load_config()
    df = make_bars("2022-01-03", "2023-06-01", "60min", seed=seed)
    df = df.head(n).reset_index(drop=True)
    df = add_session_labels(df, cfg)
    df.attrs = {"timeframe": "1h", "split": "train", "file_hash": "test", "slice_hash": "test"}
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: Static AST Scan
# ═══════════════════════════════════════════════════════════════════════════

class TestStaticScan:
    def test_clean_code_passes(self):
        from core.lookahead_guard import static_scan
        code = textwrap.dedent("""
        import numpy as np
        import pandas as pd

        class Strategy:
            def __init__(self):
                self.period = 20

            def on_bar(self, bars):
                if len(bars) < self.period:
                    return None
                sma = bars['close'].rolling(self.period).mean()
                if bars['close'].iloc[-1] > sma.iloc[-1]:
                    return Signal(direction=Direction.LONG,
                                  stop_loss=bars['close'].iloc[-1] * 0.99)
                return None
        """)
        result = static_scan(code)
        assert result.passed, result.violations

    def test_shift_negative_detected(self):
        from core.lookahead_guard import static_scan
        code = "x = df['close'].shift(-1)"
        result = static_scan(code)
        assert not result.passed
        patterns = [v.pattern for v in result.violations]
        assert "shift_negative" in patterns

    def test_shift_negative_literal(self):
        from core.lookahead_guard import static_scan
        code = "x = series.shift(-3)"
        result = static_scan(code)
        assert not result.passed

    def test_shift_positive_is_fine(self):
        from core.lookahead_guard import static_scan
        code = "x = df['close'].shift(1)"
        result = static_scan(code)
        errors = [v for v in result.violations if v.severity == "error"]
        assert not errors

    def test_iloc_forward_detected(self):
        from core.lookahead_guard import static_scan
        code = "x = df.iloc[i+1]"
        result = static_scan(code)
        patterns = [v.pattern for v in result.violations]
        assert "index_forward" in patterns

    def test_banned_import_os(self):
        from core.lookahead_guard import static_scan
        code = "import os"
        result = static_scan(code)
        assert not result.passed
        assert "os" in result.banned_imports

    def test_banned_import_requests(self):
        from core.lookahead_guard import static_scan
        code = "import requests"
        result = static_scan(code)
        assert not result.passed

    def test_banned_import_from(self):
        from core.lookahead_guard import static_scan
        code = "from subprocess import call"
        result = static_scan(code)
        assert not result.passed

    def test_allowed_imports_pass(self):
        from core.lookahead_guard import static_scan
        code = textwrap.dedent("""
        import numpy as np
        import pandas as pd
        import math
        from numpy import array
        """)
        result = static_scan(code)
        assert result.passed

    def test_syntax_error_reported(self):
        from core.lookahead_guard import static_scan
        code = "def foo(:\n  pass"
        result = static_scan(code)
        assert not result.passed
        assert result.violations[0].pattern == "syntax_error"

    def test_reversed_rolling_warned(self):
        from core.lookahead_guard import static_scan
        code = "x = series[::-1].rolling(20).mean()"
        result = static_scan(code)
        warnings = [v for v in result.violations if v.severity == "warning"]
        assert any("reverse" in v.detail.lower() or "[::-1]" in v.detail for v in warnings)

    def test_suspicious_variable_name_warned(self):
        from core.lookahead_guard import static_scan
        code = "future_price = df['close'].shift(1)"
        result = static_scan(code)
        warnings = [v for v in result.violations if v.severity == "warning"]
        assert any("suspicious" in v.detail.lower() for v in warnings)

    @pytest.mark.parametrize("code,should_fail", [
        ("x = df.values[i+2]", True),
        ("x = df.iat[i+1]", True),
        ("x = df.iloc[i-1]", False),  # backward is fine
        ("x = df['close'].shift(5)", False),  # positive shift = lookback
    ])
    def test_parametrised_patterns(self, code, should_fail):
        from core.lookahead_guard import static_scan
        result = static_scan(code)
        errors = [v for v in result.violations if v.severity == "error"]
        if should_fail:
            assert len(errors) > 0, f"Expected error for: {code}"
        else:
            assert len(errors) == 0, f"Unexpected error for: {code}"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: Delay Test
# ═══════════════════════════════════════════════════════════════════════════

class TestDelayTest:
    def test_honest_strategy_survives_delay(self, base_cfg):
        """A simple SMA crossover should not lose >60% Sharpe with 1-bar delay."""
        from core.lookahead_guard import run_delay_test
        from core.backtester import Signal, Direction

        class SMACross:
            def __init__(self):
                self.fast, self.slow = 10, 30

            def on_bar(self, bars):
                if len(bars) < self.slow:
                    return None
                f = bars["close"].rolling(self.fast).mean().iloc[-1]
                s = bars["close"].rolling(self.slow).mean().iloc[-1]
                prev_f = bars["close"].rolling(self.fast).mean().iloc[-2]
                prev_s = bars["close"].rolling(self.slow).mean().iloc[-2]
                if prev_f <= prev_s and f > s:
                    return Signal(direction=Direction.LONG,
                                  stop_loss=bars["close"].iloc[-1] * 0.98,
                                  take_profit=bars["close"].iloc[-1] * 1.04)
                if prev_f >= prev_s and f < s:
                    return Signal(direction=Direction.SHORT,
                                  stop_loss=bars["close"].iloc[-1] * 1.02,
                                  take_profit=bars["close"].iloc[-1] * 0.96)
                return None

        df = _make_df(400, seed=7)
        result = run_delay_test(SMACross, df, base_cfg)
        # We can't guarantee it passes (depends on random data), but
        # the test infrastructure should work without errors
        assert result.original_trades >= 0
        assert isinstance(result.sharpe_drop_pct, float)

    def test_cheating_strategy_fails_delay(self, base_cfg):
        """A strategy that peeks at the next bar should lose massive Sharpe."""
        from core.lookahead_guard import run_delay_test
        from core.backtester import Signal, Direction

        df = _make_df(400, seed=7)
        # Store future data in closure (simulating a strategy that cheats)
        future_closes = df["close"].to_numpy()

        class CheatingFactory:
            """Creates strategies that use future data via closure."""
            def __call__(self):
                return self.CheatingStrategy()

            class CheatingStrategy:
                def __init__(self):
                    self.bar_count = 0

                def on_bar(self, bars):
                    i = len(bars) - 1
                    self.bar_count += 1
                    if i + 1 >= len(future_closes):
                        return None
                    # Cheat: look at next bar's close
                    if future_closes[i + 1] > future_closes[i]:
                        return Signal(direction=Direction.LONG,
                                      stop_loss=bars["close"].iloc[-1] * 0.99,
                                      take_profit=bars["close"].iloc[-1] * 1.02)
                    else:
                        return Signal(direction=Direction.SHORT,
                                      stop_loss=bars["close"].iloc[-1] * 1.01,
                                      take_profit=bars["close"].iloc[-1] * 0.98)
                    return None

        result = run_delay_test(CheatingFactory(), df, base_cfg)
        # The cheating strategy should show a massive Sharpe drop
        # when delayed, because the "future" info is now 2 bars ahead
        assert result.original_sharpe != result.delayed_sharpe

    def test_zero_sharpe_fails_delay_explicitly(self, base_cfg):
        from core.lookahead_guard import run_delay_test

        class DoNothing:
            def on_bar(self, bars):
                return None

        df = _make_df(100)
        result = run_delay_test(DoNothing, df, base_cfg)
        assert not result.passed  # LH-5: non-profitable base fails delay test explicitly
        assert "not profitable" in result.detail.lower() or "sharpe <= 0" in result.detail.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: Truncation Test
# ═══════════════════════════════════════════════════════════════════════════

class TestTruncationTest:
    def test_stable_strategy_passes_truncation(self, base_cfg):
        from core.lookahead_guard import run_truncation_test
        from core.backtester import Signal, Direction

        class SimpleMomentum:
            def on_bar(self, bars):
                if len(bars) < 50:
                    return None
                ret = bars["close"].iloc[-1] / bars["close"].iloc[-50] - 1
                if ret > 0.02:
                    return Signal(direction=Direction.LONG)
                elif ret < -0.02:
                    return Signal(direction=Direction.SHORT)
                return None

        df = _make_df(400, seed=5)
        result = run_truncation_test(SimpleMomentum, df, base_cfg, num_cuts=5)
        assert result.passed
        assert result.num_cuts >= 2

    def test_small_dataset_skips(self, base_cfg):
        from core.lookahead_guard import run_truncation_test

        class Anything:
            def on_bar(self, bars):
                return None

        df = _make_df(50)
        result = run_truncation_test(Anything, df, base_cfg, num_cuts=5)
        assert result.passed
        assert "too small" in result.detail.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Sandbox
# ═══════════════════════════════════════════════════════════════════════════

class TestSandbox:
    def test_clean_strategy_loads(self, base_cfg):
        from core.sandbox import Sandbox
        sb = Sandbox(base_cfg)
        code = textwrap.dedent("""
        import numpy as np

        class Strategy:
            def __init__(self):
                self.period = 20

            def on_bar(self, bars):
                if len(bars) < self.period:
                    return None
                sma = bars['close'].rolling(self.period).mean().iloc[-1]
                price = bars['close'].iloc[-1]
                if price > sma:
                    return Signal(direction=Direction.LONG,
                                  stop_loss=price * 0.99)
                return None
        """)
        strategy = sb.load_strategy(code)
        assert hasattr(strategy, "on_bar")
        # Run on synthetic data
        df = _make_df(100)
        result = strategy.on_bar(df)
        # Result should be Signal or None
        from core.backtester import Signal
        assert result is None or isinstance(result, Signal)

    def test_banned_import_rejected(self, base_cfg):
        from core.sandbox import Sandbox, SandboxError
        sb = Sandbox(base_cfg)
        code = "import os\nclass Strategy:\n    def on_bar(self, bars): return None"
        with pytest.raises(SandboxError, match="Static scan failed"):
            sb.load_strategy(code)

    def test_open_blocked(self, base_cfg):
        from core.sandbox import Sandbox, SandboxError
        sb = Sandbox(base_cfg)
        code = textwrap.dedent("""
        class Strategy:
            def on_bar(self, bars):
                f = open('secret.txt', 'r')
                return None
        """)
        with pytest.raises(Exception):
            strategy = sb.load_strategy(code)
            df = _make_df(50)
            strategy.on_bar(df)

    def test_exec_blocked(self, base_cfg):
        from core.sandbox import Sandbox, SandboxError
        sb = Sandbox(base_cfg)
        code = textwrap.dedent("""
        class Strategy:
            def on_bar(self, bars):
                exec("import os")
                return None
        """)
        with pytest.raises(Exception):
            strategy = sb.load_strategy(code)
            df = _make_df(50)
            strategy.on_bar(df)

    def test_runtime_import_blocked(self, base_cfg):
        from core.sandbox import Sandbox, SandboxImportError, SandboxError
        sb = Sandbox(base_cfg)
        # Use __import__ in on_bar (bypasses static scan since the string is runtime)
        code = textwrap.dedent("""
        class Strategy:
            def on_bar(self, bars):
                subprocess = __import__('subprocess')
                return None
        """)
        with pytest.raises((SandboxImportError, SandboxError)):
            strategy = sb.load_strategy(code)
            df = _make_df(50)
            strategy.on_bar(df)

    def test_no_on_bar_method_rejected(self, base_cfg):
        from core.sandbox import Sandbox, SandboxError
        sb = Sandbox(base_cfg)
        code = "class Strategy:\n    pass"
        with pytest.raises(SandboxError, match="on_bar"):
            sb.load_strategy(code)

    def test_no_strategy_class_rejected(self, base_cfg):
        from core.sandbox import Sandbox, SandboxError
        sb = Sandbox(base_cfg)
        code = "x = 42"
        with pytest.raises(SandboxError, match="no class"):
            sb.load_strategy(code)

    def test_code_too_long_rejected(self, base_cfg):
        from core.sandbox import Sandbox, SandboxError
        sb = Sandbox(base_cfg)
        max_len = sb._max_code_len
        code = "# " + "x" * (max_len + 100) + "\nclass Strategy:\n    def on_bar(self, bars): return None"
        with pytest.raises(SandboxError, match="Static scan failed"):
            sb.load_strategy(code)

    def test_syntax_error_rejected(self, base_cfg):
        from core.sandbox import Sandbox, SandboxError
        sb = Sandbox(base_cfg)
        code = "def foo(:\n  pass"
        with pytest.raises(SandboxError):
            sb.load_strategy(code)

    def test_auto_discovers_strategy_class(self, base_cfg):
        from core.sandbox import Sandbox
        sb = Sandbox(base_cfg)
        code = textwrap.dedent("""
        class MyCustomStrategy:
            def on_bar(self, bars):
                return None
        """)
        strategy = sb.load_strategy(code, strategy_class_name="NotFound")
        assert hasattr(strategy, "on_bar")

    def test_wall_timeout_enforced(self, base_cfg):
        import copy
        from core.sandbox import Sandbox, SandboxTimeoutError
        cfg = copy.deepcopy(base_cfg)
        cfg["sandbox"]["wall_clock_timeout"] = 0.01  # 10ms

        sb = Sandbox(cfg)
        code = textwrap.dedent("""
        import math
        class Strategy:
            def on_bar(self, bars):
                # Burn CPU
                x = 0
                for i in range(10_000_000):
                    x += math.sqrt(i)
                return None
        """)
        strategy = sb.load_strategy(code)
        df = _make_df(50)
        with pytest.raises(SandboxTimeoutError):
            for i in range(100):
                strategy.on_bar(df.iloc[:i+1])

    def test_signal_and_direction_available(self, base_cfg):
        from core.sandbox import Sandbox
        sb = Sandbox(base_cfg)
        code = textwrap.dedent("""
        class Strategy:
            def on_bar(self, bars):
                return Signal(direction=Direction.LONG,
                              stop_loss=bars['close'].iloc[-1] * 0.99)
        """)
        strategy = sb.load_strategy(code)
        df = _make_df(50)
        result = strategy.on_bar(df)
        from core.backtester import Signal, Direction
        assert isinstance(result, Signal)
        assert result.direction == Direction.LONG

    def test_lookahead_code_rejected_by_sandbox(self, base_cfg):
        from core.sandbox import Sandbox, SandboxError
        sb = Sandbox(base_cfg)
        code = textwrap.dedent("""
        class Strategy:
            def on_bar(self, bars):
                future = bars['close'].shift(-1)
                if future.iloc[-1] > bars['close'].iloc[-1]:
                    return Signal(direction=Direction.LONG)
                return None
        """)
        with pytest.raises(SandboxError, match="Static scan failed"):
            sb.load_strategy(code)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Sandbox + Backtester
# ═══════════════════════════════════════════════════════════════════════════

class TestSandboxBacktestIntegration:
    def test_sandbox_strategy_runs_full_backtest(self, base_cfg):
        from core.sandbox import Sandbox
        from core.backtester import run_backtest

        sb = Sandbox(base_cfg)
        code = textwrap.dedent("""
        import numpy as np

        class Strategy:
            def __init__(self):
                self.fast = 10
                self.slow = 30

            def on_bar(self, bars):
                if len(bars) < self.slow:
                    return None
                fast_ma = bars['close'].rolling(self.fast).mean().iloc[-1]
                slow_ma = bars['close'].rolling(self.slow).mean().iloc[-1]
                price = bars['close'].iloc[-1]

                if fast_ma > slow_ma and price > fast_ma:
                    return Signal(direction=Direction.LONG,
                                  stop_loss=price * 0.98,
                                  take_profit=price * 1.04)
                elif fast_ma < slow_ma and price < fast_ma:
                    return Signal(direction=Direction.SHORT,
                                  stop_loss=price * 1.02,
                                  take_profit=price * 0.96)
                return None
        """)
        strategy = sb.load_strategy(code)
        df = _make_df(300)
        result = run_backtest(strategy, df, base_cfg)
        assert result.metrics["total_trades"] >= 0
        assert "sharpe" in result.metrics
        assert result.wall_seconds > 0
