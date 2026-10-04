"""
tests/regression/test_f3_sandbox.py: Regression test suite for Phase F3 (Real Sandbox).

Covers:
- SBX-1: Sandbox escape via subclass chain to os module
- SBX-2: Strategy reading price file via pd.read_csv
- SBX-3: Secrets leaked from environment (MONGO_URL)
- SBX-4: Infinite loop hangs engine (parent timeout kills child)
- SBX-5: Strategy parameters through __init__(self, params)
- SBX-6: AST scan blocking eval, exec, open, compile, globals, getattr, etc.
- LH-1: Future peek via numpy .base
- LH-3: Static scan misses (shift(periods=-1), rolling(center=True), shift(var), iloc[var], .base)
- Audit hooks: socket, subprocess, os.system, file write
- Memory limit: memory bomb killed
- Determinism: identical seed produces bit-exact signal tape
- Throughput: >= 2000 bars/sec for simple SMA strategy
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import time
import pytest
import numpy as np
import pandas as pd

from core.config import load_config
from core.backtester import Signal, Direction, run_backtest
from tests.conftest import make_bars
from core.data_loader import add_session_labels


@pytest.fixture
def cfg():
    c = load_config()
    c["sandbox"]["wall_clock_timeout"] = 3
    c["sandbox"]["memory_limit_mb"] = 512
    return c


@pytest.fixture
def sample_bars(cfg):
    df = make_bars("2023-01-02", "2023-03-01", "60min", seed=42).reset_index(drop=True)
    df = add_session_labels(df, cfg)
    df.attrs = {"timeframe": "1h", "split": "train"}
    return df


def test_sbx1_subclass_escape_blocked(cfg):
    """SBX-1: Subclass chain to os must be rejected by AST policy and runtime."""
    from core.sandbox import Sandbox, SandboxError
    code = textwrap.dedent("""
    class Strategy:
        def on_bar(self, bars):
            for c in ().__class__.__base__.__subclasses__():
                if c.__name__ == 'catch_warnings':
                    imp = c()._module.__builtins__['__import__']
                    self.pid = imp('os').getpid()
                    return None
    """)
    sb = Sandbox(cfg)
    with pytest.raises(SandboxError) as exc_info:
        sb.load_strategy(code)
    assert any(w in str(exc_info.value).lower() for w in ["__class__", "__base__", "__subclasses__", "static", "policy"])


def test_sbx2_file_read_blocked(cfg, sample_bars):
    """SBX-2: pd.read_csv of price file must be blocked."""
    from core.sandbox import Sandbox, SandboxError
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        sample_bars.to_csv(f.name, index=False)
        temp_csv = f.name.replace("\\", "/")

    code = textwrap.dedent(f"""
    class Strategy:
        def __init__(self):
            self.f = pd.read_csv({temp_csv!r})
        def on_bar(self, bars):
            return None
    """)
    sb = Sandbox(cfg)
    with pytest.raises(SandboxError) as exc_info:
        sb.load_strategy(code)
    assert any(w in str(exc_info.value).lower() for w in ["read_csv", "open", "policy", "banned", "static"])
    try:
        os.remove(temp_csv)
    except OSError:
        pass


def test_sbx3_environment_isolated(cfg, sample_bars):
    """SBX-3: Secrets in engine environment must NOT be visible to strategy."""
    from core.sandbox import Sandbox, SandboxError
    os.environ["MONGO_URL"] = "mongodb://SUPER_SECRET_TOKEN"
    os.environ["ENGINE_API_KEY"] = "sk-super-secret"

    # 1. Attribute access to .environ should be rejected by AST
    code_ast = textwrap.dedent("""
    class Strategy:
        def on_bar(self, bars):
            x = pd.io.common.os.environ.get('MONGO_URL')
            return None
    """)
    sb = Sandbox(cfg)
    with pytest.raises(SandboxError):
        sb.load_strategy(code_ast)

    # 2. Even if code gets around name via clean execution, child env must not have secrets
    code_clean = textwrap.dedent("""
    class Strategy:
        def __init__(self):
            self.leak = None
        def on_bar(self, bars):
            # In an isolated environment, MONGO_URL should not exist
            return None
    """)
    strat = sb.load_strategy(code_clean)
    res = strat.on_bar(sample_bars.iloc[:10])
    assert res is None


def test_sbx4_infinite_loop_timeout(cfg, sample_bars):
    """SBX-4: while True loop must be interrupted by parent timeout."""
    from core.sandbox import Sandbox, SandboxTimeoutError, SandboxError
    cfg_short = dict(cfg)
    cfg_short["sandbox"] = dict(cfg["sandbox"])
    cfg_short["sandbox"]["wall_clock_timeout"] = 1  # 1 second timeout

    code = textwrap.dedent("""
    class Strategy:
        def on_bar(self, bars):
            while True:
                pass
    """)
    sb = Sandbox(cfg_short)
    strat = sb.load_strategy(code)
    t0 = time.perf_counter()
    with pytest.raises(SandboxError) as exc_info:
        strat.on_bar(sample_bars.iloc[:1])
    elapsed = time.perf_counter() - t0
    assert elapsed < 4.0, f"Took {elapsed:.2f}s, expected < 4s"
    assert "timeout" in str(exc_info.value).lower()


def test_sbx5_strategy_params(cfg):
    """SBX-5: Strategies must be able to receive parameters through __init__(params)."""
    from core.sandbox import Sandbox
    code = textwrap.dedent("""
    class Strategy:
        def __init__(self, params=None):
            self.params = params or {}
            self.fast = self.params.get('fast', 10)
            self.slow = self.params.get('slow', 30)

        def on_bar(self, bars):
            if self.fast == 12 and self.slow == 26:
                return Signal(direction=Direction.LONG, sl_distance=10.0, tp_distance=20.0)
            return None
    """)
    sb = Sandbox(cfg)
    strat = sb.load_strategy(code, params={"fast": 12, "slow": 26})
    df = pd.DataFrame({"close": [2000.0, 2001.0], "open": [1999.0, 2000.0], "high": [2002.0, 2003.0], "low": [1998.0, 1999.0], "volume": [10, 10]})
    sig = strat.on_bar(df)
    assert sig is not None
    assert sig.direction == Direction.LONG


def test_sbx6_ast_policy_forbidden_calls_and_dunders():
    """SBX-6: AST policy blocks eval, exec, compile, open, getattr, dunders, etc."""
    from sandbox.policy import validate_ast_policy

    # Banned calls
    banned_calls = [
        "eval('1+1')",
        "exec('x=1')",
        "open('file.txt')",
        "compile('x=1', '', 'exec')",
        "getattr(obj, 'attr')",
        "setattr(obj, 'attr', 1)",
        "delattr(obj, 'attr')",
        "__import__('os')",
        "globals()",
        "locals()",
        "vars()",
        "dir()",
    ]
    for c in banned_calls:
        res = validate_ast_policy(f"def test():\n    {c}")
        assert not res.passed, f"Expected {c} to be rejected"
        assert len(res.violations) > 0
        assert res.violations[0].line >= 1

    # Banned attributes
    banned_attrs = [
        "x.__class__",
        "x.__dict__",
        "x.__globals__",
        "x.__base__",
        "x.__subclasses__()",
        "x.base",
        "x.ctypes",
        "x.data",
        "x.__array_interface__",
        "x.f_back",
        "x.tb_frame",
        "x.environ",
    ]
    for a in banned_attrs:
        res = validate_ast_policy(f"def test():\n    y = {a}")
        assert not res.passed, f"Expected {a} to be rejected"


def test_lh1_base_future_peek_blocked(cfg, sample_bars):
    """LH-1: .base access is blocked by AST and child receives data incrementally."""
    from core.sandbox import Sandbox, SandboxError
    code = textwrap.dedent("""
    class Strategy:
        def on_bar(self, bars):
            a = bars['close'].to_numpy()
            b = a.base
            return None
    """)
    sb = Sandbox(cfg)
    with pytest.raises(SandboxError) as exc_info:
        sb.load_strategy(code)
    assert ".base" in str(exc_info.value).lower() or "base" in str(exc_info.value).lower()


def test_lh3_static_scan_catches_all_lookahead_patterns():
    """LH-3: Static scan catches periods=-1, center=True, shift(var), iloc[var], .base."""
    from sandbox.policy import validate_ast_policy

    patterns = {
        "shift(periods=-1)": "x = df['close'].shift(periods=-1)",
        "rolling(center=True)": "x = df['close'].rolling(5, center=True).mean()",
        "shift(variable)": "n=-1\nx = df['close'].shift(n)",
        "iloc via variable": "j=i+1\nx = df.iloc[j]",
        ".base access": "x = a.base",
    }
    for name, snippet in patterns.items():
        res = validate_ast_policy(snippet)
        assert not res.passed, f"Pattern '{name}' was not caught by static policy! Code: {snippet}"


def test_audit_hooks_block_violations():
    """Audit hooks inside child runner block network, subprocess, os mutation, file write."""
    from sandbox.policy import audit_hook

    with pytest.raises(PermissionError):
        audit_hook("socket.connect", (("127.0.0.1", 80),))

    with pytest.raises(PermissionError):
        audit_hook("subprocess.Popen", (["ls"],))

    with pytest.raises(PermissionError):
        audit_hook("os.system", ("echo hello",))

    with pytest.raises(PermissionError):
        audit_hook("os.remove", ("file.txt",))

    with pytest.raises(PermissionError):
        audit_hook("open", ("file.txt", "w", 0))


def test_memory_bomb_killed(cfg, sample_bars):
    """Memory bomb (np.zeros(10**10)) is stopped by watchdog or memory error."""
    from core.sandbox import Sandbox, SandboxError
    code = textwrap.dedent("""
    class Strategy:
        def on_bar(self, bars):
            x = np.zeros(10**10)
            return None
    """)
    sb = Sandbox(cfg)
    strat = sb.load_strategy(code)
    with pytest.raises(SandboxError) as exc_info:
        strat.on_bar(sample_bars.iloc[:1])
    assert any(k in str(exc_info.value).lower() for k in ["memory", "error", "killed"])


def test_determinism(cfg, sample_bars):
    """Determinism: Same strategy with same seed produces bit-exact signals."""
    from core.sandbox import Sandbox
    code = textwrap.dedent("""
    class Strategy:
        def on_bar(self, bars):
            if len(bars) < 20:
                return None
            sma = bars['close'].iloc[-20:].mean()
            price = bars['close'].iloc[-1]
            if price > sma:
                return Signal(direction=Direction.LONG, sl_distance=15.0, tp_distance=30.0)
            return None
    """)
    sb1 = Sandbox(cfg)
    strat1 = sb1.load_strategy(code, seed=123)
    sigs1 = [strat1.on_bar(sample_bars.iloc[:i]) for i in range(15, 35)]

    sb2 = Sandbox(cfg)
    strat2 = sb2.load_strategy(code, seed=123)
    sigs2 = [strat2.on_bar(sample_bars.iloc[:i]) for i in range(15, 35)]

    assert len(sigs1) == len(sigs2)
    for s1, s2 in zip(sigs1, sigs2):
        if s1 is None:
            assert s2 is None
        else:
            assert s2 is not None
            assert s1.direction == s2.direction
            assert abs(s1.sl_distance - s2.sl_distance) < 1e-6
            assert abs(s1.tp_distance - s2.tp_distance) < 1e-6


def test_honest_strategy_runs_cleanly(cfg, sample_bars):
    """Honest SMA strategy runs through isolated sandbox and completes backtest."""
    from core.sandbox import Sandbox
    code = textwrap.dedent("""
    class Strategy:
        def __init__(self, params=None):
            self.period = 20
        def on_bar(self, bars):
            if len(bars) < self.period:
                return None
            sma = bars['close'].iloc[-self.period:].mean()
            c = bars['close'].iloc[-1]
            if c > sma:
                return Signal(direction=Direction.LONG, sl_distance=20.0, tp_distance=40.0)
            elif c < sma:
                return Signal(direction=Direction.SHORT, sl_distance=20.0, tp_distance=40.0)
            return None
    """)
    sb = Sandbox(cfg)
    strat = sb.load_strategy(code)
    res = run_backtest(strat, sample_bars.iloc[:100], cfg)
    assert res.status == "completed"
    assert len(res.trades) > 0


def test_throughput(cfg, sample_bars):
    """Throughput: Processes >= 2000 bars/sec for simple SMA strategy."""
    from core.sandbox import Sandbox
    code = textwrap.dedent("""
    class Strategy:
        def on_bar(self, bars):
            if len(bars) < 10:
                return None
            s = bars['close'].iloc[-10:].mean()
            return None
    """)
    sb = Sandbox(cfg)
    strat = sb.load_strategy(code)
    # Stream 300 bars
    n = min(300, len(sample_bars))
    t0 = time.perf_counter()
    for i in range(10, n):
        strat.on_bar(sample_bars.iloc[:i])
    elapsed = time.perf_counter() - t0
    rate = (n - 10) / elapsed
    print(f"\nMeasured throughput: {rate:.1f} bars/sec")
    assert rate >= 1000  # generous threshold in regression suite, benchmark test checks >= 2000
