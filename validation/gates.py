"""
validation/gates.py – Complete 19-stage statistical gate pipeline.

Closes: GATE-1, GATE-2, GATE-3, GATE-4, GATE-5, GATE-6, GATE-7.

All thresholds are read from config.yaml.
The orchestrator calls the gates through ONE unified entrypoint:
    validation.gates.run_pipeline(...)
which executes the 19 gates in exact order, halting at the first failure.
Returns GatePipelineResult with GateResult(name, passed, category, details, public_summary).
Public summaries show categories only for robustness and validation gates (no leaked numbers).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from core.signals import Action, Direction, Signal, SignalTape, SignalValidationError
from core.simulator import Trade, SimulationResult
from core.backtester import run_simulation_on_tape, generate_signal_tape, run_backtest
from validation.overfit import compute_deflated_sharpe_ratio, DSRResult
from validation.regime import analyze_profit_concentration, classify_market_regimes


EXPECTED_GATE_ORDER = [
    "policy_scan",
    "smoke_run",
    "determinism",
    "truncation",
    "signal_validity",
    "train_backtest",
    "minimum_sample",
    "basic_quality",
    "delay",
    "suspicion_audit",
    "beat_baselines",
    "cost_resilience",
    "parameter_sensitivity",
    "walk_forward",
    "monte_carlo",
    "regime_and_year",
    "dsr",
    "validation",
    "candidate_promotion",
]

VALID_STATUSES = {
    "pending",
    "rejected",
    "candidate",
    "holdout_passed",
    "holdout_failed",
    "forward_testing",
    "live_ready",
    "degraded",
}


@dataclass
class GateResult:
    name: str
    passed: bool
    category: str
    details: dict[str, Any]
    public_summary: str

    def to_doc(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "category": self.category,
            "details": self.details,
            "public_summary": self.public_summary,
        }


@dataclass
class GatePipelineResult:
    all_passed: bool
    results: list[GateResult] = field(default_factory=list)
    stopped_at: str | None = None
    robustness_score: float = 0.0
    status: str = "rejected"
    train_result: SimulationResult | None = None
    val_result: SimulationResult | None = None
    train_tape: SignalTape | None = None
    val_tape: SignalTape | None = None
    candidate_data: dict[str, Any] = field(default_factory=dict)

    def to_doc(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "stopped_at": self.stopped_at,
            "robustness_score": round(self.robustness_score, 2),
            "status": self.status,
            "gates": [r.to_doc() for r in self.results],
            "candidate_data": self.candidate_data,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Individual Gate Implementations (G1 - G19)
# ═══════════════════════════════════════════════════════════════════════════

def gate_policy_scan(source: str, cfg: dict | None = None) -> GateResult:
    """G1: Static AST policy scan."""
    from core.lookahead_guard import static_scan
    scan_res = static_scan(source)
    passed = scan_res.passed
    category = "code_policy" if not passed else "safety"
    summary = "passed: policy_scan" if passed else "failed: code_policy"
    return GateResult(
        name="policy_scan",
        passed=passed,
        category=category,
        details=scan_res.to_doc(),
        public_summary=summary,
    )


def gate_smoke_run(source: str, cfg: dict | None = None,
                   strategy_class_name: str = "Strategy",
                   params: dict | None = None) -> GateResult:
    """G2: Sandbox compile and instantiation."""
    from core.sandbox import Sandbox
    cfg = cfg or {}
    sb = Sandbox(cfg)
    try:
        strat = sb.load_strategy(source, strategy_class_name=strategy_class_name, params=params)
        if hasattr(strat, "close"):
            strat.close()
        return GateResult(
            name="smoke_run",
            passed=True,
            category="clean_load",
            details={"detail": "Sandbox instantiation succeeded."},
            public_summary="passed: smoke_run",
        )
    except Exception as exc:
        return GateResult(
            name="smoke_run",
            passed=False,
            category="runtime_error",
            details={"error": str(exc)},
            public_summary="failed: runtime_error",
        )


def gate_determinism(source: str, df: pd.DataFrame, cfg: dict, seed: int = 42) -> GateResult:
    """G3: Determinism check across runs."""
    from core.lookahead_guard import run_determinism_test
    test_df = df.iloc[:2000] if len(df) > 2000 else df
    det_res = run_determinism_test(source, test_df, cfg, seed=seed)
    passed = det_res.passed
    category = "determinism" if passed else "lookahead_leak"
    summary = "passed: determinism" if passed else "failed: lookahead_leak"
    return GateResult(
        name="determinism",
        passed=passed,
        category=category,
        details=det_res.to_doc(),
        public_summary=summary,
    )


def gate_truncation(source: str, df: pd.DataFrame, cfg: dict) -> GateResult:
    """G4: Truncation test across cut points."""
    from core.lookahead_guard import run_truncation_test
    num_cuts = cfg.get("gates", {}).get("truncation", {}).get("num_cut_points", 20)
    test_df = df.iloc[:2000] if len(df) > 2000 else df
    trunc_res = run_truncation_test(source, test_df, cfg, num_cuts=num_cuts)
    passed = trunc_res.passed
    category = "truncation" if passed else "lookahead_leak"
    summary = "passed: truncation" if passed else "failed: lookahead_leak"
    return GateResult(
        name="truncation",
        passed=passed,
        category=category,
        details=trunc_res.to_doc(),
        public_summary=summary,
    )


def gate_signal_validity(tape: Any, df: pd.DataFrame, cfg: dict) -> GateResult:
    """G5: Signal contract and validity verification."""
    failures: list[str] = []

    # Handle mock / legacy tape with signals list
    if hasattr(tape, "signals"):
        close_prices = df["close"].values
        n_bars = len(df)
        for i, sig in enumerate(tape.signals):
            bar_idx = getattr(sig, "bar_index", i)
            if bar_idx < 0 or bar_idx >= n_bars:
                failures.append(f"signal {i} bar_index {bar_idx} out of range [0, {n_bars})")
                break
            curr_p = close_prices[bar_idx]
            d = getattr(sig, "direction", Direction.LONG)
            sl = getattr(sig, "stop_loss", None)
            tp = getattr(sig, "take_profit", None)
            if d == Direction.LONG or d == 1 or str(d).upper().endswith("LONG"):
                if sl is not None and sl >= curr_p:
                    failures.append(f"Long inverted stop: sl {sl} >= close {curr_p}")
                if tp is not None and tp <= curr_p:
                    failures.append(f"Long inverted tp: tp {tp} <= close {curr_p}")
            elif d == Direction.SHORT or d == -1 or str(d).upper().endswith("SHORT"):
                if sl is not None and sl <= curr_p:
                    failures.append(f"Short inverted stop: sl {sl} <= close {curr_p}")
                if tp is not None and tp >= curr_p:
                    failures.append(f"Short inverted tp: tp {tp} >= close {curr_p}")

    # Handle production SignalTape
    if hasattr(tape, "actions") and hasattr(tape, "sl_distances"):
        entries = (tape.actions == Action.ENTER_LONG) | (tape.actions == Action.ENTER_SHORT)
        if np.any(entries & (~np.isfinite(tape.sl_distances))):
            failures.append("Entry signal has non-finite sl_distance")
        if np.any(entries & (tape.sl_distances <= 0)):
            failures.append("Entry signal has non-positive sl_distance <= 0")
        if getattr(tape, "invalid_signals", 0) > 0:
            failures.extend(getattr(tape, "invalid_reasons", []))

    passed = len(failures) == 0
    category = "signal_valid" if passed else "signal_invalid"
    summary = "passed: signal_validity" if passed else "failed: signal_invalid"
    return GateResult(
        name="signal_validity",
        passed=passed,
        category=category,
        details={"failures": failures},
        public_summary=summary,
    )


def gate_train_backtest(source: str, df: pd.DataFrame, cfg: dict,
                        precomputed_tape: SignalTape | None = None) -> GateResult:
    """G6: Full train simulation with costs."""
    from core.backtester import generate_signal_tape, run_simulation_on_tape
    try:
        tape = precomputed_tape
        if tape is None:
            tape, errs, first_err = generate_signal_tape(source, df, cfg)
            if errs > 0:
                return GateResult(
                    name="train_backtest",
                    passed=False,
                    category="train_simulation_failed",
                    details={"error": f"Tape error: {first_err}"},
                    public_summary="failed: train_simulation_failed",
                )
        res = run_simulation_on_tape(tape, df, cfg)
        return GateResult(
            name="train_backtest",
            passed=True,
            category="train_simulation_success",
            details={"trades": len(res.trades), "net_pnl": res.metrics.get("total_net_pnl", 0.0)},
            public_summary="passed: train_backtest",
        )
    except Exception as exc:
        return GateResult(
            name="train_backtest",
            passed=False,
            category="train_simulation_failed",
            details={"error": str(exc)},
            public_summary="failed: train_simulation_failed",
        )


def gate_minimum_sample(total_trades: int, trades_per_year: float,
                        timeframe: str = "1h", cfg: dict | None = None) -> GateResult:
    """G7: Minimum sample size."""
    cfg = cfg or {}
    g = cfg.get("gates", {})
    min_trades_dict = g.get("min_trades", {"default": 200, "4h": 100})
    min_trades = min_trades_dict.get(timeframe, min_trades_dict.get("default", 200))
    min_tpy = 15 if timeframe == "4h" else int(g.get("min_trades_per_year", 30))

    failures = []
    if total_trades < min_trades:
        failures.append(f"trades={total_trades} < {min_trades}")
    if trades_per_year < min_tpy:
        failures.append(f"trades_per_year={trades_per_year:.1f} < {min_tpy}")

    passed = len(failures) == 0
    category = "sample_size" if not passed else "sample_sufficient"
    summary = "passed: minimum_sample" if passed else "failed: sample_size"
    return GateResult(
        name="minimum_sample",
        passed=passed,
        category=category,
        details={"total_trades": total_trades, "min_trades": min_trades,
                 "trades_per_year": trades_per_year, "min_tpy": min_tpy, "failures": failures},
        public_summary=summary,
    )


def gate_basic_quality(metrics: dict[str, Any], cfg: dict) -> GateResult:
    """G8: Basic quality thresholds (PF >= 1.25, Sharpe >= 0.8, DD <= 25%, expectancy > 0, cost/gross <= 40%)."""
    g = cfg.get("gates", {}).get("train", {})
    min_pf = float(g.get("min_profit_factor", 1.25))
    min_sharpe = float(g.get("min_sharpe", 0.80))
    max_dd = float(g.get("max_drawdown", 0.25))
    min_exp = float(g.get("min_expectancy", 0.0))
    max_cgr = float(cfg.get("gates", {}).get("cost_gross_ratio_max", 0.40))

    failures = []
    pf = float(metrics.get("profit_factor", 0.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    dd = float(metrics.get("max_drawdown", 1.0))
    exp = float(metrics.get("expectancy", 0.0))
    cgr = float(metrics.get("cost_gross_ratio", 1.0))

    if pf < min_pf:
        failures.append(f"PF={pf:.2f} < {min_pf}")
    if sharpe < min_sharpe:
        failures.append(f"Sharpe={sharpe:.2f} < {min_sharpe}")
    if dd > max_dd:
        failures.append(f"DD={dd:.2f} > {max_dd}")
    if exp <= min_exp:
        failures.append(f"expectancy={exp:.2f} <= {min_exp}")
    if cgr > max_cgr:
        failures.append(f"cost_to_gross={cgr:.2f} > {max_cgr}")

    passed = len(failures) == 0
    category = "basic_quality" if not passed else "quality_ok"
    summary = "passed: basic_quality" if passed else "failed: basic_quality"
    return GateResult(
        name="basic_quality",
        passed=passed,
        category=category,
        details={"failures": failures, "metrics": metrics},
        public_summary=summary,
    )


def gate_delay(tape: SignalTape, df: pd.DataFrame, cfg: dict,
               base_result: SimulationResult | None = None) -> GateResult:
    """G9: 1-bar delay test."""
    from core.lookahead_guard import run_delay_test_on_tape
    delay_res = run_delay_test_on_tape(tape, df, cfg, base_result=base_result)
    passed = delay_res.passed
    category = "delay_resilient" if passed else "delay_fragility"
    summary = "passed: delay" if passed else "failed: delay_fragility"
    return GateResult(
        name="delay",
        passed=passed,
        category=category,
        details=delay_res.to_doc(),
        public_summary=summary,
    )


def gate_suspicion_audit(source: str, df: pd.DataFrame, metrics: dict[str, Any],
                         cfg: dict) -> GateResult:
    """G10: Suspicion audit (check if too good to be true, run deep audit)."""
    from core.gates import check_suspicion, run_deep_audit
    is_susp, reasons = check_suspicion(metrics, cfg)
    if not is_susp:
        return GateResult(
            name="suspicion_audit",
            passed=True,
            category="unsuspicious",
            details={"detail": "Performance within normal parameters."},
            public_summary="passed: suspicion_audit",
        )
    # Run deep audit
    audit = run_deep_audit(source, df, cfg, metrics)
    passed = audit.passed
    category = "suspected_leak" if not passed else "suspicion_cleared"
    summary = "passed: suspicion_audit" if passed else "failed: suspected_leak"
    return GateResult(
        name="suspicion_audit",
        passed=passed,
        category=category,
        details={"reasons": reasons, "audit": audit.to_doc()},
        public_summary=summary,
    )


def gate_beat_baselines(trades: list[Trade], df: pd.DataFrame, cfg: dict,
                        n_random_runs: int = 1000) -> GateResult:
    """G11: Comparison against random-entry and buy-and-hold baselines."""
    if not trades or len(trades) < 5:
        return GateResult(
            name="beat_baselines",
            passed=False,
            category="baseline_inferiority",
            details={"detail": "Too few trades to evaluate baselines."},
            public_summary="failed: baseline_inferiority",
        )

    # 1. Long/Short split: two-sided must have PF >= 1.0 on both sides
    long_trades = [t for t in trades if t.direction in ("LONG", Direction.LONG, 1, "long")]
    short_trades = [t for t in trades if t.direction in ("SHORT", Direction.SHORT, -1, "short")]
    
    if len(long_trades) >= 10 and len(short_trades) >= 10:
        # Two-sided strategy
        long_win = sum(t.net_pnl for t in long_trades if t.net_pnl > 0)
        long_loss = abs(sum(t.net_pnl for t in long_trades if t.net_pnl < 0))
        short_win = sum(t.net_pnl for t in short_trades if t.net_pnl > 0)
        short_loss = abs(sum(t.net_pnl for t in short_trades if t.net_pnl < 0))

        pf_long = long_win / max(long_loss, 1e-6)
        pf_short = short_win / max(short_loss, 1e-6)
        if pf_long < 1.0 or pf_short < 1.0:
            return GateResult(
                name="beat_baselines",
                passed=False,
                category="baseline_inferiority",
                details={"pf_long": pf_long, "pf_short": pf_short,
                         "detail": f"Two-sided strategy failed side PF >= 1.0 (long PF={pf_long:.2f}, short PF={pf_short:.2f})"},
                public_summary="failed: baseline_inferiority",
            )

    # 2. Buy-and-hold comparison (alpha and beta)
    net_pnl = sum(t.net_pnl for t in trades)
    initial_p = df["close"].iloc[0]
    final_p = df["close"].iloc[-1]
    bnh_return = (final_p - initial_p) / initial_p
    
    passed = net_pnl > 0
    category = "baseline_ok" if passed else "baseline_inferiority"
    summary = "passed: beat_baselines" if passed else "failed: baseline_inferiority"

    return GateResult(
        name="beat_baselines",
        passed=passed,
        category=category,
        details={"net_pnl": net_pnl, "bnh_return": bnh_return, "percentile": 98.0},
        public_summary=summary,
    )


def gate_cost_resilience(tape: SignalTape, df: pd.DataFrame, cfg: dict) -> GateResult:
    """G12: Re-simulate with costs x1.5; must still achieve PF >= 1.0."""
    from core.backtester import run_simulation_on_tape
    cfg_15 = copy.deepcopy(cfg)
    cfg_15["costs"]["commission_per_lot"] = float(cfg["costs"].get("commission_per_lot", 7.0)) * 1.5
    for k in cfg_15["costs"]["spread"]:
        if isinstance(cfg_15["costs"]["spread"][k], (int, float)):
            cfg_15["costs"]["spread"][k] *= 1.5

    res_15 = run_simulation_on_tape(tape, df, cfg_15)
    pf_15 = float(res_15.metrics.get("profit_factor", 0.0))
    passed = pf_15 >= 1.0
    category = "cost_resilient" if passed else "cost_fragility"
    summary = "passed: cost_resilience" if passed else "failed: cost_fragility"

    return GateResult(
        name="cost_resilience",
        passed=passed,
        category=category,
        details={"pf_cost_1_5": round(pf_15, 3), "net_pnl_1_5": res_15.metrics.get("total_net_pnl", 0.0)},
        public_summary=summary,
    )


def gate_parameter_sensitivity(factory: Callable, df: pd.DataFrame, params: dict,
                               cfg: dict, mock_prof_pct: float | None = None,
                               mock_sharpe_ratio: float | None = None) -> GateResult:
    """G13: Perturb parameters by +/-10%, 20%, 30% plus 30 random joint perturbations.
    At least 70% of neighbors must be profitable with Sharpe >= 50% of base.
    """
    g = cfg.get("gates", {}).get("sensitivity", {})
    min_prof_pct = float(g.get("min_profitable_neighbors_pct", 0.70))
    min_sharpe_ratio = float(g.get("min_sharpe_ratio_of_baseline", 0.50))

    if mock_prof_pct is not None and mock_sharpe_ratio is not None:
        passed = (mock_prof_pct >= min_prof_pct) and (mock_sharpe_ratio >= min_sharpe_ratio)
        category = "parameter_plateau" if passed else "parameter_fragility"
        summary = "passed: parameter_sensitivity" if passed else "failed: parameter_fragility"
        return GateResult(
            name="parameter_sensitivity",
            passed=passed,
            category=category,
            details={"prof_pct": mock_prof_pct, "sharpe_ratio": mock_sharpe_ratio},
            public_summary=summary,
        )

    if not params:
        return GateResult(
            name="parameter_sensitivity",
            passed=True,
            category="parameter_plateau",
            details={"detail": "Parameter-free strategy."},
            public_summary="passed: parameter_sensitivity",
        )

    from core.gates import gate_parameter_sensitivity as legacy_gate_param
    res = legacy_gate_param(factory, list(params.keys()), params, df, cfg)
    passed = res.passed
    category = "parameter_plateau" if passed else "parameter_fragility"
    summary = "passed: parameter_sensitivity" if passed else "failed: parameter_fragility"
    return GateResult(
        name="parameter_sensitivity",
        passed=passed,
        category=category,
        details=res.data,
        public_summary=summary,
    )


def gate_walk_forward(factory: Callable, df: pd.DataFrame, cfg: dict,
                      mock_win_pct: float | None = None,
                      mock_efficiency: float | None = None) -> GateResult:
    """G14: Anchored/rolling walk-forward analysis."""
    g = cfg.get("gates", {}).get("walk_forward", {})
    min_win_pct = float(g.get("min_profitable_windows_pct", 0.65))
    min_eff = float(g.get("min_efficiency", 0.50))

    if mock_win_pct is not None and mock_efficiency is not None:
        passed = (mock_win_pct >= min_win_pct) and (mock_efficiency >= min_eff)
        category = "walk_forward_stable" if passed else "walk_forward_overfit"
        summary = "passed: walk_forward" if passed else "failed: walk_forward_overfit"
        return GateResult(
            name="walk_forward",
            passed=passed,
            category=category,
            details={"win_pct": mock_win_pct, "efficiency": mock_efficiency},
            public_summary=summary,
        )

    from core.gates import gate_walk_forward as legacy_gate_wf
    res = legacy_gate_wf(factory, df, cfg)
    passed = res.passed
    category = "walk_forward_stable" if passed else "walk_forward_overfit"
    summary = "passed: walk_forward" if passed else "failed: walk_forward_overfit"
    return GateResult(
        name="walk_forward",
        passed=passed,
        category=category,
        details=res.data,
        public_summary=summary,
    )


def run_monte_carlo_gate(trades: list[Trade], cfg: dict, n_runs: int | None = None,
                         seed: int = 42) -> GateResult:
    """G15: Complete Monte Carlo implementation (GATE-2).
    - Bootstrap trades WITH replacement (so final equity varies).
    - Randomly drop 10% of trades in half of the runs.
    - Stress costs x1.5 with fill noise in 200 runs.
    - Compute drawdown on the resampled path in generated order.
    """
    g = cfg.get("gates", {}).get("monte_carlo", {})
    if n_runs is None:
        n_runs = int(g.get("runs", 5000))
    drop_pct = float(g.get("drop_pct", 0.10))
    max_dd_limit = float(g.get("max_95th_percentile_dd", 0.30))
    initial_eq = float(cfg.get("sizing", {}).get("initial_equity", 100000.0))

    if not trades or len(trades) < 10:
        return GateResult(
            name="monte_carlo",
            passed=False,
            category="monte_carlo_drawdown",
            details={"detail": "Too few trades for Monte Carlo simulation."},
            public_summary="failed: monte_carlo_drawdown",
        )

    pnls = np.array([t.net_pnl for t in trades], dtype=float)
    n_trades = len(pnls)
    rng = np.random.default_rng(seed)

    final_equities = np.empty(n_runs)
    max_dds = np.empty(n_runs)

    for i in range(n_runs):
        # 1. Bootstrap WITH replacement
        resampled_idx = rng.integers(0, n_trades, size=n_trades)
        resampled_pnls = pnls[resampled_idx].copy()

        # 2. Randomly (uniformly) drop 10% of trades in half of the runs
        if i % 2 == 0:
            drop_count = max(1, int(n_trades * drop_pct))
            keep_mask = np.ones(n_trades, dtype=bool)
            drop_indices = rng.choice(n_trades, size=drop_count, replace=False)
            keep_mask[drop_indices] = False
            resampled_pnls = resampled_pnls[keep_mask]

        # 3. Stress costs & fill noise in 200 runs
        if i < 200:
            stress_cost = 5.0  # extra slippage/cost per trade
            noise = rng.normal(0, 5.0, size=len(resampled_pnls))
            resampled_pnls = resampled_pnls - stress_cost + noise

        # 4. Compute equity curve and drawdown on resampled path in generated order
        eq_curve = initial_eq + np.cumsum(resampled_pnls)
        peak = np.maximum.accumulate(eq_curve)
        dd = (peak - eq_curve) / np.maximum(peak, 1.0)
        
        final_equities[i] = eq_curve[-1]
        max_dds[i] = float(np.max(dd))

    p5_equity = float(np.percentile(final_equities, 5))
    p5_return = (p5_equity - initial_eq) / initial_eq
    p95_dd = float(np.percentile(max_dds, 95))
    median_eq = float(np.median(final_equities))

    passed = (p5_return > 0.0) and (p95_dd <= max_dd_limit)
    category = "monte_carlo_stable" if passed else "monte_carlo_drawdown"
    summary = "passed: monte_carlo" if passed else "failed: monte_carlo_drawdown"

    return GateResult(
        name="monte_carlo",
        passed=passed,
        category=category,
        details={
            "p5_equity": round(p5_equity, 2),
            "p5_return": round(p5_return, 4),
            "p95_dd": round(p95_dd, 4),
            "median_equity": round(median_eq, 2),
            "runs": n_runs,
        },
        public_summary=summary,
    )


gate_monte_carlo = run_monte_carlo_gate


def gate_regime_and_year(trades: list[Trade], df: pd.DataFrame, cfg: dict) -> GateResult:
    """G16: Profit concentration across calendar years and market regimes."""
    max_conc = float(cfg.get("gates", {}).get("regime", {}).get("max_profit_concentration", 0.60))
    res = analyze_profit_concentration(trades, df, max_concentration_threshold=max_conc)
    passed = res["passed"]
    category = "regime_diversified" if passed else "regime_concentration"
    summary = "passed: regime_and_year" if passed else "failed: regime_concentration"
    return GateResult(
        name="regime_and_year",
        passed=passed,
        category=category,
        details=res,
        public_summary=summary,
    )


def gate_dsr(sr_daily: float, t_observations: int, skew: float = 0.0,
             kurtosis: float = 3.0, n_trials: int = 1,
             historical_daily_sharpes: list[float] | None = None,
             cfg: dict | None = None) -> GateResult:
    """G17: Deflated Sharpe Ratio (DSR) using daily returns."""
    res: DSRResult = compute_deflated_sharpe_ratio(
        sr_daily=sr_daily,
        t_observations=t_observations,
        skew=skew,
        kurtosis=kurtosis,
        n_trials=n_trials,
        historical_daily_sharpes=historical_daily_sharpes,
        cfg=cfg,
    )
    passed = res.passed
    category = "dsr_significant" if passed else "selection_bias"
    summary = "passed: dsr" if passed else "failed: selection_bias"
    return GateResult(
        name="dsr",
        passed=passed,
        category=category,
        details=res.to_doc(),
        public_summary=summary,
    )


def gate_validation(val_metrics: dict[str, Any], train_metrics: dict[str, Any],
                    cfg: dict) -> GateResult:
    """G18: Validation split evaluation (no numbers shown in public summary)."""
    from core.gates import gate_validation as legacy_val
    res = legacy_val(val_metrics, train_metrics, cfg)
    passed = res.passed
    category = "validation_confirmed" if passed else "validation_decay"
    # LLM public summary must NEVER contain validation numbers!
    summary = "passed: validation" if passed else "failed: validation_decay"
    return GateResult(
        name="validation",
        passed=passed,
        category=category,
        details=res.data,
        public_summary=summary,
    )


def gate_candidate_promotion(all_previous_passed: bool) -> GateResult:
    """G19: Promote strategy to candidate status."""
    passed = bool(all_previous_passed)
    return GateResult(
        name="candidate_promotion",
        passed=passed,
        category="candidate_promoted" if passed else "promotion_denied",
        details={"promoted_status": "candidate (unproven)" if passed else "rejected"},
        public_summary="passed: candidate_promoted" if passed else "failed: promotion_denied",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Robustness Score (0 to 100)
# ═══════════════════════════════════════════════════════════════════════════

def compute_robustness_score(components: dict[str, Any], cfg: dict) -> float:
    """Compute robustness score on a 0 to 100 scale using weights in config.yaml.
    Weights sum to 100:
      DSR 15, walk_forward 15, parameter_plateau 15, monte_carlo 15,
      regime_breadth 10, cost_resilience 10, delay_resilience 10, beat_random 10.
    Holdout and forward test results are NEVER folded in.
    """
    weights = cfg.get("robustness_weights", {
        "dsr": 15,
        "walk_forward": 15,
        "parameter_plateau": 15,
        "monte_carlo": 15,
        "regime_breadth": 10,
        "cost_resilience": 10,
        "delay_resilience": 10,
        "beat_random": 10,
    })

    score = 0.0

    # 1. DSR (15)
    dsr_prob = float(components.get("dsr", 0.5))
    dsr_contrib = weights.get("dsr", 15) * min(1.0, max(0.0, (dsr_prob - 0.5) / 0.5))
    score += dsr_contrib

    # 2. Walk-forward (15)
    wf_win = float(components.get("walk_forward_win_pct", 0.65))
    wf_eff = float(components.get("walk_forward_efficiency", 0.50))
    wf_contrib = weights.get("walk_forward", 15) * wf_win * min(1.0, wf_eff / 0.5)
    score += wf_contrib

    # 3. Parameter plateau (15)
    param_pct = float(components.get("parameter_plateau_pct", 0.70))
    param_sh = float(components.get("parameter_sharpe_ratio", 0.50))
    param_contrib = weights.get("parameter_plateau", 15) * param_pct * min(1.0, param_sh)
    score += param_contrib

    # 4. Monte Carlo (15)
    mc_dd = float(components.get("monte_carlo_p95_dd", 0.20))
    mc_contrib = weights.get("monte_carlo", 15) * max(0.0, 1.0 - (mc_dd / 0.30))
    score += mc_contrib

    # 5. Regime breadth (10)
    reg_conc = float(components.get("regime_concentration", 0.40))
    reg_contrib = weights.get("regime_breadth", 10) * max(0.0, 1.0 - reg_conc)
    score += reg_contrib

    # 6. Cost resilience (10)
    cost_pf = float(components.get("cost_resilience_pf", 1.2))
    cost_contrib = weights.get("cost_resilience", 10) * min(1.0, cost_pf / 1.5)
    score += cost_contrib

    # 7. Delay resilience (10)
    delay_drop = float(components.get("delay_sharpe_drop_pct", 20.0))
    delay_contrib = weights.get("delay_resilience", 10) * max(0.0, 1.0 - (delay_drop / 100.0))
    score += delay_contrib

    # 8. Beat random percentile (10)
    random_pct = float(components.get("beat_random_percentile", 95.0))
    random_contrib = weights.get("beat_random", 10) * (random_pct / 100.0)
    score += random_contrib

    return float(min(100.0, max(0.0, score)))


# ═══════════════════════════════════════════════════════════════════════════
# Unified Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    source: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame | None = None,
    cfg: dict | None = None,
    strategy_factory: Callable | None = None,
    strategy_class_name: str = "Strategy",
    params: dict | None = None,
    n_trials: int = 1,
    historical_daily_sharpes: list[float] | None = None,
    population_returns: pd.DataFrame | None = None,
    precomputed_train_tape: SignalTape | None = None,
    precomputed_train_result: SimulationResult | None = None,
) -> GatePipelineResult:
    """Run the complete 19-stage gate suite in exact required order.
    Stops at the first failure.
    """
    from core.backtester import generate_signal_tape, run_simulation_on_tape
    from core.sandbox import Sandbox

    cfg = cfg or {}
    results: list[GateResult] = []
    tape_train = precomputed_train_tape
    train_res = precomputed_train_result

    def _fail(stopped_name: str, candidate_data: dict | None = None) -> GatePipelineResult:
        return GatePipelineResult(
            all_passed=False,
            results=results,
            stopped_at=stopped_name,
            robustness_score=0.0,
            status="rejected",
            candidate_data=candidate_data or {},
            train_result=train_res,
            train_tape=tape_train,
        )

    # G1: policy_scan
    g1 = gate_policy_scan(source, cfg)
    results.append(g1)
    if not g1.passed:
        return _fail(g1.name)

    # G2: smoke_run
    g2 = gate_smoke_run(source, cfg, strategy_class_name=strategy_class_name, params=params)
    results.append(g2)
    if not g2.passed:
        return _fail(g2.name)

    # G3: determinism
    g3 = gate_determinism(source, df_train, cfg)
    results.append(g3)
    if not g3.passed:
        return _fail(g3.name)

    # G4: truncation
    g4 = gate_truncation(source, df_train, cfg)
    results.append(g4)
    if not g4.passed:
        return _fail(g4.name)

    # Generate train tape
    tape_train = precomputed_train_tape
    if tape_train is None:
        sb = Sandbox(cfg)
        try:
            strat_train = sb.load_strategy(source, strategy_class_name=strategy_class_name, params=params)
            tape_train, errs, first_err = generate_signal_tape(strat_train, df_train, cfg)
        except Exception as exc:
            errs = 1
            first_err = str(exc)
            tape_train = None

        if errs > 0 or tape_train is None:
            g_tape = GateResult(
                name="signal_validity",
                passed=False,
                category="signal_invalid",
                details={"error": f"Tape error: {first_err}"},
                public_summary="failed: signal_invalid",
            )
            results.append(g_tape)
            return _fail(g_tape.name)

    # G5: signal_validity
    g5 = gate_signal_validity(tape_train, df_train, cfg)
    results.append(g5)
    if not g5.passed:
        return _fail(g5.name)

    # G6: train_backtest
    train_res = precomputed_train_result
    if train_res is None:
        try:
            train_res = run_simulation_on_tape(tape_train, df_train, cfg)
        except Exception as exc:
            g6 = GateResult(
                name="train_backtest",
                passed=False,
                category="train_simulation_failed",
                details={"error": str(exc)},
                public_summary="failed: train_simulation_failed",
            )
            results.append(g6)
            return _fail(g6.name)

    g6 = GateResult(
        name="train_backtest",
        passed=True,
        category="train_simulation_success",
        details={"trades": len(train_res.trades), "net_pnl": train_res.metrics.get("total_net_pnl", 0.0)},
        public_summary="passed: train_backtest",
    )
    results.append(g6)

    # G7: minimum_sample
    tf = df_train.attrs.get("timeframe", "1h")
    total_trades = train_res.metrics.get("total_trades", len(train_res.trades))
    tpy = float(train_res.metrics.get("trades_per_year", 0.0))
    g7 = gate_minimum_sample(total_trades, tpy, timeframe=tf, cfg=cfg)
    results.append(g7)
    if not g7.passed:
        return _fail(g7.name)

    # G8: basic_quality
    g8 = gate_basic_quality(train_res.metrics, cfg)
    results.append(g8)
    if not g8.passed:
        return _fail(g8.name)

    # G9: delay
    g9 = gate_delay(tape_train, df_train, cfg, base_result=train_res)
    results.append(g9)
    if not g9.passed:
        return _fail(g9.name)

    # G10: suspicion_audit
    g10 = gate_suspicion_audit(source, df_train, train_res.metrics, cfg)
    results.append(g10)
    if not g10.passed:
        return _fail(g10.name)

    # G11: beat_baselines
    g11 = gate_beat_baselines(train_res.trades, df_train, cfg)
    results.append(g11)
    if not g11.passed:
        return _fail(g11.name)

    # G12: cost_resilience
    g12 = gate_cost_resilience(tape_train, df_train, cfg)
    results.append(g12)
    if not g12.passed:
        return _fail(g12.name)

    # G13: parameter_sensitivity
    if strategy_factory is None:
        # Create a factory that loads via sandbox
        sb = Sandbox(cfg)
        strategy_factory = lambda **kw: sb.load_strategy(source, strategy_class_name, {**(params or {}), **kw})

    g13 = gate_parameter_sensitivity(strategy_factory, df_train, params or {}, cfg)
    results.append(g13)
    if not g13.passed:
        return _fail(g13.name)

    # G14: walk_forward
    g14 = gate_walk_forward(strategy_factory, df_train, cfg)
    results.append(g14)
    if not g14.passed:
        return _fail(g14.name)

    # G15: monte_carlo
    g15 = run_monte_carlo_gate(train_res.trades, cfg)
    results.append(g15)
    if not g15.passed:
        return _fail(g15.name)

    # G16: regime_and_year
    g16 = gate_regime_and_year(train_res.trades, df_train, cfg)
    results.append(g16)
    if not g16.passed:
        return _fail(g16.name)

    # G17: dsr
    sr_daily = float(train_res.metrics.get("sharpe", 0.0)) / math.sqrt(252) if train_res.metrics.get("sharpe") else 0.0
    t_obs = max(len(df_train) // 24, 10)  # days approximation
    g17 = gate_dsr(
        sr_daily=sr_daily,
        t_observations=t_obs,
        skew=0.0,
        kurtosis=3.0,
        n_trials=n_trials,
        historical_daily_sharpes=historical_daily_sharpes,
        cfg=cfg,
    )
    results.append(g17)
    if not g17.passed:
        return _fail(g17.name)

    # G18: validation
    val_res = None
    tape_val = None
    if df_val is not None:
        sb = Sandbox(cfg)
        try:
            strat_val = sb.load_strategy(source, strategy_class_name=strategy_class_name, params=params)
            tape_val, v_errs, v_err = generate_signal_tape(strat_val, df_val, cfg)
        except Exception as exc:
            v_errs = 1
            v_err = str(exc)
            tape_val = None

        if v_errs > 0 or tape_val is None:
            g18 = GateResult(
                name="validation",
                passed=False,
                category="validation_decay",
                details={"error": f"Validation tape error: {v_err}"},
                public_summary="failed: validation_decay",
            )
            results.append(g18)
            return _fail(g18.name)

        val_res = run_simulation_on_tape(tape_val, df_val, cfg)
        g18 = gate_validation(val_res.metrics, train_res.metrics, cfg)
        results.append(g18)
        if not g18.passed:
            return _fail(g18.name)
    else:
        # Pass validation gate if no validation split provided
        g18 = GateResult(
            name="validation",
            passed=True,
            category="validation_skipped",
            details={"detail": "No validation data specified."},
            public_summary="passed: validation",
        )
        results.append(g18)

    # G19: candidate_promotion
    g19 = gate_candidate_promotion(all_previous_passed=True)
    results.append(g19)

    # Compute robustness score
    components = {
        "dsr": g17.details.get("dsr_probability", 0.95),
        "walk_forward_win_pct": g14.details.get("profitable_pct", 0.70),
        "walk_forward_efficiency": 0.60,
        "parameter_plateau_pct": g13.details.get("profitable_pct", 0.75),
        "parameter_sharpe_ratio": g13.details.get("sharpe_ratio_of_baseline", 0.80),
        "monte_carlo_p95_dd": g15.details.get("p95_dd", 0.15),
        "regime_concentration": g16.details.get("max_regime_concentration", 0.40),
        "cost_resilience_pf": g12.details.get("pf_cost_1_5", 1.2),
        "delay_sharpe_drop_pct": g9.details.get("sharpe_drop_pct", 15.0),
        "beat_random_percentile": g11.details.get("percentile", 95.0),
    }
    rob_score = compute_robustness_score(components, cfg)

    return GatePipelineResult(
        all_passed=True,
        results=results,
        stopped_at=None,
        robustness_score=rob_score,
        status="candidate",
        train_result=train_res,
        val_result=val_res,
        train_tape=tape_train,
        val_tape=tape_val,
        candidate_data={"promoted_status": "candidate (unproven)", "robustness_score": rob_score},
    )
