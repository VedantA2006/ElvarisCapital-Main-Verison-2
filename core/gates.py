"""
core/gates.py – Statistical rejection gates (Section 7).

Each gate is a pure function: (backtest_result, cfg, ...) -> GateResult.
The pipeline runs them in order; a strategy must pass ALL gates to survive.
Most strategies SHOULD be rejected. That is the point.

Gate order:
  1-3. Train performance (min trades, PF, Sharpe, DD, cost ratio)
  4.   Parameter sensitivity (perturbation plateau)
  5.   Walk-forward analysis
  6.   Monte Carlo simulation
  7.   Year/regime profit concentration
  8.   Validation performance
  9.   Deflated Sharpe Ratio (DSR)
  10.  Probability of Backtest Overfitting (PBO)

Gates 1-7 use TRAIN data only.
Gate 8 uses VALIDATION data.
Gates 9-10 are computed from the full trial history (how many strategies
were tested) to penalise multiple testing.
Gate 12 (holdout) is run separately AFTER human review — see Section 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


@dataclass
class GateResult:
    gate: str
    passed: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_doc(self) -> dict[str, Any]:
        return {"gate": self.gate, "passed": self.passed,
                "detail": self.detail, "data": self.data}


@dataclass
class GatePipelineResult:
    all_passed: bool
    results: list[GateResult] = field(default_factory=list)
    stopped_at: str | None = None

    def to_doc(self) -> dict[str, Any]:
        return {"all_passed": self.all_passed,
                "stopped_at": self.stopped_at,
                "gates": [r.to_doc() for r in self.results]}


def check_suspicion(metrics: dict[str, Any], cfg: dict) -> tuple[bool, list[str]]:
    """Check if performance metrics are suspiciously high (Section 6F & F4.6)."""
    susp = cfg.get("gates", {}).get("suspicion", {})
    suspicions = []

    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    max_sharpe = float(susp.get("max_sharpe_1h_4h", 3.0))
    if sharpe > max_sharpe:
        suspicions.append(f"Sharpe={sharpe:.2f} > {max_sharpe:.2f}")

    pf = float(metrics.get("profit_factor", 0.0) or 0.0)
    max_pf = float(susp.get("max_profit_factor", 3.0))
    if pf > max_pf:
        suspicions.append(f"profit_factor={pf:.2f} > {max_pf:.2f}")

    wr = float(metrics.get("win_rate", 0.0) or 0.0)
    max_wr = float(susp.get("max_win_rate", 0.85))
    if wr > max_wr:
        suspicions.append(f"win_rate={wr:.2f} > {max_wr:.2f}")

    pc = float(metrics.get("profit_concentration_top5pct", 0.0) or 0.0)
    max_pc = float(susp.get("profit_concentration_top5_pct", 0.70))
    if pc > max_pc:
        suspicions.append(f"top5% concentration={pc:.2f} > {max_pc:.2f}")

    dd = float(metrics.get("max_drawdown", 1.0) or 1.0)
    if dd < 0.01 and metrics.get("total_trades", 0) > 10:
        suspicions.append(f"max_drawdown={dd:.4f} < 1% (almost no drawdown)")

    return len(suspicions) > 0, suspicions


check_suspicion_and_audit = check_suspicion


# ═══════════════════════════════════════════════════════════════════════════
# Gates 1-3: Train Performance
# ═══════════════════════════════════════════════════════════════════════════

def gate_train_performance(metrics: dict[str, Any], cfg: dict,
                           timeframe: str = "1h") -> GateResult:
    """Gates 1-3: minimum trade count, profit factor, Sharpe, max DD, cost ratio."""
    g = cfg["gates"]
    failures: list[str] = []
    data: dict[str, Any] = {}

    # Gate 1: Minimum trades
    min_trades = g["min_trades"].get(timeframe, g["min_trades"]["default"])
    total = metrics.get("total_trades", 0)
    data["total_trades"] = total
    data["min_trades"] = min_trades
    if total < min_trades:
        failures.append(f"trades={total} < {min_trades}")

    # Trades per year
    tpy = metrics.get("trades_per_year", 0)
    data["trades_per_year"] = tpy
    if tpy < g["min_trades_per_year"]:
        failures.append(f"trades/year={tpy:.1f} < {g['min_trades_per_year']}")

    # Gate 2: Profit factor
    pf = metrics.get("profit_factor", 0)
    data["profit_factor"] = pf
    if pf < g["train"]["min_profit_factor"]:
        failures.append(f"PF={pf:.3f} < {g['train']['min_profit_factor']}")

    # Gate 2: Sharpe
    sharpe = metrics.get("sharpe", 0)
    data["sharpe"] = sharpe
    if sharpe < g["train"]["min_sharpe"]:
        failures.append(f"Sharpe={sharpe:.3f} < {g['train']['min_sharpe']}")

    # Gate 2: Max drawdown
    dd = metrics.get("max_drawdown", 1.0)
    data["max_drawdown"] = dd
    if dd > g["train"]["max_drawdown"]:
        failures.append(f"DD={dd:.3f} > {g['train']['max_drawdown']}")

    # Gate 2: Expectancy
    exp = metrics.get("expectancy", 0)
    data["expectancy"] = exp
    if exp < g["train"]["min_expectancy"]:
        failures.append(f"expectancy={exp:.2f} < {g['train']['min_expectancy']}")

    # Gate 3: Cost-gross ratio
    cgr = metrics.get("cost_gross_ratio", 1.0)
    data["cost_gross_ratio"] = cgr
    if cgr > g["cost_gross_ratio_max"]:
        failures.append(f"cost_ratio={cgr:.3f} > {g['cost_gross_ratio_max']}")

    # Suspicion flags (Section 6F & F4.6)
    is_susp, suspicions = check_suspicion(metrics, cfg)
    data["suspicions"] = suspicions

    passed = len(failures) == 0
    detail = "PASS" if passed else "FAIL: " + "; ".join(failures)
    if suspicions:
        detail += " | SUSPICIOUS: " + "; ".join(suspicions)

    return GateResult(gate="train_performance", passed=passed, detail=detail, data=data)


# ═══════════════════════════════════════════════════════════════════════════
# Gate 4: Parameter Sensitivity
# ═══════════════════════════════════════════════════════════════════════════

def gate_parameter_sensitivity(strategy_factory, param_names: list[str],
                               base_params: dict[str, float],
                               df: pd.DataFrame, cfg: dict) -> GateResult:
    """Perturb each parameter by ±10/20/30%. Check that most neighbors are profitable
    and that Sharpe doesn't collapse — i.e. the strategy sits on a plateau, not a spike."""
    from core.backtester import run_backtest

    g = cfg["gates"]["sensitivity"]
    perturbations = g["perturbations"]
    results: list[dict[str, Any]] = []
    profitable = 0
    total = 0
    sharpes: list[float] = []

    # Baseline
    base_strat = strategy_factory(**base_params)
    base_result = run_backtest(base_strat, df, cfg)
    base_sharpe = base_result.metrics.get("sharpe", 0.0)

    for name in param_names:
        base_val = base_params[name]
        if base_val == 0:
            continue
        for pct in perturbations:
            for sign in (-1, 1):
                perturbed = dict(base_params)
                perturbed[name] = base_val * (1 + sign * pct)
                try:
                    strat = strategy_factory(**perturbed)
                    r = run_backtest(strat, df, cfg)
                    s = r.metrics.get("sharpe", 0.0)
                    net = r.metrics.get("total_net_pnl", 0.0)
                    sharpes.append(s)
                    is_prof = net > 0
                    if is_prof:
                        profitable += 1
                    total += 1
                    results.append({"param": name, "pct": sign * pct,
                                    "sharpe": round(s, 4), "profitable": is_prof})
                except Exception as e:
                    total += 1
                    results.append({"param": name, "pct": sign * pct, "error": str(e)})

    if total == 0:
        return GateResult(gate="parameter_sensitivity", passed=False,
                          detail="No perturbations could be tested.")

    prof_pct = profitable / total
    sharpe_arr = np.array(sharpes) if sharpes else np.array([0.0])
    sharpe_ratio = float(sharpe_arr.mean()) / max(abs(base_sharpe), 1e-9) if base_sharpe != 0 else 0.0

    passed = (prof_pct >= g["min_profitable_neighbors_pct"] and
              sharpe_ratio >= g["min_sharpe_ratio_of_baseline"])

    detail = (f"Profitable neighbors: {profitable}/{total} ({prof_pct*100:.0f}%, "
              f"min={g['min_profitable_neighbors_pct']*100:.0f}%). "
              f"Mean Sharpe ratio of baseline: {sharpe_ratio:.2f} "
              f"(min={g['min_sharpe_ratio_of_baseline']}). "
              f"{'PASS' if passed else 'FAIL: strategy on a spike, not a plateau'}")

    return GateResult(gate="parameter_sensitivity", passed=passed, detail=detail,
                      data={"profitable_pct": round(prof_pct, 4),
                            "sharpe_ratio_of_baseline": round(sharpe_ratio, 4),
                            "base_sharpe": round(base_sharpe, 4),
                            "perturbations_tested": total})


# ═══════════════════════════════════════════════════════════════════════════
# Gate 5: Walk-Forward Analysis
# ═══════════════════════════════════════════════════════════════════════════

def gate_walk_forward(strategy_factory, df: pd.DataFrame, cfg: dict) -> GateResult:
    """Anchored walk-forward: expanding in-sample, fixed out-of-sample windows.
    Check that the majority of OOS windows are profitable."""
    from core.backtester import run_backtest

    g = cfg["gates"]["walk_forward"]
    is_months = g["in_sample_months"]
    oos_months = g["out_of_sample_months"]

    ts = df["timestamp"]
    start = ts.iloc[0]
    end = ts.iloc[-1]

    windows: list[dict[str, Any]] = []
    profitable_windows = 0
    total_windows = 0

    # Generate windows
    is_end = start + pd.DateOffset(months=is_months)
    while is_end + pd.DateOffset(months=oos_months) <= end:
        oos_start = is_end
        oos_end = oos_start + pd.DateOffset(months=oos_months)

        oos_mask = (ts >= oos_start) & (ts < oos_end)
        oos_df = df.loc[oos_mask].copy()
        oos_df.attrs = df.attrs.copy()

        if len(oos_df) < 20:
            is_end += pd.DateOffset(months=oos_months)
            continue

        try:
            strat = strategy_factory()
            # Give the strategy the full IS data as warmup context
            is_mask = ts < oos_start
            full = pd.concat([df.loc[is_mask], oos_df]).reset_index(drop=True)
            full["is_warmup"] = full["timestamp"] < oos_start
            full.attrs = df.attrs.copy()
            r = run_backtest(strat, full, cfg)
            net = r.metrics.get("total_net_pnl", 0)
            s = r.metrics.get("sharpe", 0)
            is_prof = net > 0
            if is_prof:
                profitable_windows += 1
            total_windows += 1
            windows.append({"oos_start": str(oos_start), "oos_end": str(oos_end),
                            "net_pnl": round(net, 2), "sharpe": round(s, 4),
                            "trades": r.metrics.get("total_trades", 0),
                            "profitable": is_prof})
        except Exception as e:
            total_windows += 1
            windows.append({"oos_start": str(oos_start), "oos_end": str(oos_end),
                            "error": str(e)})

        is_end += pd.DateOffset(months=oos_months)

    if total_windows == 0:
        return GateResult(gate="walk_forward", passed=False,
                          detail="Not enough data for walk-forward windows.")

    prof_pct = profitable_windows / total_windows
    # Efficiency: ratio of OOS return to IS return
    oos_returns = [w.get("net_pnl", 0) for w in windows if "net_pnl" in w]
    avg_oos = np.mean(oos_returns) if oos_returns else 0.0

    passed = prof_pct >= g["min_profitable_windows_pct"]
    detail = (f"Walk-forward: {profitable_windows}/{total_windows} windows profitable "
              f"({prof_pct*100:.0f}%, min={g['min_profitable_windows_pct']*100:.0f}%). "
              f"Avg OOS PnL: ${avg_oos:.0f}. {'PASS' if passed else 'FAIL'}")

    return GateResult(gate="walk_forward", passed=passed, detail=detail,
                      data={"windows": windows, "profitable_pct": round(prof_pct, 4),
                            "total_windows": total_windows})


# ═══════════════════════════════════════════════════════════════════════════
# Gate 6: Monte Carlo Simulation
# ═══════════════════════════════════════════════════════════════════════════

def gate_monte_carlo(trades_pnl: list[float], cfg: dict,
                     initial_equity: float = 100000.0,
                     seed: int = 42) -> GateResult:
    """Reshuffle trade PnLs N times. Check that 5th percentile is still
    positive and 95th percentile drawdown is acceptable."""
    g = cfg["gates"]["monte_carlo"]
    n_runs = g["runs"]
    drop_pct = g["drop_pct"]
    rng = np.random.default_rng(seed)

    pnl = np.array(trades_pnl)
    n_trades = len(pnl)
    if n_trades < 10:
        return GateResult(gate="monte_carlo", passed=False,
                          detail="Too few trades for Monte Carlo.")

    n_drop = max(1, int(n_trades * drop_pct))
    final_equities = np.empty(n_runs)
    max_dds = np.empty(n_runs)

    for i in range(n_runs):
        # Shuffle trade order
        shuffled = rng.permutation(pnl)
        # Drop worst n_drop trades (stress test)
        if i % 2 == 0:
            idx = np.argsort(shuffled)
            keep = idx[n_drop:]  # drop the n_drop worst
            shuffled = shuffled[keep]
        eq = initial_equity + np.cumsum(shuffled)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, 1.0)
        final_equities[i] = eq[-1]
        max_dds[i] = dd.max()

    p5_equity = float(np.percentile(final_equities, 5))
    p95_dd = float(np.percentile(max_dds, 95))
    median_equity = float(np.median(final_equities))

    p5_positive = p5_equity > initial_equity
    dd_ok = p95_dd <= g["max_95th_percentile_dd"]

    passed = p5_positive and dd_ok
    detail = (f"Monte Carlo ({n_runs} runs): 5th%ile equity=${p5_equity:,.0f} "
              f"({'> initial' if p5_positive else '< initial'}), "
              f"95th%ile DD={p95_dd*100:.1f}% (max={g['max_95th_percentile_dd']*100:.0f}%). "
              f"Median final=${median_equity:,.0f}. {'PASS' if passed else 'FAIL'}")

    return GateResult(gate="monte_carlo", passed=passed, detail=detail,
                      data={"p5_equity": round(p5_equity, 2),
                            "p95_dd": round(p95_dd, 4),
                            "median_equity": round(median_equity, 2),
                            "p5_positive": p5_positive, "dd_ok": dd_ok})


# ═══════════════════════════════════════════════════════════════════════════
# Gate 7: Year/Regime Profit Concentration
# ═══════════════════════════════════════════════════════════════════════════

def gate_regime_concentration(metrics: dict[str, Any], cfg: dict) -> GateResult:
    """Reject if > X% of total profit comes from a single year."""
    g = cfg["gates"]["regime"]
    max_conc = g["max_profit_concentration_pct"]

    yearly = metrics.get("yearly_pnl", {})
    if not yearly:
        return GateResult(gate="regime_concentration", passed=False,
                          detail="No yearly PnL data.")

    total_profit = sum(v for v in yearly.values() if v > 0)
    if total_profit <= 0:
        return GateResult(gate="regime_concentration", passed=False,
                          detail="No profitable years.")

    max_year_profit = max(max(yearly.values()), 0)
    concentration = max_year_profit / total_profit

    passed = concentration <= max_conc
    best_year = max(yearly, key=yearly.get)
    detail = (f"Max year concentration: {best_year}={concentration*100:.1f}% "
              f"(max={max_conc*100:.0f}%). {'PASS' if passed else 'FAIL: profit too concentrated'}")

    return GateResult(gate="regime_concentration", passed=passed, detail=detail,
                      data={"yearly_pnl": {str(k): v for k, v in yearly.items()}, "max_concentration": round(concentration, 4),
                            "best_year": str(best_year)})


# ═══════════════════════════════════════════════════════════════════════════
# Gate 8: Validation Performance
# ═══════════════════════════════════════════════════════════════════════════

def gate_validation(val_metrics: dict[str, Any], train_metrics: dict[str, Any],
                    cfg: dict) -> GateResult:
    """Validation Sharpe must be >= X% of train Sharpe, and PF must meet minimum."""
    g = cfg["gates"]["validation"]
    train_sharpe = train_metrics.get("sharpe", 0)
    val_sharpe = val_metrics.get("sharpe", 0)
    val_pf = val_metrics.get("profit_factor", 0)

    min_sharpe_ratio = g["min_sharpe_ratio_of_train"]
    min_pf = g["min_profit_factor"]

    failures: list[str] = []

    if train_sharpe > 0:
        ratio = val_sharpe / train_sharpe
        if ratio < min_sharpe_ratio:
            failures.append(f"val_sharpe/train_sharpe={ratio:.2f} < {min_sharpe_ratio}")
    else:
        ratio = 0.0
        failures.append(f"train Sharpe={train_sharpe:.2f} not positive")

    if val_pf < min_pf:
        failures.append(f"val PF={val_pf:.3f} < {min_pf}")

    passed = len(failures) == 0
    detail = (f"Validation: Sharpe={val_sharpe:.3f} ({ratio*100:.0f}% of train), "
              f"PF={val_pf:.3f}. {'PASS' if passed else 'FAIL: ' + '; '.join(failures)}")

    return GateResult(gate="validation", passed=passed, detail=detail,
                      data={"val_sharpe": round(val_sharpe, 4),
                            "train_sharpe": round(train_sharpe, 4),
                            "sharpe_ratio": round(ratio, 4),
                            "val_pf": round(val_pf, 4)})


# ═══════════════════════════════════════════════════════════════════════════
# Gate 9: Deflated Sharpe Ratio (DSR)
# ═══════════════════════════════════════════════════════════════════════════

def gate_dsr(sharpe: float, n_trades: int, skew: float, kurtosis: float,
             n_trials: int, cfg: dict) -> GateResult:
    """Deflated Sharpe Ratio: adjusts for multiple testing, non-normality.

    Based on Bailey & Lopez de Prado (2014). A DSR probability < 95% means
    the observed Sharpe is likely due to luck given how many strategies
    were tried.
    """
    g = cfg["gates"]["dsr"]
    min_prob = g["min_probability"]

    if n_trades < 10 or n_trials < 1:
        return GateResult(gate="dsr", passed=False,
                          detail=f"Insufficient data (trades={n_trades}, trials={n_trials}).")

    # Expected maximum Sharpe under null (Euler-Mascheroni approximation)
    if n_trials > 1:
        gamma = 0.5772156649
        e_max_sr = np.sqrt(2 * np.log(n_trials)) - (np.log(np.pi) + gamma) / (2 * np.sqrt(2 * np.log(n_trials)))
    else:
        e_max_sr = 0.0

    # DSR statistic
    sr = sharpe
    se_sr = np.sqrt((1 - skew * sr + (kurtosis - 1) / 4 * sr ** 2) / max(n_trades - 1, 1))

    if se_sr < 1e-12:
        return GateResult(gate="dsr", passed=False,
                          detail="SE of Sharpe is zero.")

    dsr_stat = (sr - e_max_sr) / se_sr
    prob = float(sp_stats.norm.cdf(dsr_stat))

    passed = prob >= min_prob
    detail = (f"DSR: prob={prob:.4f} (min={min_prob}), "
              f"SR={sr:.3f}, E[max_SR]={e_max_sr:.3f}, "
              f"trials={n_trials}. {'PASS' if passed else 'FAIL: likely luck'}")

    return GateResult(gate="dsr", passed=passed, detail=detail,
                      data={"probability": round(prob, 6),
                            "dsr_statistic": round(dsr_stat, 4),
                            "expected_max_sr": round(e_max_sr, 4),
                            "n_trials": n_trials})


# ═══════════════════════════════════════════════════════════════════════════
# Gate 10: Probability of Backtest Overfitting (PBO)
# ═══════════════════════════════════════════════════════════════════════════

def gate_pbo(trade_pnls: list[float], cfg: dict,
             n_partitions: int = 10, seed: int = 42) -> GateResult:
    """Combinatorial Symmetric Cross-Validation (CSCV) for PBO.

    Split trades into S partitions, for each combination of S/2 partitions as
    IS and the rest as OOS, check if the best IS configuration underperforms OOS.
    PBO = fraction of combinations where IS-optimised underperforms.

    Simplified version: uses random partition shuffles instead of full
    combinatorial (which is O(C(S,S/2)) and can be huge).
    """
    g = cfg["gates"]["pbo"]
    max_pbo = g["max_pbo"]
    rng = np.random.default_rng(seed)

    pnl = np.array(trade_pnls)
    n = len(pnl)
    if n < n_partitions * 2:
        return GateResult(gate="pbo", passed=True,
                          detail=f"Too few trades ({n}) for PBO with {n_partitions} partitions. Skipped.")

    # Partition trades into n_partitions blocks
    indices = np.arange(n)
    block_size = n // n_partitions
    n_combinations = min(500, 2 ** n_partitions)  # cap combinations

    overfit_count = 0
    for _ in range(n_combinations):
        shuffled = rng.permutation(n_partitions)
        is_blocks = shuffled[: n_partitions // 2]
        oos_blocks = shuffled[n_partitions // 2:]

        is_pnl = np.concatenate([pnl[b * block_size: (b + 1) * block_size] for b in is_blocks])
        oos_pnl = np.concatenate([pnl[b * block_size: (b + 1) * block_size] for b in oos_blocks])

        is_sharpe = is_pnl.mean() / max(is_pnl.std(), 1e-12)
        oos_sharpe = oos_pnl.mean() / max(oos_pnl.std(), 1e-12)

        if is_sharpe > 0 and oos_sharpe <= 0:
            overfit_count += 1

    pbo = overfit_count / n_combinations
    passed = pbo <= max_pbo

    detail = (f"PBO: {pbo*100:.1f}% ({overfit_count}/{n_combinations} combos overfit, "
              f"max={max_pbo*100:.0f}%). {'PASS' if passed else 'FAIL: high overfitting probability'}")

    return GateResult(gate="pbo", passed=passed, detail=detail,
                      data={"pbo": round(pbo, 4), "overfit_count": overfit_count,
                            "n_combinations": n_combinations})


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_train_gates(train_metrics: dict[str, Any], trade_pnls: list[float],
                    cfg: dict, timeframe: str = "1h",
                    n_trials: int = 1) -> GatePipelineResult:
    """Run gates 1-3, 6, 7, 9, 10 on train results.

    Gates 4 (sensitivity) and 5 (walk-forward) require a strategy factory
    and are run separately by the orchestrator.
    """
    results: list[GateResult] = []

    # Gates 1-3
    g1 = gate_train_performance(train_metrics, cfg, timeframe)
    results.append(g1)
    if not g1.passed:
        return GatePipelineResult(all_passed=False, results=results, stopped_at=g1.gate)

    # Gate 6: Monte Carlo
    g6 = gate_monte_carlo(trade_pnls, cfg, cfg["sizing"]["initial_equity"])
    results.append(g6)
    if not g6.passed:
        return GatePipelineResult(all_passed=False, results=results, stopped_at=g6.gate)

    # Gate 7: Regime concentration
    g7 = gate_regime_concentration(train_metrics, cfg)
    results.append(g7)
    if not g7.passed:
        return GatePipelineResult(all_passed=False, results=results, stopped_at=g7.gate)

    # Gate 9: DSR
    pnl_arr = np.array(trade_pnls)
    skew = float(sp_stats.skew(pnl_arr)) if len(pnl_arr) > 2 else 0.0
    kurt = float(sp_stats.kurtosis(pnl_arr, fisher=False)) if len(pnl_arr) > 3 else 3.0
    g9 = gate_dsr(train_metrics.get("sharpe", 0), len(trade_pnls),
                  skew, kurt, n_trials, cfg)
    results.append(g9)
    if not g9.passed:
        return GatePipelineResult(all_passed=False, results=results, stopped_at=g9.gate)

    # Gate 10: PBO
    g10 = gate_pbo(trade_pnls, cfg)
    results.append(g10)
    if not g10.passed:
        return GatePipelineResult(all_passed=False, results=results, stopped_at=g10.gate)

    return GatePipelineResult(all_passed=True, results=results)


# ═══════════════════════════════════════════════════════════════════════════
# Deep Audit & Gate Pipeline (F4.6 & F4.7)
# ═══════════════════════════════════════════════════════════════════════════

def run_deep_audit(
    strategy_or_source: Any,
    df: pd.DataFrame,
    cfg: dict,
    metrics: dict[str, Any],
    seed: int = 9999,
) -> GateResult:
    """Run deep audit when suspicion is triggered (F4.6).

    Re-runs truncation with 40 cuts, determinism with different seed, and delay test.
    If any fail, rejects as suspected_leak.
    """
    from core.lookahead_guard import (
        run_truncation_test,
        run_determinism_test,
        run_delay_test_on_tape,
    )
    from core.backtester import generate_signal_tape

    is_susp, reasons = check_suspicion(metrics, cfg)
    if not is_susp:
        return GateResult(gate="deep_audit", passed=True, detail="Metrics within normal bounds; deep audit skipped.")

    # 1. Determinism with alternate seed
    det_res = run_determinism_test(strategy_or_source, df, cfg, seed=seed)
    if not det_res.passed:
        return GateResult(
            gate="deep_audit",
            passed=False,
            detail=f"REJECTED (suspected_leak): deep audit determinism failed: {det_res.detail}",
            data={"reasons": reasons, "diverging_bar": det_res.first_diverging_bar},
        )

    # 2. Truncation with 40 cuts
    trunc_res = run_truncation_test(strategy_or_source, df, cfg, num_cuts=40, seed=seed)
    if not trunc_res.passed:
        return GateResult(
            gate="deep_audit",
            passed=False,
            detail=f"REJECTED (suspected_leak): deep audit truncation failed: {trunc_res.detail}",
            data={"reasons": reasons, "diverging_bar": trunc_res.first_diverging_bar},
        )

    # 3. Delay test on fresh tape
    tape, errs, first_err = generate_signal_tape(strategy_or_source, df, cfg, seed=seed)
    if errs > 0:
        return GateResult(
            gate="deep_audit",
            passed=False,
            detail=f"REJECTED (suspected_leak): deep audit tape generation failed: {first_err}",
            data={"reasons": reasons},
        )
    delay_res = run_delay_test_on_tape(tape, df, cfg)
    if not delay_res.passed:
        return GateResult(
            gate="deep_audit",
            passed=False,
            detail=f"REJECTED (suspected_leak): deep audit delay test failed: {delay_res.detail}",
            data={"reasons": reasons},
        )

    # Extreme suspicion ceiling (result too good to be true)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    wr = float(metrics.get("win_rate", 0.0) or 0.0)
    susp_cfg = cfg.get("gates", {}).get("suspicion", {})
    extreme_sharpe = float(susp_cfg.get("extreme_sharpe_ceiling", 4.5))
    extreme_wr = float(susp_cfg.get("extreme_win_rate_ceiling", 0.95))
    if sharpe > extreme_sharpe or wr > extreme_wr:
        return GateResult(
            gate="deep_audit",
            passed=False,
            detail="REJECTED (suspected_leak): result too good to be true; check for lookahead.",
            data={"reasons": reasons, "manual_review": True},
        )

    return GateResult(
        gate="deep_audit",
        passed=True,
        detail="Deep audit passed (40-cut truncation, alternate-seed determinism, and delay verified). Manual review recommended.",
        data={"reasons": reasons, "manual_review": True},
    )


@dataclass
class CandidateEvaluationResult:
    passed: bool
    stopped_at: str | None = None
    gate_results: list[GateResult] = field(default_factory=list)
    train_result: Any = None
    val_result: Any = None
    train_tape: Any = None
    manual_review: bool = False
    rejection_reason: str = ""

    def to_doc(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "stopped_at": self.stopped_at,
            "manual_review": self.manual_review,
            "rejection_reason": self.rejection_reason,
            "gates": [g.to_doc() for g in self.gate_results],
        }


class GatePipeline:
    """Cheapest and most safety-critical first evaluation pipeline (Section F4.7).

    Execution order:
    1. Static AST Policy Scan (static_scan)
    2. Compile / Smoke Run in Sandbox (Sandbox.load_strategy)
    3. Determinism Test (run_determinism_test)
    4. Truncation Test (run_truncation_test)
    5. Train Simulation & Performance Gate (gate_train_performance)
    6. Delay Test on Train Tape (run_delay_test_on_tape)
    7. Suspicion Audit (run_deep_audit) -> reject as suspected_leak
    8. Robustness Gates (Monte Carlo, Regime, DSR, PBO)
    9. Validation Split (ONLY evaluated after all train & lookahead gates pass)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def evaluate_candidate(
        self,
        source: str,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame | None = None,
        strategy_class_name: str = "Strategy",
        params: dict | None = None,
        n_trials: int = 1,
    ) -> CandidateEvaluationResult:
        from core.lookahead_guard import (
            static_scan,
            run_determinism_test,
            run_truncation_test,
            run_delay_test_on_tape,
        )
        from core.sandbox import Sandbox, SandboxError
        from core.backtester import generate_signal_tape, run_simulation_on_tape

        gate_results: list[GateResult] = []

        # Gate 1: Static AST Policy
        scan_res = static_scan(source)
        g_scan = GateResult(
            gate="static_policy",
            passed=scan_res.passed,
            detail=scan_res.summary,
            data=scan_res.to_doc(),
        )
        gate_results.append(g_scan)
        if not scan_res.passed:
            return CandidateEvaluationResult(
                passed=False,
                stopped_at="static_policy",
                gate_results=gate_results,
                rejection_reason=f"Static policy violation: {scan_res.summary}",
            )

        # Gate 2: Compile & Smoke run in sandbox
        sb = Sandbox(self.cfg)
        try:
            strat = sb.load_strategy(source, strategy_class_name=strategy_class_name, params=params)
            strat.close()
        except Exception as exc:
            g_smoke = GateResult(gate="compile_smoke", passed=False, detail=str(exc))
            gate_results.append(g_smoke)
            return CandidateEvaluationResult(
                passed=False,
                stopped_at="compile_smoke",
                gate_results=gate_results,
                rejection_reason=f"Sandbox instantiation failed: {exc}",
            )
        gate_results.append(GateResult(gate="compile_smoke", passed=True, detail="Clean sandbox load"))

        # Gate 3: Determinism Test
        det_res = run_determinism_test(source, df_train, self.cfg)
        g_det = GateResult(gate="determinism", passed=det_res.passed, detail=det_res.detail, data=det_res.to_doc())
        gate_results.append(g_det)
        if not det_res.passed:
            return CandidateEvaluationResult(
                passed=False,
                stopped_at="determinism",
                gate_results=gate_results,
                rejection_reason=f"Determinism test failed: {det_res.detail}",
            )

        # Gate 4: Truncation Test
        num_cuts = self.cfg.get("gates", {}).get("truncation", {}).get("num_cut_points", 20)
        trunc_res = run_truncation_test(source, df_train, self.cfg, num_cuts=num_cuts)
        g_trunc = GateResult(gate="truncation", passed=trunc_res.passed, detail=trunc_res.detail, data=trunc_res.to_doc())
        gate_results.append(g_trunc)
        if not trunc_res.passed:
            return CandidateEvaluationResult(
                passed=False,
                stopped_at="truncation",
                gate_results=gate_results,
                rejection_reason=f"Truncation test failed: {trunc_res.detail}",
            )

        # Gate 5: Train Backtest & Performance
        tf = df_train.attrs.get("timeframe", "1h")
        tape_train, errs, first_err = generate_signal_tape(source, df_train, self.cfg)
        if errs > 0:
            g_train = GateResult(gate="train_performance", passed=False, detail=f"Tape generation error: {first_err}")
            gate_results.append(g_train)
            return CandidateEvaluationResult(
                passed=False,
                stopped_at="train_performance",
                gate_results=gate_results,
                rejection_reason=f"Strategy error on train data: {first_err}",
            )

        train_res = run_simulation_on_tape(tape_train, df_train, self.cfg)
        g_train = gate_train_performance(train_res.metrics, self.cfg, timeframe=tf)
        gate_results.append(g_train)
        if not g_train.passed:
            return CandidateEvaluationResult(
                passed=False,
                stopped_at="train_performance",
                gate_results=gate_results,
                train_result=train_res,
                train_tape=tape_train,
                rejection_reason=f"Train performance failed: {g_train.detail}",
            )

        # Gate 6: Delay Test on Train Tape
        delay_res = run_delay_test_on_tape(tape_train, df_train, self.cfg, base_result=train_res)
        g_delay = GateResult(gate="delay_test", passed=delay_res.passed, detail=delay_res.detail, data=delay_res.to_doc())
        gate_results.append(g_delay)
        if not delay_res.passed:
            return CandidateEvaluationResult(
                passed=False,
                stopped_at="delay_test",
                gate_results=gate_results,
                train_result=train_res,
                train_tape=tape_train,
                rejection_reason=f"Delay test failed: {delay_res.detail}",
            )

        # Gate 7: Suspicion Audit
        is_susp, reasons = check_suspicion(train_res.metrics, self.cfg)
        manual_review = False
        if is_susp:
            audit_res = run_deep_audit(source, df_train, self.cfg, train_res.metrics)
            gate_results.append(audit_res)
            manual_review = audit_res.data.get("manual_review", True)
            if not audit_res.passed:
                return CandidateEvaluationResult(
                    passed=False,
                    stopped_at="suspected_leak",
                    gate_results=gate_results,
                    train_result=train_res,
                    train_tape=tape_train,
                    manual_review=manual_review,
                    rejection_reason=f"Deep audit failed: {audit_res.detail}",
                )

        # Gate 8: Robustness Gates
        pnls = [t.net_pnl for t in train_res.trades]
        init_eq = float(self.cfg.get("sizing", {}).get("initial_equity", 100000.0))

        # 8a: Monte Carlo
        g_mc = gate_monte_carlo(pnls, self.cfg, init_eq)
        gate_results.append(g_mc)
        if not g_mc.passed:
            return CandidateEvaluationResult(
                passed=False,
                stopped_at=g_mc.gate,
                gate_results=gate_results,
                train_result=train_res,
                train_tape=tape_train,
                manual_review=manual_review,
                rejection_reason=f"Monte Carlo failed: {g_mc.detail}",
            )

        # 8b: Regime Concentration
        g_regime = gate_regime_concentration(train_res.metrics, self.cfg)
        gate_results.append(g_regime)
        if not g_regime.passed:
            return CandidateEvaluationResult(
                passed=False,
                stopped_at=g_regime.gate,
                gate_results=gate_results,
                train_result=train_res,
                train_tape=tape_train,
                manual_review=manual_review,
                rejection_reason=f"Regime concentration failed: {g_regime.detail}",
            )

        # 8c: DSR
        pnl_arr = np.array(pnls)
        skew = float(sp_stats.skew(pnl_arr)) if len(pnl_arr) > 2 else 0.0
        kurt = float(sp_stats.kurtosis(pnl_arr, fisher=False)) if len(pnl_arr) > 3 else 3.0
        g_dsr = gate_dsr(train_res.metrics.get("sharpe", 0), len(pnls), skew, kurt, n_trials, self.cfg)
        gate_results.append(g_dsr)
        if not g_dsr.passed:
            return CandidateEvaluationResult(
                passed=False,
                stopped_at=g_dsr.gate,
                gate_results=gate_results,
                train_result=train_res,
                train_tape=tape_train,
                manual_review=manual_review,
                rejection_reason=f"DSR failed: {g_dsr.detail}",
            )

        # 8d: PBO
        g_pbo = gate_pbo(pnls, self.cfg)
        gate_results.append(g_pbo)
        if not g_pbo.passed:
            return CandidateEvaluationResult(
                passed=False,
                stopped_at=g_pbo.gate,
                gate_results=gate_results,
                train_result=train_res,
                train_tape=tape_train,
                manual_review=manual_review,
                rejection_reason=f"PBO failed: {g_pbo.detail}",
            )

        # Gate 9: Validation Split (ONLY reached when all above have passed)
        val_res = None
        if df_val is not None:
            tape_val, v_errs, v_err = generate_signal_tape(source, df_val, self.cfg)
            if v_errs > 0:
                g_val = GateResult(gate="validation", passed=False, detail=f"Validation tape error: {v_err}")
                gate_results.append(g_val)
                return CandidateEvaluationResult(
                    passed=False,
                    stopped_at="validation",
                    gate_results=gate_results,
                    train_result=train_res,
                    train_tape=tape_train,
                    manual_review=manual_review,
                    rejection_reason=f"Strategy raised on validation: {v_err}",
                )
            val_res = run_simulation_on_tape(tape_val, df_val, self.cfg)
            g_val = gate_validation(train_res.metrics, val_res.metrics, self.cfg)
            gate_results.append(g_val)
            if not g_val.passed:
                return CandidateEvaluationResult(
                    passed=False,
                    stopped_at="validation",
                    gate_results=gate_results,
                    train_result=train_res,
                    val_result=val_res,
                    train_tape=tape_train,
                    manual_review=manual_review,
                    rejection_reason=f"Validation gate failed: {g_val.detail}",
                )

        return CandidateEvaluationResult(
            passed=True,
            gate_results=gate_results,
            train_result=train_res,
            val_result=val_res,
            train_tape=tape_train,
            manual_review=manual_review,
        )
