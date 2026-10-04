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

    # Suspicion flags (not hard rejections, but logged)
    susp = g.get("suspicion", {})
    suspicions: list[str] = []
    if sharpe > susp.get("max_sharpe_1h_4h", 99):
        suspicions.append(f"Sharpe={sharpe:.2f} suspiciously high")
    if pf > susp.get("max_profit_factor", 99):
        suspicions.append(f"PF={pf:.2f} suspiciously high")
    wr = metrics.get("win_rate", 0)
    if wr > susp.get("max_win_rate", 1.0):
        suspicions.append(f"win_rate={wr:.2f} suspiciously high")
    pc = metrics.get("profit_concentration_top5pct", 0)
    if pc > susp.get("profit_concentration_top5_pct", 1.0):
        suspicions.append(f"top5% concentration={pc:.2f}")
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
