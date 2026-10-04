"""
validation/overfit.py – Deflated Sharpe Ratio (DSR) and Population-Level PBO (CSCV).

Closes: GATE-1, GATE-3.

1. Deflated Sharpe Ratio (DSR):
   Correct Bailey & Lopez de Prado (2014) per-period (daily) formulation:
   - Uses daily returns, NOT annualized.
   - SR = mean_daily / std_daily.
   - Adjusts for non-normality (skew, non-excess kurtosis) and multiple testing across N trials.
   - Expected max SR under null (SR0) scales with cross-trial variance V.
   - Pass threshold >= 0.95 (or 0.99 if population overfit warning).

2. Probability of Backtest Overfitting (PBO):
   Real population-level Combinatorial Symmetric Cross-Validation (CSCV):
   - Aligned daily returns matrix of M >= 20 candidate strategies across T days.
   - 16 blocks, combinations of 8 IS / 8 OOS blocks.
   - Evaluates whether IS-best candidate falls below median in OOS (logit lambda <= 0).
   - Rejects / returns explicit non-pass when M < 20 (never fails open).
   - When PBO > 0.50, flags population_overfit_warning and recommends tightening DSR to 0.99.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


@dataclass
class DSRResult:
    passed: bool
    dsr_probability: float
    sr_daily: float
    expected_max_sr: float
    z_stat: float
    se_sr: float
    n_trials: int
    threshold: float
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_doc(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "dsr_probability": round(self.dsr_probability, 6),
            "sr_daily": round(self.sr_daily, 6),
            "expected_max_sr": round(self.expected_max_sr, 6),
            "z_stat": round(self.z_stat, 4),
            "n_trials": self.n_trials,
            "threshold": self.threshold,
            "detail": self.detail,
        }


def compute_deflated_sharpe_ratio(
    sr_daily: float,
    t_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    n_trials: int = 1,
    historical_daily_sharpes: list[float] | None = None,
    cfg: dict | None = None,
    dsr_threshold: float = 0.95,
) -> DSRResult:
    """Compute Deflated Sharpe Ratio (DSR) using per-period daily statistics.
    
    Formula:
      SR0 = sqrt(V) * ((1 - gamma)*Phi^-1(1 - 1/N) + gamma*Phi^-1(1 - 1/(N*e)))
      where gamma = 0.5772156649, SR0 = 0 when N = 1.
      V = variance of daily Sharpe estimates across trials (if < 10, V = 1/(T-1)).
      SE = sqrt((1 - skew*SR + ((kurtosis - 1)/4)*SR^2) / (T - 1))
      Z = (SR - SR0) / SE
      DSR = Phi(Z)
    """
    if cfg is not None:
        dsr_threshold = float(cfg.get("gates", {}).get("dsr", {}).get("min_probability", dsr_threshold))

    t = max(int(t_observations), 2)
    n = max(int(n_trials), 1)

    # 1. Variance of Sharpe estimates across trials (V)
    if historical_daily_sharpes and len(historical_daily_sharpes) >= 10:
        v = float(np.var(historical_daily_sharpes, ddof=1))
    else:
        v = 1.0 / (t - 1)

    # 2. Expected maximum Sharpe under null (SR0)
    if n > 1:
        gamma = 0.5772156649
        p1 = 1.0 - 1.0 / n
        p2 = 1.0 - 1.0 / (n * math.e)
        z1 = float(sp_stats.norm.ppf(p1))
        z2 = float(sp_stats.norm.ppf(p2))
        sr0 = math.sqrt(v) * ((1.0 - gamma) * z1 + gamma * z2)
    else:
        sr0 = 0.0

    # 3. Standard error of Sharpe ratio
    # kurtosis is Pearson non-excess kurtosis (normal distribution = 3.0)
    var_sr = (1.0 - skew * sr_daily + ((kurtosis - 1.0) / 4.0) * (sr_daily ** 2)) / (t - 1)
    se_sr = math.sqrt(max(var_sr, 1e-12))

    # 4. Z-statistic and probability
    z_stat = (sr_daily - sr0) / se_sr
    dsr_prob = float(sp_stats.norm.cdf(z_stat))

    passed = bool(dsr_prob >= dsr_threshold)

    detail = (
        f"DSR: prob={dsr_prob:.4f} (min={dsr_threshold:.2f}), "
        f"SR_daily={sr_daily:.4f}, SR0={sr0:.4f}, trials={n}, "
        f"{'PASS' if passed else 'FAIL: likely selection bias / multiple testing overfit'}"
    )

    data = {
        "sr_daily": sr_daily,
        "sr0": sr0,
        "v_variance": v,
        "se_sr": se_sr,
        "z_stat": z_stat,
        "dsr_prob": dsr_prob,
        "n_trials": n,
        "t_observations": t,
        "skew": skew,
        "kurtosis": kurtosis,
    }

    return DSRResult(
        passed=passed,
        dsr_probability=dsr_prob,
        sr_daily=sr_daily,
        expected_max_sr=sr0,
        z_stat=z_stat,
        se_sr=se_sr,
        n_trials=n,
        threshold=dsr_threshold,
        detail=detail,
        data=data,
    )


@dataclass
class PBOResult:
    passed: bool
    status: str
    pbo: float | None
    population_overfit_warning: bool
    recommended_dsr_threshold: float
    detail: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_doc(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status,
            "pbo": round(self.pbo, 4) if self.pbo is not None else None,
            "population_overfit_warning": self.population_overfit_warning,
            "recommended_dsr_threshold": self.recommended_dsr_threshold,
            "detail": self.detail,
            "details": self.details,
        }


def compute_population_pbo(
    returns_matrix: pd.DataFrame | np.ndarray,
    cfg: dict | None = None,
    min_candidates: int = 20,
    n_blocks: int = 16,
    n_combos: int = 2000,
    max_pbo: float = 0.50,
    seed: int = 42,
) -> PBOResult:
    """Combinatorial Symmetric Cross-Validation (CSCV) at population level.

    Input:
    - returns_matrix: T days x M candidate strategies matrix of daily returns.
    - min_candidates: Minimum required candidate strategies (spec: >= 20).
    - n_blocks: 16 contiguous blocks.
    - n_combos: Number of half/half combinations evaluated (sample up to 2000 of 12870).

    Returns PBOResult.
    """
    if cfg is not None:
        max_pbo = float(cfg.get("gates", {}).get("pbo", {}).get("max_pbo", max_pbo))

    if isinstance(returns_matrix, pd.DataFrame):
        arr = returns_matrix.values
    else:
        arr = np.asarray(returns_matrix)

    n_days, n_candidates = arr.shape

    if n_candidates < min_candidates:
        return PBOResult(
            passed=False,
            status="insufficient_candidates",
            pbo=None,
            population_overfit_warning=False,
            recommended_dsr_threshold=0.95,
            detail=f"Insufficient candidates for CSCV PBO (have {n_candidates}, need >={min_candidates}).",
            details={"candidates_count": n_candidates, "required_min": min_candidates},
        )

    if n_days < n_blocks * 2:
        return PBOResult(
            passed=False,
            status="insufficient_days",
            pbo=None,
            population_overfit_warning=False,
            recommended_dsr_threshold=0.95,
            detail=f"Insufficient time observations for CSCV ({n_days} days < {n_blocks*2}).",
            details={"days_count": n_days, "required_min": n_blocks * 2},
        )

    # 1. Partition days into n_blocks contiguous slices
    block_indices = np.array_split(np.arange(n_days), n_blocks)

    # 2. Select combinations of n_blocks // 2 for IS
    half_blocks = n_blocks // 2
    all_combos = list(itertools.combinations(range(n_blocks), half_blocks))
    rng = np.random.default_rng(seed)

    if len(all_combos) > n_combos:
        chosen_indices = rng.choice(len(all_combos), size=n_combos, replace=False)
        selected_combos = [all_combos[i] for i in chosen_indices]
    else:
        selected_combos = all_combos

    overfit_count = 0
    logits: list[float] = []

    for is_block_ids in selected_combos:
        is_block_set = set(is_block_ids)
        oos_block_ids = [b for b in range(n_blocks) if b not in is_block_set]

        # Gather day indices
        is_day_idx = np.concatenate([block_indices[b] for b in is_block_ids])
        oos_day_idx = np.concatenate([block_indices[b] for b in oos_block_ids])

        is_returns = arr[is_day_idx, :]
        oos_returns = arr[oos_day_idx, :]

        # Compute Sharpe across candidates
        is_mean = np.mean(is_returns, axis=0)
        is_std = np.std(is_returns, axis=0, ddof=1)
        is_sharpes = np.where(is_std > 1e-12, is_mean / is_std, 0.0)

        best_m = int(np.argmax(is_sharpes))

        oos_mean = np.mean(oos_returns, axis=0)
        oos_std = np.std(oos_returns, axis=0, ddof=1)
        oos_sharpes = np.where(oos_std > 1e-12, oos_mean / oos_std, 0.0)

        # Compute percentile rank omega of best_m in OOS
        best_oos_sharpe = oos_sharpes[best_m]
        # rank: fraction of candidates strictly worse than best_m
        strictly_worse = np.sum(oos_sharpes < best_oos_sharpe)
        ties = np.sum(oos_sharpes == best_oos_sharpe) - 1
        rank = (strictly_worse + 0.5 * max(ties, 0) + 1.0) / (n_candidates + 1.0)

        omega = min(max(rank, 1e-5), 1.0 - 1e-5)
        logit_lambda = math.log(omega / (1.0 - omega))
        logits.append(logit_lambda)

        if logit_lambda <= 0.0:
            overfit_count += 1

    total_evaluated = len(selected_combos)
    pbo = float(overfit_count / total_evaluated) if total_evaluated > 0 else 1.0
    passed = bool(pbo <= max_pbo)
    warning = bool(pbo > 0.50)
    rec_dsr = 0.99 if warning else 0.95

    detail = (
        f"CSCV PBO: {pbo*100:.1f}% ({overfit_count}/{total_evaluated} overfit combos, max={max_pbo*100:.0f}%). "
        f"{'PASS' if passed else 'FAIL: high population overfitting probability'}"
    )

    return PBOResult(
        passed=passed,
        status="computed",
        pbo=pbo,
        population_overfit_warning=warning,
        recommended_dsr_threshold=rec_dsr,
        detail=detail,
        details={
            "pbo": pbo,
            "overfit_count": overfit_count,
            "combinations_evaluated": total_evaluated,
            "candidates_count": n_candidates,
            "days_count": n_days,
            "median_logit_lambda": float(np.median(logits)) if logits else 0.0,
        },
    )
