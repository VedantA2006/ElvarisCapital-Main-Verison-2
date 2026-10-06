"""
llm/failure_analyzer.py – Comprehensive Strategy Autopsy and Failure Diagnostics.

Guarantees (Sections 24, 25, 26):
1. Produces a detailed StrategyAutopsy after every backtest evaluation.
2. Standardized 18-class taxonomy of primary and secondary failure modes.
3. Classifies repairability so Qwen is only prompted for genuinely improvable strategies.
4. Generates sanitized improvement guidance for the autonomous loop.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class FailureCategory(str, Enum):
    NO_TRADES = "NO_TRADES"
    TOO_FEW_TRADES = "TOO_FEW_TRADES"
    NEGATIVE_EXPECTANCY = "NEGATIVE_EXPECTANCY"
    LOW_PROFIT_FACTOR = "LOW_PROFIT_FACTOR"
    LOW_SHARPE = "LOW_SHARPE"
    HIGH_DRAWDOWN = "HIGH_DRAWDOWN"
    COST_SENSITIVE = "COST_SENSITIVE"
    PARAMETER_FRAGILE = "PARAMETER_FRAGILE"
    REGIME_FRAGILE = "REGIME_FRAGILE"
    LOOKAHEAD = "LOOKAHEAD"
    REPAINTING = "REPAINTING"
    NON_DETERMINISTIC = "NON_DETERMINISTIC"
    INVALID_CODE = "INVALID_CODE"
    SANDBOX_VIOLATION = "SANDBOX_VIOLATION"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    DATA_ERROR = "DATA_ERROR"
    OVERFIT = "OVERFIT"
    DUPLICATE = "DUPLICATE"


REPAIRABLE_FAILURES = {
    FailureCategory.TOO_FEW_TRADES,
    FailureCategory.LOW_PROFIT_FACTOR,
    FailureCategory.LOW_SHARPE,
    FailureCategory.HIGH_DRAWDOWN,
    FailureCategory.COST_SENSITIVE,
    FailureCategory.PARAMETER_FRAGILE,
    FailureCategory.REGIME_FRAGILE,
}


@dataclass
class StrategyAutopsy:
    strategy_id: str
    family: str
    status: str  # PASS, FAIL_REPAIRABLE, FAIL_FATAL
    primary_failure: str
    secondary_failures: List[str] = field(default_factory=list)
    is_repairable: bool = False
    is_promising: bool = False

    # Backtest Metrics
    total_trades: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_cost: float = 0.0
    cost_gross_ratio: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    trade_frequency_per_year: float = 0.0

    # Diagnostic Observations & Guidance
    diagnostic_notes: List[str] = field(default_factory=list)
    improvement_guidance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FailureAnalyzer:
    """Analyzes backtest results and gate evaluations to produce StrategyAutopsy reports."""

    def analyze(
        self,
        strategy_id: str,
        family: str,
        backtest_result: Any,
        gate_results: Optional[List[Any]] = None,
        min_required_trades: int = 80,
    ) -> StrategyAutopsy:
        """Construct StrategyAutopsy from execution and gate results."""
        metrics: Dict[str, Any] = {}
        if isinstance(backtest_result, dict):
            metrics = backtest_result.get("metrics", {})
        elif hasattr(backtest_result, "metrics"):
            metrics = getattr(backtest_result, "metrics", {})

        trades_count = int(metrics.get("total_trades", 0))
        net_pnl = float(metrics.get("net_pnl", 0.0))
        gross_pnl = float(metrics.get("gross_pnl", 0.0))
        total_cost = float(metrics.get("total_cost", 0.0))
        pf = float(metrics.get("profit_factor", 0.0))
        sharpe = float(metrics.get("sharpe", 0.0))
        sortino = float(metrics.get("sortino", 0.0))
        max_dd = float(metrics.get("max_drawdown", 0.0))
        win_rate = float(metrics.get("win_rate", 0.0))
        cgr = float(metrics.get("cost_gross_ratio", 0.0))
        trades_per_yr = float(metrics.get("trades_per_year", 0.0))

        primary_fail = ""
        sec_fails: List[str] = []
        notes: List[str] = []

        # 1. Zero trades or compile errors
        if trades_count == 0:
            primary_fail = FailureCategory.NO_TRADES
            notes.append("Strategy generated 0 trades across evaluation period.")
        elif trades_count < min_required_trades:
            primary_fail = FailureCategory.TOO_FEW_TRADES
            notes.append(f"Trade count {trades_count} below statistical sample floor ({min_required_trades}).")

        # 2. Financial quality failures
        elif cgr > 0.40:
            primary_fail = FailureCategory.COST_SENSITIVE
            notes.append(f"Friction drag ({cgr:.1%}) absorbed more than 40% of gross profits.")
            if pf < 1.25:
                sec_fails.append(FailureCategory.LOW_PROFIT_FACTOR)
        elif pf < 1.0:
            primary_fail = FailureCategory.NEGATIVE_EXPECTANCY
            notes.append(f"Profit factor {pf:.2f} indicates negative expectancy.")
        elif pf < 1.25:
            primary_fail = FailureCategory.LOW_PROFIT_FACTOR
            notes.append(f"Profit factor {pf:.2f} below target hurdle 1.25.")
        elif sharpe < 0.80:
            primary_fail = FailureCategory.LOW_SHARPE
            notes.append(f"Sharpe ratio {sharpe:.2f} below institutional target 0.80.")
        elif max_dd > 0.25:
            primary_fail = FailureCategory.HIGH_DRAWDOWN
            notes.append(f"Max drawdown {max_dd:.1%} exceeded 25% risk ceiling.")

        # Inspect gate failures if present
        if gate_results:
            for g in gate_results:
                g_passed = getattr(g, "passed", True) if not isinstance(g, dict) else g.get("passed", True)
                g_name = getattr(g, "name", "") if not isinstance(g, dict) else g.get("name", "")
                if not g_passed:
                    if g_name == "determinism":
                        primary_fail = FailureCategory.NON_DETERMINISTIC
                    elif g_name in ("policy_scan", "truncation"):
                        primary_fail = FailureCategory.LOOKAHEAD
                    elif g_name == "parameter_sensitivity":
                        sec_fails.append(FailureCategory.PARAMETER_FRAGILE)
                    elif g_name == "regime_and_year":
                        sec_fails.append(FailureCategory.REGIME_FRAGILE)

        if not primary_fail:
            status = "PASS"
            is_promising = True
            is_repairable = False
            guidance = "Strategy passed basic backtest hurdles. Ready for adversarial robustness validation."
        else:
            is_repairable = primary_fail in REPAIRABLE_FAILURES
            is_promising = (pf >= 1.05 and trades_count >= 50 and primary_fail != FailureCategory.LOOKAHEAD)
            status = "FAIL_REPAIRABLE" if is_repairable else "FAIL_FATAL"
            guidance = self._formulate_guidance(primary_fail, sec_fails, cgr, pf, trades_count)

        return StrategyAutopsy(
            strategy_id=strategy_id,
            family=family,
            status=status,
            primary_failure=primary_fail or "NONE",
            secondary_failures=sec_fails,
            is_repairable=is_repairable,
            is_promising=is_promising,
            total_trades=trades_count,
            gross_pnl=round(gross_pnl, 2),
            net_pnl=round(net_pnl, 2),
            total_cost=round(total_cost, 2),
            cost_gross_ratio=round(cgr, 4),
            profit_factor=round(pf, 3),
            sharpe=round(sharpe, 3),
            sortino=round(sortino, 3),
            max_drawdown=round(max_dd, 4),
            win_rate=round(win_rate, 4),
            trade_frequency_per_year=round(trades_per_yr, 1),
            diagnostic_notes=notes,
            improvement_guidance=guidance,
        )

    def _formulate_guidance(
        self,
        primary_fail: str,
        sec_fails: List[str],
        cgr: float,
        pf: float,
        trades: int,
    ) -> str:
        """Generate actionable directions for Qwen without exposing holdout numbers."""
        if primary_fail == FailureCategory.COST_SENSITIVE:
            return (
                "Strategy is severely dragged by transaction costs. "
                "Do NOT increase trade frequency. "
                "Increase average holding duration, widen stop loss and take profit multiples (e.g. SL >= 1.8 ATR, TP >= 2.5 ATR), "
                "or restrict entries to high-volatility session expansions to overcome the spread."
            )
        elif primary_fail in (FailureCategory.TOO_FEW_TRADES, FailureCategory.NO_TRADES):
            return (
                "Strategy entry criteria are excessively restrictive, starving the sample size. "
                "Reduce the number of simultaneous indicator filters (e.g. from 4 indicators to 2 core conditions). "
                "Widen session participation or relax thresholds to capture valid trend continuation waves."
            )
        elif primary_fail == FailureCategory.LOW_PROFIT_FACTOR:
            return (
                f"Gross edge exists but profit factor ({pf:.2f}) lacks sufficient margin. "
                "Refine the asymmetry of the trade: improve entry timing via pullbacks to structure (FVG or EMA) "
                "and widen the reward-to-risk ratio to at least 2:1."
            )
        elif primary_fail == FailureCategory.HIGH_DRAWDOWN:
            return (
                "Drawdown is excessive. Add a volatility filter (e.g. avoid entries during extreme ATR contraction) "
                "or incorporate a breakeven trailing rule to protect floating profits."
            )
        return "Modify entry logic to improve edge asymmetry while maintaining sufficient sample size."
