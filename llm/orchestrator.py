"""
llm/orchestrator.py – Autonomous strategy discovery loop.

Flow per trial:
  1. IDEATE  – LLM invents a strategy concept (JSON)
  2. GENERATE – LLM writes code implementing it
  3. SCAN    – Static AST scan for lookahead (retry with feedback if fail)
  4. LOAD    – Sandbox loads and validates the code
  5. BACKTEST – Run on TRAIN split with full cost model
  6. GATE    – Run gates 1-3, 6, 7, 9, 10 (train gates)
  7. VALIDATE – Backtest on VALIDATION split, run gate 8
  8. ADVANCED – Delay test, truncation test
  9. LOG     – Record everything to MongoDB
  10. RANK   – Update leaderboard

Most strategies SHOULD be rejected. That is the design goal.
"""

from __future__ import annotations

import hashlib
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from core.backtester import run_backtest, result_to_doc
from core.config import load_config
from core.gates import (
    gate_monte_carlo, gate_regime_concentration, gate_train_performance,
    gate_validation, gate_dsr, gate_pbo, GateResult, run_train_gates,
)
from core.lookahead_guard import (
    run_delay_test, run_truncation_test, static_scan,
)
from core.sandbox import Sandbox, SandboxError
from llm.client import LLMClient, LLMError, LLMRateLimitError
from llm.prompts import (
    ideation_prompt, code_generation_prompt,
    code_fix_prompt, lookahead_fix_prompt,
)
from storage.mongo import get_db
from core.config import mongo_db_name

from scipy import stats as sp_stats

import logging
_log = logging.getLogger("quantforge.orchestrator")


@dataclass
class TrialRecord:
    """Complete record of one strategy trial."""
    trial_id: str
    strategy_name: str
    timeframe: str
    idea: dict[str, Any]
    source_code: str
    source_hash: str

    # Results (filled as pipeline progresses)
    static_scan: dict[str, Any] | None = None
    train_result: dict[str, Any] | None = None
    train_gates: dict[str, Any] | None = None
    val_result: dict[str, Any] | None = None
    val_gate: dict[str, Any] | None = None
    delay_test: dict[str, Any] | None = None
    truncation_test: dict[str, Any] | None = None

    # Outcome
    status: str = "pending"  # pending, rejected, survived, error
    rejected_at: str | None = None
    error_message: str | None = None

    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    wall_seconds: float = 0.0

    # LLM usage
    llm_calls: int = 0
    llm_tokens: int = 0

    def to_doc(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if v is not None}
        d["started_at"] = self.started_at
        d["finished_at"] = self.finished_at
        return _stringify_keys(d)


def _stringify_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_keys(x) for x in obj]
    return obj


class Orchestrator:
    """Autonomous strategy discovery engine."""

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            from core.config import load_env
            load_env()
            cfg = load_config()
        self._cfg = cfg
        self._llm = LLMClient(cfg)
        self._sandbox = Sandbox(cfg)
        self._db = get_db(mongo_db_name(cfg))

        # Track past ideas for diversity
        self._past_ideas: list[str] = []
        self._past_failures: list[str] = []
        self._trial_count = 0

    def run_trial(self, timeframe: str = "1h") -> TrialRecord:
        """Run a single trial: ideate → generate → validate → backtest → gate."""
        trial_id = f"trial-{uuid.uuid4().hex[:12]}"
        t0 = time.perf_counter()

        record = TrialRecord(
            trial_id=trial_id,
            strategy_name="",
            timeframe=timeframe,
            idea={},
            source_code="",
            source_hash="",
        )

        try:
            # ── Step 1: Ideate ──────────────────────────────────────────
            idea = self._ideate(timeframe)
            record.idea = idea
            record.strategy_name = idea.get("name", "unnamed")
            self._past_ideas.append(record.strategy_name)

            # ── Step 2: Generate code ───────────────────────────────────
            code = self._generate_code(idea, timeframe)
            record.source_code = code
            record.source_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

            # ── Step 3: Static scan ─────────────────────────────────────
            code = self._scan_and_fix(code, record)

            # ── Step 4: Sandbox load ────────────────────────────────────
            strategy_obj = self._sandbox.load_strategy(code)

            # ── Step 5: Backtest on TRAIN ───────────────────────────────
            from core.splits import DataStore
            store = DataStore(self._cfg)
            train_df = store.get_data(timeframe, "train")
            train_result = run_backtest(strategy_obj, train_df, self._cfg)
            record.train_result = result_to_doc(train_result)

            # ── Step 6: Train gates ─────────────────────────────────────
            trade_pnls = [t.net_pnl for t in train_result.trades]
            n_trials = self._get_trial_count()
            gate_result = run_train_gates(
                train_result.metrics, trade_pnls, self._cfg,
                timeframe=timeframe, n_trials=max(n_trials, 1),
            )
            record.train_gates = gate_result.to_doc()

            if not gate_result.all_passed:
                record.status = "rejected"
                record.rejected_at = gate_result.stopped_at
                self._past_failures.append(
                    f"{record.strategy_name}: failed {gate_result.stopped_at}")
                return self._finish(record, t0)

            # ── Step 7: Validation backtest + gate 8 ────────────────────
            # Need a fresh strategy instance for validation
            strategy_val = self._sandbox.load_strategy(code)
            val_df = store.get_data(timeframe, "validation")
            val_result = run_backtest(strategy_val, val_df, self._cfg)
            record.val_result = result_to_doc(val_result)

            val_gate = gate_validation(
                val_result.metrics, train_result.metrics, self._cfg)
            record.val_gate = val_gate.to_doc()

            if not val_gate.passed:
                record.status = "rejected"
                record.rejected_at = "validation"
                self._past_failures.append(
                    f"{record.strategy_name}: failed validation")
                return self._finish(record, t0)

            # ── Step 8: Advanced tests ──────────────────────────────────
            # Delay test
            def strat_factory():
                return self._sandbox.load_strategy(code)

            delay = run_delay_test(strat_factory, train_df, self._cfg)
            record.delay_test = delay.to_doc()
            if not delay.passed:
                record.status = "rejected"
                record.rejected_at = "delay_test"
                self._past_failures.append(
                    f"{record.strategy_name}: failed delay test")
                return self._finish(record, t0)

            # Truncation test
            trunc = run_truncation_test(strat_factory, train_df, self._cfg, num_cuts=10)
            record.truncation_test = trunc.to_doc()
            if not trunc.passed:
                record.status = "rejected"
                record.rejected_at = "truncation_test"
                self._past_failures.append(
                    f"{record.strategy_name}: failed truncation test")
                return self._finish(record, t0)

            # ── SURVIVED! ──────────────────────────────────────────────
            record.status = "survived"
            return self._finish(record, t0)

        except LLMRateLimitError as e:
            record.status = "error"
            record.error_message = f"Rate limit: {e}"
            return self._finish(record, t0)
        except SandboxError as e:
            record.status = "rejected"
            record.rejected_at = "sandbox"
            record.error_message = str(e)
            self._past_failures.append(
                f"{record.strategy_name}: sandbox error")
            return self._finish(record, t0)
        except Exception as e:
            record.status = "error"
            record.error_message = f"{type(e).__name__}: {e}"
            return self._finish(record, t0)

    def run_loop(self, max_trials: int = 50, timeframes: list[str] | None = None,
                 stop_on_survivor: bool = False) -> list[TrialRecord]:
        """Run the discovery loop for up to max_trials."""
        if timeframes is None:
            timeframes = self._cfg["data"]["enabled_timeframes"]

        records: list[TrialRecord] = []
        tf_idx = 0

        for i in range(max_trials):
            tf = timeframes[tf_idx % len(timeframes)]
            tf_idx += 1

            print(f"\n{'='*60}")
            print(f"Trial {i+1}/{max_trials} | TF={tf} | "
                  f"Survived: {sum(1 for r in records if r.status == 'survived')} | "
                  f"Rejected: {sum(1 for r in records if r.status == 'rejected')}")
            print(f"{'='*60}")

            record = self.run_trial(tf)
            records.append(record)

            status_icon = {"survived": "[SURVIVED]", "rejected": "[REJECTED]",
                           "error": "[ERROR]"}.get(record.status, "[???]")
            print(f"{status_icon} {record.strategy_name} "
                  f"({record.rejected_at or record.status}) "
                  f"[{record.wall_seconds:.1f}s]")

            if stop_on_survivor and record.status == "survived":
                print("\nSurvivor found! Stopping loop.")
                break

            # Cooldown between trials
            time.sleep(2)

        return records

    # ── Internal methods ────────────────────────────────────────────────

    def _ideate(self, timeframe: str) -> dict[str, Any]:
        """Ask LLM for a strategy idea."""
        messages = ideation_prompt(
            timeframe,
            past_ideas=self._past_ideas[-20:],
            past_failures=self._past_failures[-10:],
        )
        resp = self._llm.chat(messages, temperature=self._cfg["llm"]["ideation_temperature"])
        self._trial_count += 1
        idea = self._llm.extract_json(resp.content)
        return idea

    def _generate_code(self, idea: dict, timeframe: str) -> str:
        """Ask LLM to write strategy code."""
        messages = code_generation_prompt(idea, timeframe)
        resp = self._llm.chat(messages, temperature=self._cfg["llm"]["code_fix_temperature"])
        code = self._llm.extract_code(resp.content)
        return code

    def _scan_and_fix(self, code: str, record: TrialRecord) -> str:
        """Static scan + auto-fix loop."""
        max_attempts = self._cfg["llm"]["max_lookahead_fix_attempts"]

        for attempt in range(max_attempts + 1):
            scan = static_scan(code)
            record.static_scan = scan.to_doc()

            if scan.passed:
                return code

            if attempt >= max_attempts:
                raise SandboxError(
                    f"Static scan failed after {max_attempts} fix attempts: {scan.summary}")

            # Ask LLM to fix
            violations = [{"line": v.line, "detail": v.detail}
                          for v in scan.violations if v.severity == "error"]
            messages = lookahead_fix_prompt(code, violations)
            resp = self._llm.chat(messages, temperature=self._cfg["llm"]["code_fix_temperature"])
            code = self._llm.extract_code(resp.content)
            record.source_code = code
            record.source_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
            record.llm_calls += 1

        return code

    def _finish(self, record: TrialRecord, t0: float) -> TrialRecord:
        """Finalise and persist a trial record."""
        record.finished_at = datetime.now(timezone.utc)
        record.wall_seconds = round(time.perf_counter() - t0, 2)

        # Persist to MongoDB — errors MUST surface (ORCH-3).
        from storage.mongo import increment_trial_counter
        col = self._db["runs"]
        col.insert_one(record.to_doc())
        increment_trial_counter()

        self._trial_count += 1
        return record

    def _get_trial_count(self) -> int:
        """Get total trials from MongoDB for DSR calculation."""
        from storage.mongo import get_trial_count
        try:
            return get_trial_count()
        except Exception:
            _log.warning("could not read trial count from Mongo, using in-memory count")
            return self._trial_count


# ─── Leaderboard ────────────────────────────────────────────────────────────

def compute_robustness_score(record: TrialRecord, cfg: dict) -> float:
    """Compute the weighted robustness score (Section 11)."""
    w = cfg["robustness_weights"]
    score = 0.0

    # DSR
    if record.train_gates:
        for g in record.train_gates.get("gates", []):
            if g["gate"] == "dsr" and g["passed"]:
                prob = g["data"].get("probability", 0)
                score += w["dsr"] * min(prob / 0.99, 1.0)

    # Walk-forward (from train gates if available)
    if record.train_gates:
        for g in record.train_gates.get("gates", []):
            if g["gate"] == "walk_forward" and g["passed"]:
                pct = g["data"].get("profitable_pct", 0)
                score += w["walk_forward"] * min(pct / 0.8, 1.0)

    # Monte Carlo
    if record.train_gates:
        for g in record.train_gates.get("gates", []):
            if g["gate"] == "monte_carlo" and g["passed"]:
                score += w["monte_carlo_5th"] * 1.0

    # Parameter plateau
    if record.train_gates:
        for g in record.train_gates.get("gates", []):
            if g["gate"] == "parameter_sensitivity" and g["passed"]:
                score += w["parameter_plateau"] * 1.0

    # Regime breadth
    if record.train_gates:
        for g in record.train_gates.get("gates", []):
            if g["gate"] == "regime_concentration" and g["passed"]:
                conc = g["data"].get("max_concentration", 1.0)
                score += w["regime_breadth"] * (1.0 - conc)

    # Cost resilience
    if record.train_result:
        cgr = record.train_result.get("metrics", {}).get("cost_gross_ratio", 1.0)
        score += w["cost_resilience"] * max(0, 1.0 - cgr / 0.4)

    # Delay resilience
    if record.delay_test:
        drop = record.delay_test.get("sharpe_drop_pct", 100)
        score += w["delay_resilience"] * max(0, 1.0 - drop / 60.0)

    # Holdout (only after holdout is run)
    # score += w["holdout_result"] * ...

    return round(min(score, 1.0), 4)


def update_leaderboard(record: TrialRecord, cfg: dict) -> dict[str, Any]:
    """Add a survived strategy to the leaderboard in MongoDB."""
    if record.status != "survived":
        return {"error": "Only survived strategies go on the leaderboard"}

    db = get_db(mongo_db_name(cfg))
    score = compute_robustness_score(record, cfg)

    entry = {
        "trial_id": record.trial_id,
        "strategy_name": record.strategy_name,
        "timeframe": record.timeframe,
        "robustness_score": score,
        "train_sharpe": record.train_result.get("metrics", {}).get("sharpe", 0) if record.train_result else 0,
        "train_pf": record.train_result.get("metrics", {}).get("profit_factor", 0) if record.train_result else 0,
        "val_sharpe": record.val_result.get("metrics", {}).get("sharpe", 0) if record.val_result else 0,
        "train_trades": record.train_result.get("metrics", {}).get("total_trades", 0) if record.train_result else 0,
        "source_hash": record.source_hash,
        "created_at": datetime.now(timezone.utc),
    }

    db["leaderboard"].update_one(
        {"trial_id": record.trial_id},
        {"$set": entry},
        upsert=True,
    )

    return entry
