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

        # Track past ideas and diversity system (Phase F9)
        self._past_ideas: list[str] = []
        self._past_failures: list[str] = []
        self._trial_count = 0
        from strategy.diversity import DiversityRegistry
        self._diversity = DiversityRegistry(cfg)

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
            spec = idea.get("spec", idea) if isinstance(idea, dict) else {}
            record.strategy_name = spec.get("name") or idea.get("name") or f"Strategy_{trial_id[:8]}"
            self._past_ideas.append(record.strategy_name)

            # Diversity Gate: Logic Fingerprint check before backtest
            dup_res = self._diversity.check_logic_fingerprint(record.strategy_name, idea)
            if dup_res.is_duplicate:
                self._trial_count = max(0, self._trial_count - 1)
                record.status = "rejected"
                record.rejected_at = "logic_duplicate"
                record.error_message = dup_res.reason
                self._past_failures.append(f"{record.strategy_name}: duplicate of {dup_res.duplicate_of}")
                return self._finish(record, t0)

            # ── Step 2: Generate code ───────────────────────────────────
            code = idea.get("code") if (isinstance(idea, dict) and idea.get("code")) else None
            if not code or "class Strategy" not in code:
                code = self._generate_code(idea, timeframe)
            record.source_code = code
            record.source_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

            # Diversity Gate: Code Structure check
            is_struct_dup, struct_dup_of, struct_sim = self._diversity.check_code_structure(code)
            if is_struct_dup:
                record.status = "rejected"
                record.rejected_at = "code_structure_duplicate"
                record.error_message = f"High AST structural similarity ({struct_sim:.2f}) with {struct_dup_of}"
                self._past_failures.append(f"{record.strategy_name}: structural duplicate of {struct_dup_of}")
                return self._finish(record, t0)

            # Diversity Gate: Semantic check
            is_sem_dup, sem_dup_of, sem_sim = self._diversity.check_semantic_similarity(idea)
            if is_sem_dup:
                record.status = "rejected"
                record.rejected_at = "semantic_duplicate"
                record.error_message = f"High semantic similarity ({sem_sim:.2f}) with {sem_dup_of}"
                self._past_failures.append(f"{record.strategy_name}: semantic duplicate of {sem_dup_of}")
                return self._finish(record, t0)

            # ── Step 3: Static scan ─────────────────────────────────────
            code = self._scan_and_fix(code, record)

            # ── Steps 4-8: Run Unified 19-Stage Gate Pipeline ───────────
            from core.splits import DataStore
            from validation.gates import run_pipeline
            store = DataStore(self._cfg)
            train_df = store.get_data(timeframe, "train")
            val_df = store.get_data(timeframe, "validation")

            params = spec.get("parameters") or idea.get("parameters")
            pipeline_res = run_pipeline(
                source=code,
                df_train=train_df,
                df_val=val_df,
                cfg=self._cfg,
                params=params,
                n_trials=max(self._get_trial_count(), 1),
            )

            record.train_result = result_to_doc(pipeline_res.train_result) if pipeline_res.train_result else None
            record.val_result = result_to_doc(pipeline_res.val_result) if pipeline_res.val_result else None
            record.train_gates = pipeline_res.to_doc()
            record.status = pipeline_res.status
            record.rejected_at = pipeline_res.stopped_at

            concept_family = spec.get("concept_family", idea.get("concept_family", "momentum"))
            cell = (
                str(concept_family).lower(),
                timeframe,
                str(spec.get("session_filter", idea.get("session", "all"))).lower(),
                str(idea.get("regime", "all")).lower(),
            )

            if not pipeline_res.all_passed:
                self._past_failures.append(f"{record.strategy_name}: failed {pipeline_res.stopped_at}")
                self._diversity.coverage_map.record_attempt(cell, passed=False)
                return self._finish(record, t0)

            # Diversity Gate: Behavioural redundancy check
            from strategy.diversity import check_behavioural_similarity
            cand_ret = pipeline_res.train_result.daily_returns if pipeline_res.train_result else None
            is_redundant = False
            if cand_ret is not None:
                for cid, cdata in self._diversity.strategies.items():
                    stored_ret = cdata.get("daily_returns")
                    if stored_ret is not None:
                        b_check = check_behavioural_similarity(cand_ret, stored_ret)
                        if b_check.is_redundant:
                            is_redundant = True
                            record.status = "redundant"
                            record.rejected_at = "behavioural_redundancy"
                            record.error_message = f"Redundant with {cid}: {b_check.reason}"
                            self._past_failures.append(f"{record.strategy_name}: redundant with {cid}")
                            break

            if is_redundant:
                self._diversity.coverage_map.record_attempt(cell, passed=False)
                return self._finish(record, t0)

            # Promoted to candidate (unproven)
            record.status = "candidate"
            self._diversity.register_strategy(
                record.strategy_name,
                spec=idea,
                source=code,
                daily_returns=cand_ret,
            )
            self._diversity.coverage_map.record_attempt(cell, passed=True)
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
        doc = record.to_doc()
        col = self._db["runs"]
        col.insert_one(doc)
        # Mirror to 'trials' collection for backward compatibility with dashboard counters
        try:
            self._db["trials"].insert_one(record.to_doc())
        except Exception as exc:
            _log.warning("could not mirror trial to trials collection: %s", exc)
        increment_trial_counter()

        if record.status in ("candidate", "candidate (unproven)", "survived"):
            update_leaderboard(record, self._cfg)

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
    """Compute the weighted robustness score on a 0 to 100 scale (GATE-7)."""
    from validation.gates import compute_robustness_score as val_robustness

    components: dict[str, Any] = {}
    if record.train_gates:
        gates_list = record.train_gates.get("gates", [])
        for g in gates_list:
            gate_name = g.get("name") or g.get("gate")
            g_data = g.get("details", g.get("data", {}))
            if gate_name == "dsr":
                components["dsr"] = float(g_data.get("probability", g_data.get("dsr_probability", 0.5)))
            elif gate_name in ("walk_forward", "walk_forward_stable"):
                components["walk_forward_win_pct"] = float(g_data.get("profitable_pct", g_data.get("win_pct", 0.65)))
                components["walk_forward_efficiency"] = float(g_data.get("efficiency", 0.50))
            elif gate_name in ("parameter_sensitivity", "parameter_plateau"):
                components["parameter_plateau_pct"] = float(g_data.get("profitable_pct", g_data.get("prof_pct", 0.70)))
                components["parameter_sharpe_ratio"] = float(g_data.get("sharpe_ratio_of_baseline", g_data.get("sharpe_ratio", 0.50)))
            elif gate_name in ("monte_carlo", "monte_carlo_stable"):
                components["monte_carlo_p95_dd"] = float(g_data.get("p95_dd", 0.20))
            elif gate_name in ("regime_and_year", "regime_concentration"):
                components["regime_concentration"] = float(g_data.get("max_regime_concentration", g_data.get("max_concentration", 0.40)))

    if record.train_result:
        metrics = record.train_result.get("metrics", {})
        cgr = float(metrics.get("cost_gross_ratio", 0.30))
        components["cost_resilience_pf"] = 1.0 + max(0.0, 0.40 - cgr)

    if record.delay_test:
        components["delay_sharpe_drop_pct"] = float(record.delay_test.get("sharpe_drop_pct", 20.0))

    return val_robustness(components, cfg)


def update_leaderboard(record: TrialRecord, cfg: dict) -> dict[str, Any]:
    """Add a promoted candidate strategy to the leaderboard and candidates in MongoDB."""
    if record.status not in ("candidate", "candidate (unproven)", "survived"):
        return {"error": "Only candidate strategies go on the leaderboard"}

    db = get_db(mongo_db_name(cfg))
    score = compute_robustness_score(record, cfg)

    train_metrics = {}
    if record.train_result and isinstance(record.train_result, dict):
        train_metrics = record.train_result.get("metrics", {})
    elif hasattr(record.train_result, "metrics"):
        train_metrics = record.train_result.metrics

    spec = record.idea.get("spec", record.idea) if isinstance(record.idea, dict) else {}
    concept_family = spec.get("concept_family", record.idea.get("concept_family", "trend"))

    entry = {
        "strategy_id": record.strategy_name or record.trial_id,
        "trial_id": record.trial_id,
        "name": record.strategy_name,
        "strategy_name": record.strategy_name,
        "timeframe": record.timeframe,
        "concept_family": concept_family,
        "status": "candidate (unproven)",
        "robustness_score": score,
        "train_metrics": train_metrics,
        "train_sharpe": float(train_metrics.get("sharpe", 0)),
        "train_pf": float(train_metrics.get("profit_factor", 0)),
        "val_sharpe": float(record.val_result.get("metrics", {}).get("sharpe", 0)) if record.val_result else 0.0,
        "train_trades": int(train_metrics.get("total_trades", 0)),
        "source_code": record.source_code,
        "source_hash": record.source_hash,
        "idea": record.idea,
        "train_result": record.train_result,
        "val_result": record.val_result,
        "train_gates": record.train_gates,
        "created_at": datetime.now(timezone.utc),
    }

    # Upsert into both 'candidates' (for web UI) and 'leaderboard' (for CLI & DB)
    db["candidates"].update_one(
        {"strategy_id": entry["strategy_id"]},
        {"$set": entry},
        upsert=True,
    )
    db["leaderboard"].update_one(
        {"trial_id": record.trial_id},
        {"$set": entry},
        upsert=True,
    )

    return entry
