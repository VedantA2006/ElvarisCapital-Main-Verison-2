"""
llm/improve_loop.py – Autonomous strategy improvement and structural refinement loop.

Takes candidates that completed in-sample train runs, feeds train diagnostics
back to the LLM, asks for ONE targeted structural refinement, validates the new
candidate, increments the global trial counter, and links parentage.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from storage import mongo
from llm.client import LLMClient
from llm.diagnostics import build_train_diagnostics, build_train_diagnostics_from_trades
from llm.prompts import render_code_fix_prompt, render_improve_prompt, render_rethink_prompt
from llm.schemas import StrategyImproveResponse, StrategyResponse, StrategySpec

_log = logging.getLogger("quantforge.llm.improve")


@dataclass
class EvalOutcome:
    """Outcome of evaluating a strategy version."""

    status: str = "ok"  # "ok", "robustness_failed", "validation_failed"
    train_metrics: dict[str, Any] = field(default_factory=dict)
    train_trades: list[Any] = field(default_factory=list)
    category: str | None = None
    private_details: dict[str, Any] | None = None


@dataclass
class VersionRecord:
    """Record of one evaluated strategy version in the improve lineage."""

    strategy_id: str
    root_id: str
    parent_id: str | None
    version: int
    spec: dict[str, Any]
    code: str
    score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    fix_attempts: int = 0
    refinement_reason: str = ""
    trial_id: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ImproveLoopResult:
    """Complete outcome of the autonomous improvement loop."""

    root_id: str
    versions: list[VersionRecord]
    best: VersionRecord | None
    stop_reason: str
    public_feedback: str | None = None


class StrategyImproveLoop:
    """Manages the autonomous feedback-and-improve cycle for strategy evolution."""

    def __init__(self, cfg: dict, client: LLMClient, db=None):
        self._cfg = cfg
        self._llm_cfg = cfg.get("llm", {})
        self._client = client
        self._db = db if db is not None else mongo.get_db()
        self._max_improve_rounds = self._llm_cfg.get("max_improve_rounds", 5)
        self._max_code_fix_attempts = self._llm_cfg.get("max_code_fix_attempts", 3)
        self._code_fix_temperature = self._llm_cfg.get("code_fix_temperature", 0.2)
        self._ideation_temperature = self._llm_cfg.get("ideation_temperature", 0.9)

    def improve_strategy(
        self,
        parent_strategy_id: str,
        parent_version: int,
        parent_spec: dict[str, Any],
        parent_code: str,
        train_diagnostics: str,
    ) -> dict[str, Any]:
        """Execute one structural refinement round on an existing strategy."""
        _log.info(
            "Starting single improve cycle for parent %s (version %d)",
            parent_strategy_id,
            parent_version,
        )

        spec_obj = StrategySpec.model_validate(parent_spec)
        messages = render_improve_prompt(
            spec=spec_obj,
            code=parent_code,
            diagnostics=train_diagnostics,
        )

        res: StrategyImproveResponse = self._client.chat_and_validate(
            messages=messages,
            schema=StrategyImproveResponse,
            purpose="improve",
            strategy_id=parent_strategy_id,
        )

        child_version = parent_version + 1
        child_id = f"{parent_strategy_id}_v{child_version}"
        trial_id = mongo.increment_trial_counter()
        now = datetime.now(timezone.utc)

        child_data = {
            "strategy_id": child_id,
            "parent_id": parent_strategy_id,
            "version": child_version,
            "spec": res.spec.model_dump(),
            "code": res.code,
            "refinement_reason": res.refinement_reason,
            "trial_id": trial_id,
            "status": "candidate",
            "created_at": now,
        }

        try:
            self._db["strategies"].insert_one(dict(child_data))
        except Exception as exc:
            _log.error("Failed to persist improved strategy %s: %s", child_id, exc)

        return child_data

    def _eval_with_code_fix(
        self,
        spec: dict[str, Any],
        code: str,
        evaluate_fn: Callable[[dict[str, Any], str], EvalOutcome],
    ) -> tuple[EvalOutcome | None, str, int, bool]:
        """Evaluate strategy code, retrying with fix_code up to max_code_fix_attempts on error."""
        import traceback

        current_code = code
        fix_attempts = 0

        while True:
            try:
                outcome = evaluate_fn(spec, current_code)
                return outcome, current_code, fix_attempts, True
            except Exception as exc:
                if fix_attempts >= self._max_code_fix_attempts:
                    _log.warning("Code error could not be fixed after %d attempts: %s", fix_attempts, exc)
                    return None, current_code, fix_attempts, False

                fix_attempts += 1
                tb = traceback.format_exc()
                err_text = tb if ("Traceback" in tb and type(exc).__name__ in tb) else f"{type(exc).__name__}: {exc}"
                _log.info("Code execution error (attempt %d/%d): %s", fix_attempts, self._max_code_fix_attempts, err_text)
                messages = render_code_fix_prompt(code=current_code, error=err_text, spec=spec)
                try:
                    fix_res = self._client.chat_and_validate(
                        messages=messages,
                        schema=StrategyResponse,
                        purpose="fix_code",
                        temperature=self._code_fix_temperature,
                    )
                    current_code = fix_res.code
                except Exception as fix_exc:
                    _log.warning("fix_code LLM call failed: %s", fix_exc)

    def run(
        self,
        root_id: str,
        spec: dict[str, Any],
        code: str,
        evaluate_fn: Callable[[dict[str, Any], str], EvalOutcome],
    ) -> ImproveLoopResult:
        """Run the full autonomous improvement loop on a candidate strategy."""
        versions: list[VersionRecord] = []
        best_record: VersionRecord | None = None
        best_sharpe: float = -999.0
        best_exp: float = -999.0
        consecutive_no_improvement = 0
        rethink_attempted = False
        stop_reason = ""
        public_feedback: str | None = None

        current_spec = dict(spec)
        current_code = str(code)
        current_version = 1
        parent_id: str | None = None
        current_reason = ""

        while True:
            # 1. Evaluate with code fix retry if necessary
            outcome, final_code, fix_attempts, success = self._eval_with_code_fix(
                current_spec, current_code, evaluate_fn
            )

            if not success or outcome is None:
                # Failed due to unfixable code error
                strat_id = f"{root_id}_v{current_version}"
                v_rec = VersionRecord(
                    strategy_id=strat_id,
                    root_id=root_id,
                    parent_id=parent_id,
                    version=current_version,
                    spec=current_spec,
                    code=final_code,
                    score=-999.0,
                    metrics={},
                    fix_attempts=fix_attempts,
                    refinement_reason=current_reason,
                    trial_id=None if current_version == 1 else mongo.get_trial_count(),
                )
                versions.append(v_rec)
                return ImproveLoopResult(
                    root_id=root_id,
                    versions=versions,
                    best=best_record or v_rec,
                    stop_reason="code_error_unfixed",
                    public_feedback=public_feedback,
                )

            current_code = final_code
            metrics = outcome.train_metrics or {}
            cur_sharpe = float(metrics.get("sharpe", 0.0))
            cur_exp = float(metrics.get("expectancy_per_trade", 0.0))
            score = cur_sharpe if cur_sharpe != 0.0 else cur_exp
            strat_id = f"{root_id}_v{current_version}"

            v_rec = VersionRecord(
                strategy_id=strat_id,
                root_id=root_id,
                parent_id=parent_id,
                version=current_version,
                spec=current_spec,
                code=current_code,
                score=score,
                metrics=metrics,
                fix_attempts=fix_attempts,
                refinement_reason=current_reason,
                trial_id=None if current_version == 1 else mongo.get_trial_count(),
            )
            versions.append(v_rec)

            # Persist version to DB
            try:
                doc = asdict(v_rec)
                self._db["strategies"].insert_one(doc)
            except Exception as exc:
                _log.debug("could not insert version record: %s", exc)

            # Check improvement on train Sharpe OR expectancy per trade
            if best_record is None:
                best_record = v_rec
                best_sharpe = cur_sharpe
                best_exp = cur_exp
            else:
                if cur_sharpe > best_sharpe + 1e-6 or cur_exp > best_exp + 1e-6:
                    best_record = v_rec
                    best_sharpe = max(best_sharpe, cur_sharpe)
                    best_exp = max(best_exp, cur_exp)
                    consecutive_no_improvement = 0
                else:
                    consecutive_no_improvement += 1

            # 2. Check outcome status: validation_failed
            if outcome.status == "validation_failed":
                public_feedback = f"validation failed: {outcome.category or 'unspecified'}"
                stop_reason = "validation_complete"
                break

            # 3. Check outcome status: robustness_failed
            if outcome.status == "robustness_failed":
                if rethink_attempted:
                    stop_reason = "robustness_rethink_exhausted"
                    break

                rethink_attempted = True
                _log.info("Robustness failed (%s), requesting ONE structural rethink", outcome.category)
                summary = f"failed robustness: {outcome.category or 'robustness_gate'}"
                spec_obj = StrategySpec.model_validate(current_spec)
                messages = render_rethink_prompt(spec=spec_obj, code=current_code, public_summary=summary)

                try:
                    rethink_res: StrategyImproveResponse = self._client.chat_and_validate(
                        messages=messages,
                        schema=StrategyImproveResponse,
                        purpose="rethink",
                        temperature=self._code_fix_temperature,
                    )
                    parent_id = strat_id
                    current_version += 1
                    current_spec = rethink_res.spec.model_dump()
                    current_code = rethink_res.code
                    current_reason = rethink_res.refinement_reason
                    mongo.increment_trial_counter()
                    continue
                except Exception as rethink_exc:
                    _log.error("Failed to generate rethink from LLM: %s", rethink_exc)
                    stop_reason = "robustness_rethink_exhausted"
                    break

            # 4. Outcome is OK: check stop conditions
            # Notice: round count of improvements made is len(versions) - 1
            improve_rounds_done = len(versions) - 1
            if improve_rounds_done >= self._max_improve_rounds:
                stop_reason = "max_rounds"
                break

            if consecutive_no_improvement >= 2:
                stop_reason = "no_improvement"
                break

            # 5. Build train diagnostics & prompt LLM for improvement
            if outcome.train_trades:
                diag = build_train_diagnostics_from_trades(
                    outcome.train_trades, outcome.train_metrics, cfg=self._cfg, split="train"
                )
            else:
                diag = build_train_diagnostics(outcome.train_metrics)

            spec_obj = StrategySpec.model_validate(current_spec)
            messages = render_improve_prompt(spec=spec_obj, code=current_code, diagnostics=diag)

            try:
                imp_res: StrategyImproveResponse = self._client.chat_and_validate(
                    messages=messages,
                    schema=StrategyImproveResponse,
                    purpose="improve",
                    temperature=self._code_fix_temperature,
                )
                parent_id = strat_id
                current_version += 1
                current_spec = imp_res.spec.model_dump()
                current_code = imp_res.code
                current_reason = imp_res.refinement_reason
                mongo.increment_trial_counter()
            except Exception as imp_exc:
                _log.error("Failed to generate improvement from LLM: %s", imp_exc)
                stop_reason = "llm_error"
                break

        return ImproveLoopResult(
            root_id=root_id,
            versions=versions,
            best=best_record,
            stop_reason=stop_reason,
            public_feedback=public_feedback,
        )
