"""
llm/improve_loop.py – Autonomous strategy improvement and structural refinement loop.

Takes candidates that completed in-sample train runs, feeds train diagnostics
back to the LLM, asks for ONE targeted structural refinement, validates the new
candidate, increments the global trial counter, and links parentage.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from storage import mongo
from llm.client import LLMClient
from llm.prompts import render_improve_prompt
from llm.schemas import StrategyImproveResponse, StrategySpec

_log = logging.getLogger("quantforge.llm.improve")


class StrategyImproveLoop:
    """Manages the autonomous feedback-and-improve cycle for strategy evolution."""

    def __init__(self, cfg: dict, client: LLMClient, db=None):
        self._cfg = cfg
        self._llm_cfg = cfg.get("llm", {})
        self._client = client
        self._db = db if db is not None else mongo.get_db()
        self._max_improve_rounds = self._llm_cfg.get("max_improve_rounds", 5)

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
            "Starting improve cycle for parent %s (version %d)",
            parent_strategy_id,
            parent_version,
        )

        # Parse parent spec into Pydantic model for template rendering
        spec_obj = StrategySpec.model_validate(parent_spec)
        messages = render_improve_prompt(
            spec=spec_obj,
            code=parent_code,
            diagnostics=train_diagnostics,
        )

        # Call LLM with StrategyImproveResponse schema validation
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

        # Persist child candidate into strategies collection
        try:
            self._db["strategies"].insert_one(dict(child_data))
        except Exception as exc:
            _log.error("Failed to persist improved strategy %s: %s", child_id, exc)

        _log.info("Strategy %s successfully produced child %s (Trial #%d)", parent_strategy_id, child_id, trial_id)
        return child_data
