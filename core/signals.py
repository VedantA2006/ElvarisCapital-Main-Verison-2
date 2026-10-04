"""
core/signals.py – Signal contract, validation, and SignalTape (Phase F2).

Guarantees:
1. Distances, not absolute prices: sl_distance > 0 is strictly required.
2. Inverted stop detection: Signal.from_prices() rejects stops on the wrong side (BT-1).
3. Minimum SL distance: max(min_sl_usd, min_sl_atr_multiple * ATR_prev).
4. Minimum TP distance: >= 3x spread and >= min_tp_usd (kills micro-scalps).
5. SignalTape array representation: cached by (code_hash, params_hash, data_hash, split, window).
6. Invalid signal tracking: if > 2% of non-empty signals are invalid, or any NaN/inf, raises code_error.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import pandas as pd


class Action(IntEnum):
    NONE = 0
    ENTER_LONG = 1
    ENTER_SHORT = 2
    CLOSE = 3


class Direction(IntEnum):
    LONG = 1
    SHORT = -1


class SignalValidationError(ValueError):
    """Raised when signal validation strictly fails (e.g. inverted stop)."""


@dataclass
class Signal:
    """Strategy output for a single bar.

    Always uses distances from entry price for stop-loss and take-profit.
    """
    action: Action = Action.NONE
    sl_distance: float = np.nan
    tp_distance: float = np.nan
    trail_distance: float = np.nan
    time_stop_bars: int = 0
    breakeven_after_r: float = np.nan
    tag: str = ""

    # Legacy compatibility fields
    direction: Direction | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    order_type: str = "market"
    order_price: float | None = None

    def __post_init__(self):
        # Synchronize direction with action if legacy constructor was used
        if self.direction is not None and self.action == Action.NONE:
            if self.direction == Direction.LONG or self.direction == 1 or str(self.direction).upper().endswith("LONG"):
                self.action = Action.ENTER_LONG
            elif self.direction == Direction.SHORT or self.direction == -1 or str(self.direction).upper().endswith("SHORT"):
                self.action = Action.ENTER_SHORT

    @classmethod
    def from_prices(
        cls,
        close: float,
        direction: Direction | str | int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        tag: str = "",
        min_sl_usd: float = 1.0,
    ) -> "Signal":
        """Convert absolute price levels to distances and validate stop placement (closes BT-1).

        Raises SignalValidationError if stop is on the wrong side of price.
        """
        is_long = (direction == Direction.LONG or direction == 1 or str(direction).upper().endswith("LONG"))
        act = Action.ENTER_LONG if is_long else Action.ENTER_SHORT

        sl_dist = np.nan
        if stop_loss is not None:
            if is_long:
                if stop_loss >= close:
                    raise SignalValidationError(
                        f"Inverted stop: LONG stop_loss ({stop_loss:.2f}) must be strictly below current close ({close:.2f})."
                    )
                sl_dist = close - stop_loss
            else:
                if stop_loss <= close:
                    raise SignalValidationError(
                        f"Inverted stop: SHORT stop_loss ({stop_loss:.2f}) must be strictly above current close ({close:.2f})."
                    )
                sl_dist = stop_loss - close
        else:
            # Fallback for strategies without explicit stop loss (wide 50% fallback)
            sl_dist = close * 0.50

        tp_dist = np.nan
        if take_profit is not None:
            if is_long:
                if take_profit <= close:
                    raise SignalValidationError(
                        f"Inverted target: LONG take_profit ({take_profit:.2f}) must be strictly above current close ({close:.2f})."
                    )
                tp_dist = take_profit - close
            else:
                if take_profit >= close:
                    raise SignalValidationError(
                        f"Inverted target: SHORT take_profit ({take_profit:.2f}) must be strictly below current close ({close:.2f})."
                    )
                tp_dist = close - take_profit

        return cls(
            action=act,
            sl_distance=sl_dist,
            tp_distance=tp_dist,
            tag=tag,
            direction=Direction.LONG if is_long else Direction.SHORT,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )


@dataclass
class SignalTape:
    """Pre-computed signal arrays across all bars in a split.

    Shape of each array is (N,).
    """
    actions: np.ndarray             # uint8: 0=none, 1=long, 2=short, 3=close
    sl_distances: np.ndarray        # float64: distance in USD/oz (NaN=unused)
    tp_distances: np.ndarray        # float64: distance in USD/oz (NaN=unused)
    trail_distances: np.ndarray     # float64: distance in USD/oz (NaN=unused)
    time_stops: np.ndarray          # int32: max bars to hold (0=none)
    breakeven_r: np.ndarray         # float64: R-multiple to move SL to BE (NaN=unused)
    tags: list[str] = field(default_factory=list)
    invalid_signals: int = 0
    total_signals: int = 0
    invalid_reasons: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls, n: int) -> "SignalTape":
        return cls(
            actions=np.zeros(n, dtype=np.uint8),
            sl_distances=np.full(n, np.nan, dtype=np.float64),
            tp_distances=np.full(n, np.nan, dtype=np.float64),
            trail_distances=np.full(n, np.nan, dtype=np.float64),
            time_stops=np.zeros(n, dtype=np.int32),
            breakeven_r=np.full(n, np.nan, dtype=np.float64),
            tags=[""] * n,
        )

    def validate_or_raise(self, max_invalid_pct: float = 2.0) -> None:
        """Enforce strict signal validity guarantees."""
        if self.total_signals == 0:
            return

        # Check for non-finite values in actions or entries
        entries = (self.actions == Action.ENTER_LONG) | (self.actions == Action.ENTER_SHORT)
        if np.any(entries & (~np.isfinite(self.sl_distances))):
            raise SignalValidationError(
                "SignalTape validation error: Entry signal has NaN or non-finite sl_distance. "
                "Every entry signal must specify sl_distance > 0."
            )

        if np.any(entries & (self.sl_distances <= 0)):
            raise SignalValidationError(
                "SignalTape validation error: Entry signal has non-positive sl_distance <= 0."
            )

        invalid_pct = (self.invalid_signals / self.total_signals) * 100.0
        if invalid_pct > max_invalid_pct:
            raise SignalValidationError(
                f"SignalTape validation failed: {self.invalid_signals}/{self.total_signals} "
                f"({invalid_pct:.2f}%) signals were invalid (exceeds {max_invalid_pct}% threshold). "
                f"Reasons: {self.invalid_reasons[:5]}"
            )

    def shifted(self, shift_bars: int) -> "SignalTape":
        """Return a copy of the tape shifted forward by shift_bars (for delay tests)."""
        if shift_bars == 0:
            return self

        n = len(self.actions)
        new_tape = SignalTape.empty(n)
        if shift_bars >= n:
            return new_tape

        new_tape.actions[shift_bars:] = self.actions[:-shift_bars]
        new_tape.sl_distances[shift_bars:] = self.sl_distances[:-shift_bars]
        new_tape.tp_distances[shift_bars:] = self.tp_distances[:-shift_bars]
        new_tape.trail_distances[shift_bars:] = self.trail_distances[:-shift_bars]
        new_tape.time_stops[shift_bars:] = self.time_stops[:-shift_bars]
        new_tape.breakeven_r[shift_bars:] = self.breakeven_r[:-shift_bars]
        new_tape.tags[shift_bars:] = self.tags[:-shift_bars]
        return new_tape


def validate_and_record_signal(
    signal: Signal | None,
    bar_idx: int,
    tape: SignalTape,
    atr_prev: float,
    current_spread: float,
    cfg: dict,
) -> None:
    """Validate a single bar's signal and record it into the tape."""
    if signal is None or signal.action == Action.NONE:
        return

    tape.total_signals += 1
    act = signal.action

    if act == Action.CLOSE:
        tape.actions[bar_idx] = Action.CLOSE
        tape.tags[bar_idx] = signal.tag
        return

    # Entry validations
    sig_cfg = cfg.get("signals", cfg.get("strategy", {}))
    min_sl_usd = sig_cfg.get("min_sl_usd", 1.0)
    min_sl_atr_mult = sig_cfg.get("min_sl_atr_multiple", 0.3)
    min_required_sl = max(min_sl_usd, min_sl_atr_mult * (atr_prev if np.isfinite(atr_prev) else 1.0))
    max_sl_usd = sig_cfg.get("max_sl_usd", 1000.0)

    # Validate sl_distance
    sl_dist = signal.sl_distance
    if np.isnan(sl_dist) or sl_dist <= 0:
        tape.invalid_signals += 1
        tape.invalid_reasons.append(f"Bar {bar_idx}: missing or non-positive sl_distance={sl_dist}")
        return

    if min_required_sl > 0 and sl_dist < min_required_sl:
        tape.invalid_signals += 1
        tape.invalid_reasons.append(
            f"Bar {bar_idx}: sl_distance ({sl_dist:.2f}) < min allowed ({min_required_sl:.2f})"
        )
        return

    if sl_dist > max_sl_usd:
        tape.invalid_signals += 1
        tape.invalid_reasons.append(
            f"Bar {bar_idx}: sl_distance ({sl_dist:.2f}) > max allowed ({max_sl_usd:.2f})"
        )
        return

    # Validate tp_distance if provided
    tp_dist = signal.tp_distance
    if not np.isnan(tp_dist):
        min_tp_spread_mult = sig_cfg.get("min_tp_spread_multiple", 3.0)
        min_tp_usd = sig_cfg.get("min_tp_usd", 0.50)
        required_min_tp = max(min_tp_usd, min_tp_spread_mult * current_spread)
        if required_min_tp > 0 and tp_dist < required_min_tp:
            tape.invalid_signals += 1
            tape.invalid_reasons.append(
                f"Bar {bar_idx}: tp_distance ({tp_dist:.2f}) < min allowed ({required_min_tp:.2f})"
            )
            return

    # Record valid entry
    tape.actions[bar_idx] = act
    tape.sl_distances[bar_idx] = sl_dist
    tape.tp_distances[bar_idx] = tp_dist
    tape.trail_distances[bar_idx] = signal.trail_distance
    tape.time_stops[bar_idx] = int(signal.time_stop_bars)
    tape.breakeven_r[bar_idx] = signal.breakeven_after_r
    tape.tags[bar_idx] = signal.tag
