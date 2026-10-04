"""Canary 16: Unseeded random signals.
Expected gate: determinism
"""
import random
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class UnseededRandomStrategy:
    """Deliberately emits non-deterministic signals using unseeded random."""
    def __init__(self, params=None):
        # Explicit unseeded random generator initialized from system time/entropy
        self.rng = random.Random()

    def on_bar(self, bars):
        if len(bars) < 5:
            return None
        # Unseeded random call: will vary between runs
        if self.rng.random() < 0.10:
            return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
        return None
