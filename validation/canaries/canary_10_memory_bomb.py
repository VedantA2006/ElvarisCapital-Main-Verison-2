"""Canary 10: Memory bomb allocation.
Expected gate: sandbox_memory (or MemoryError / killed)
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class MemoryBombStrategy:
    """Deliberately attempts to exhaust RAM."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        # 10^10 float64 elements ~ 80 GB RAM
        bomb = np.zeros(10**10, dtype=np.float64)
        return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
