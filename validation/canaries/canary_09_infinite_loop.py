"""Canary 09: Infinite loop (SBX-4).
Expected gate: sandbox_timeout
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class InfiniteLoopStrategy:
    """Deliberately hangs during bar processing."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        # Infinite loop that hangs execution
        while 1 == 1:
            pass
        return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
