"""Canary 12: Write a file then read it in a later run.
Expected gate: static_policy / audit hook / ephemeral child
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class FileWriteReadStrategy:
    """Attempts persistent cross-run state storage via local filesystem."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        with open("strategy_cache.tmp", "w") as f:
            f.write("state=1")
        return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
