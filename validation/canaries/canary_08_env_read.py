"""Canary 08: Read MONGO_URL from the environment (SBX-3).
Expected gate: blocked: environment is empty / static scan on environ
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class EnvReadStrategy:
    """Attempts to inspect environment variables for database credentials."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        # Even if environ is accessed, MONGO_URL must be absent (empty env)
        val = getattr(pd, "__builtins__", {}).get("os", None)
        return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
