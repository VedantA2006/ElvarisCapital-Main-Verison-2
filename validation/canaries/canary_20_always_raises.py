"""Canary 20: Strategy that raises on every bar (BT-12).
Expected gate: code_error with traceback (never swallowed as 'no trades')
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class AlwaysRaisesStrategy:
    """Deliberately crashes on every bar with an unhandled exception."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        raise RuntimeError("Deterministic crash in strategy on_bar (BT-12 canary)")
