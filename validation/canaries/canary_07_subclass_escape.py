"""Canary 07: Subclass-chain escape to os (SBX-1).
Expected gate: static_policy (and audit hook blocks if reached)
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class SubclassEscapeStrategy:
    """Attempts to reach os module via dunder subclass traversal."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        subclasses = ().__class__.__base__.__subclasses__()
        for s in subclasses:
            if "catch_warnings" in getattr(s, "__name__", ""):
                mod = s._module
                break
        return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
