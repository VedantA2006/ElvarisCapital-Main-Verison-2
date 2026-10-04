"""Canary 11: Network socket call attempt.
Expected gate: banned_import / audit hook (blocks socket.*)
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class NetworkCallStrategy:
    """Attempts to connect to an external server via socket."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("1.1.1.1", 80))
        return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
