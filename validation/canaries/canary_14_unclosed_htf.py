"""Canary 14: Unclosed higher-timeframe candle lookahead / repainting.
Expected gate: truncation (repainting detected across cuts)
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Action

class UnclosedHTFStrategy:
    """Deliberately peeks ahead into the unclosed portion of a higher-timeframe bar."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        i = len(bars) - 1
        arr = bars['close'].to_numpy()
        base = arr.base
        if base is not None and hasattr(base, "shape"):
            n_rows = base.shape[1] if base.ndim > 1 else base.shape[0]
            # End of unclosed 4h candle
            next_htf = min(n_rows, ((i // 4) + 1) * 4)
            if next_htf < n_rows:
                future_close = base[3, next_htf - 1] if base.ndim > 1 else base[next_htf - 1]
                if future_close > bars['close'].iloc[-1]:
                    return Signal(action=Action.ENTER_LONG, sl_distance=10.0, tp_distance=20.0)
        return None
