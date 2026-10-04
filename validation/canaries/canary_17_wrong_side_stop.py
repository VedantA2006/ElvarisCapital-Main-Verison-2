"""Canary 17: Wrong-side inverted stop (BT-1).
Expected gate: signal validation (SignalValidationError)
"""
import numpy as np
import pandas as pd
from core.signals import Signal, Direction

class WrongSideStopStrategy:
    """Attempts to place a stop loss above entry price on a LONG trade (BT-1 exploit)."""
    def __init__(self, params=None):
        pass

    def on_bar(self, bars):
        if len(bars) < 5:
            return None
        close = bars['close'].iloc[-1]
        # Inverted stop: stop_loss is $10 ABOVE close for a LONG!
        return Signal.from_prices(
            close=close,
            direction=Direction.LONG,
            stop_loss=close + 10.0,
            take_profit=close + 20.0,
        )
