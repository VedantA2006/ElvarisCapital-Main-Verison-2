"""
strategy/dsl/compiler.py – Compiles StrategyDSL into sandboxed Python Strategy code.

Guarantees (Sections 11, 14, 15):
1. Emits standard, fully leak-free Strategy class adhering to QuantForge contract.
2. Uses only verified helpers from core.indicators.
3. Automatically injects bar warmup and stop loss distance sanitization.
4. Generates clean, deterministic python code suitable for execution inside the sandbox.
"""

from __future__ import annotations

import json
import re
from typing import Any
from strategy.dsl.schema import StrategyDSL, EntryRule, ConditionDef, IndicatorDef


class DSLCompiler:
    """Translates StrategyDSL specifications into production-grade Python Strategy classes."""

    def compile(self, dsl: StrategyDSL) -> str:
        """Compile a StrategyDSL object into Python source code."""
        lines = [
            f"# Auto-compiled from StrategyDSL: {dsl.name}",
            f"# Family: {dsl.family}",
            f"# Market: {dsl.market} | Primary TF: {dsl.timeframes.primary} | Context: {dsl.timeframes.context}",
            f"# Hypothesis: {dsl.hypothesis.replace(chr(10), ' ')}",
            "",
            "import numpy as np",
            "import pandas as pd",
            "",
            "class Strategy:",
            "    PARAMS = {",
        ]

        # Parameters block
        for p_name, p_def in dsl.parameters.items():
            lines.append(
                f"        {json.dumps(p_name)}: {{"
                f"'default': {p_def.default}, 'min': {p_def.min}, 'max': {p_def.max}"
                f"}},"
            )
        lines.extend([
            "    }",
            "",
            "    def __init__(self, params=None):",
            "        self.params = {} if params is None else dict(params)",
            "        for k, v in self.PARAMS.items():",
            "            if k not in self.params:",
            "                self.params[k] = v['default']",
            "        self._last_entry_bar = -999",
            "",
            "    def on_bar(self, bars: pd.DataFrame):",
            "        if len(bars) < 60:",
            "            return None",
            "",
            "        from core.indicators import (",
            "            ema, sma, wma, rsi, macd, atr, adx, supertrend,",
            "            bollinger, keltner, stoch, roc, donchian, session_range,",
            "            htf, swing_high, swing_low, bos, choch, fvg, is_session,",
            "        )",
            "",
            "        atr_val = float(atr(bars, period=14).iloc[-1])",
            "        if np.isnan(atr_val) or atr_val <= 0:",
            "            return None",
            "",
            "        close_cur = float(bars['close'].iloc[-1])",
            "        close_prev = float(bars['close'].iloc[-2])",
            "",
        ])

        # Generate indicator declarations
        for ind in dsl.indicators:
            call_code = self._render_indicator_call(ind)
            lines.append(f"        {ind.id} = {call_code}")

        lines.append("")

        # Generate rule conditions
        long_rule = next((r for r in dsl.entry_rules if r.direction == "LONG"), None)
        short_rule = next((r for r in dsl.entry_rules if r.direction == "SHORT"), None)

        if long_rule:
            lines.append("        # --- Long Entry Evaluation ---")
            cond_expr = self._render_rule_condition(long_rule)
            lines.append(f"        long_signal = {cond_expr}")
        else:
            lines.append("        long_signal = False")

        if short_rule:
            lines.append("        # --- Short Entry Evaluation ---")
            cond_expr = self._render_rule_condition(short_rule)
            lines.append(f"        short_signal = {cond_expr}")
        else:
            lines.append("        short_signal = False")

        # SL and TP calculation
        sl_calc = self._render_sl_calc(dsl.exit)
        tp_calc = self._render_tp_calc(dsl.exit)

        lines.extend([
            "",
            f"        sl_dist = {sl_calc}",
            f"        tp_dist = {tp_calc}",
            "        if np.isnan(sl_dist) or sl_dist <= 0:",
            "            sl_dist = max(5.0, atr_val * 1.5)",
            "",
            "        cur_bar_idx = len(bars) - 1",
            "        cooldown = 2",
            "",
            "        if long_signal and not short_signal and (cur_bar_idx - self._last_entry_bar >= cooldown):",
            "            self._last_entry_bar = cur_bar_idx",
            "            return Signal.enter_long(sl_distance=sl_dist, tp_distance=tp_dist)",
            "",
            "        if short_signal and not long_signal and (cur_bar_idx - self._last_entry_bar >= cooldown):",
            "            self._last_entry_bar = cur_bar_idx",
            "            return Signal.enter_short(sl_distance=sl_dist, tp_distance=tp_dist)",
            "",
            "        return None",
            "",
        ])

        return "\n".join(lines)

    def _render_indicator_call(self, ind: IndicatorDef) -> str:
        param_parts = []
        for k, v in ind.params.items():
            if isinstance(v, str) and v.startswith("param:"):
                p_key = v.split(":", 1)[1]
                param_parts.append(f"{k}=self.params.get('{p_key}', 14)")
            else:
                param_parts.append(f"{k}={repr(v)}")

        arg_str = ", ".join(["bars"] + param_parts)
        if ind.timeframe:
            return f"htf(bars, target_tf={repr(ind.timeframe)}, func=lambda df: {ind.type}(df, {', '.join(param_parts)}))"
        return f"{ind.type}({arg_str})"

    def _render_rule_condition(self, rule: EntryRule) -> str:
        sub_exprs = []

        if rule.session_filter and rule.session_filter != "all":
            if rule.session_filter == "london_ny":
                sub_exprs.append("(bool(is_session(bars, 'london').iloc[-1]) or bool(is_session(bars, 'ny').iloc[-1]))")
            else:
                sub_exprs.append(f"bool(is_session(bars, '{rule.session_filter}').iloc[-1])")

        for cond in rule.all_conditions:
            sub_exprs.append(self._render_single_condition(cond))

        if not sub_exprs:
            return "False"
        return " and ".join(f"({e})" for e in sub_exprs)

    def _render_single_condition(self, cond: ConditionDef) -> str:
        left = self._resolve_operand(cond.left, cond.offset)
        op = cond.operator

        if op == "is_true":
            return f"bool({left})"
        if op == "is_false":
            return f"not bool({left})"

        right = self._resolve_operand(cond.right, cond.offset) if cond.right is not None else "0"

        if op == "cross_above":
            left_prev = self._resolve_operand(cond.left, cond.offset + 1)
            right_prev = self._resolve_operand(cond.right, cond.offset + 1) if cond.right is not None else "0"
            return f"({left} > {right} and {left_prev} <= {right_prev})"
        if op == "cross_below":
            left_prev = self._resolve_operand(cond.left, cond.offset + 1)
            right_prev = self._resolve_operand(cond.right, cond.offset + 1) if cond.right is not None else "0"
            return f"({left} < {right} and {left_prev} >= {right_prev})"

        return f"({left} {op} {right})"

    def _resolve_operand(self, op_val: Any, offset: int = 0) -> str:
        idx = -1 - offset
        if isinstance(op_val, (int, float)):
            return str(op_val)
        s = str(op_val).strip()
        if re.match(r"^-?\d+(\.\d+)?$", s):
            return s
        if s.startswith("bars.") or s in ("open", "high", "low", "close", "volume"):
            col = s.replace("bars.", "")
            return f"float(bars['{col}'].iloc[{idx}])"
        return f"float({s}.iloc[{idx}])"

    def _render_sl_calc(self, exit_rule: Any) -> str:
        if exit_rule.stop_loss_type == "atr":
            return f"max(5.0, float(atr_val * {exit_rule.stop_loss_multiplier}))"
        elif exit_rule.stop_loss_type == "points":
            return f"max(5.0, float({exit_rule.stop_loss_multiplier}))"
        return "max(5.0, float(atr_val * 1.5))"

    def _render_tp_calc(self, exit_rule: Any) -> str:
        if exit_rule.take_profit_type == "rr" and exit_rule.take_profit_ratio:
            return f"max(7.5, float(sl_dist * {exit_rule.take_profit_ratio}))"
        elif exit_rule.take_profit_type == "atr" and exit_rule.take_profit_multiplier:
            return f"max(7.5, float(atr_val * {exit_rule.take_profit_multiplier}))"
        elif exit_rule.take_profit_type == "none":
            return "None"
        return "max(7.5, float(sl_dist * 2.0))"
