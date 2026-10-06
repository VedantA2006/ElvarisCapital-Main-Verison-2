"""
sandbox/child.py: Child worker process entrypoint.

Executed inside an isolated Python process:
- Started with python -I
- Fresh empty working directory
- Empty/scrubbed environment (no MONGO_URL, no API keys)
- Audit hook locked in before any strategy code executes
- Bounded sliding window with DataFrame.copy() fresh memory buffers
"""

from __future__ import annotations

import collections
import math
import os
import random
import sys
import traceback

# Ensure project root is on sys.path in python -I isolated mode
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from multiprocessing.connection import Client

import numpy as np
import pandas as pd

from core.signals import Action, Direction, Signal
from sandbox.policy import (
    audit_hook,
    get_restricted_numpy,
    get_restricted_pandas,
    make_safe_builtins,
)


def child_main(host: str, port: int, authkey: bytes):
    # 1. Connect to parent
    conn = Client((host, port), authkey=authkey)

    # 2. Warm-up runtime
    _ = pd.DataFrame({"a": [1.0, 2.0, 3.0]}).rolling(2).mean()

    # 3. Lock down audit hooks
    sys.addaudithook(audit_hook)

    # 4. Notify parent that child is ready
    conn.send({"status": "ready"})

    buf_cols: dict[str, np.ndarray] | None = None
    cols_list: list[str] | None = None
    count: int = 0

    # 5. Command loop
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        except Exception as ex:
            _ = str(ex)
            break

        cmd = msg.get("cmd")

        if cmd == "init":
            code_src = msg.get("code", "")
            params = msg.get("params") or {}
            seed = msg.get("seed", 42)
            max_lookback = int(msg.get("max_lookback", 1500))
            buf_cols = None
            cols_list = None
            count = 0

            # Seed determinism
            random.seed(seed)
            np.random.seed(seed)

            # Restricted namespace
            ns = {
                "__builtins__": make_safe_builtins(),
                "__name__": "<strategy>",
                "__doc__": None,
                "pd": get_restricted_pandas(),
                "pandas": get_restricted_pandas(),
                "np": get_restricted_numpy(),
                "numpy": get_restricted_numpy(),
                "math": math,
                "Signal": Signal,
                "Direction": Direction,
                "Action": Action,
            }

            try:
                import core.indicators as indicators_module
                for attr in dir(indicators_module):
                    if not attr.startswith("_"):
                        ns[attr] = getattr(indicators_module, attr)
            except (ImportError, AttributeError) as exc:
                sys.stderr.write(f"Warning: could not inject core.indicators: {exc}\n")

            try:
                code_obj = compile(code_src, "<strategy>", "exec")
                exec(code_obj, ns)

                # Locate strategy class
                strategy_cls = None
                for k, v in ns.items():
                    if isinstance(v, type) and hasattr(v, "on_bar"):
                        strategy_cls = v
                        break

                if strategy_cls is None:
                    raise RuntimeError("No strategy class with an 'on_bar' method found (no class).")

                # Merge defaults from class PARAMS (F3, SBX-5)
                defaults = {}
                raw_params_decl = getattr(strategy_cls, "PARAMS", None) or ns.get("PARAMS", {})
                if isinstance(raw_params_decl, dict):
                    for pk, pv in raw_params_decl.items():
                        if isinstance(pv, dict) and "default" in pv:
                            defaults[pk] = pv["default"]
                        else:
                            defaults[pk] = pv

                effective_params = {}
                for k, v in {**defaults, **(params or {})}.items():
                    if isinstance(v, dict) and "default" in v:
                        effective_params[k] = v["default"]
                    else:
                        effective_params[k] = v

                # Instantiate with parameters if supported (SBX-5)
                try:
                    if effective_params:
                        try:
                            strategy_instance = strategy_cls(effective_params)
                        except TypeError:
                            try:
                                strategy_instance = strategy_cls(**effective_params)
                            except TypeError:
                                strategy_instance = strategy_cls()
                    else:
                        try:
                            strategy_instance = strategy_cls()
                        except TypeError:
                            strategy_instance = strategy_cls({})
                except Exception as e:
                    raise RuntimeError(f"Error instantiating strategy {strategy_cls.__name__}: {e}") from e

                conn.send({"status": "initialized"})

            except Exception as exc:
                tb_str = traceback.format_exc()
                conn.send({
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": tb_str,
                })

        elif cmd == "bar":
            bar_dict = msg.get("bar")
            if strategy_instance is None:
                conn.send({
                    "status": "error",
                    "error_type": "RuntimeError",
                    "message": "Strategy has not been initialized.",
                    "traceback": "",
                })
                continue

            try:
                # Incremental feed into bounded window using fast numpy columnar buffers
                if buf_cols is None:
                    cols_list = list(bar_dict.keys())
                    buf_cols = {}
                    for col in cols_list:
                        val = bar_dict[col]
                        dt = object if isinstance(val, (str, bool)) else np.float64
                        buf_cols[col] = np.empty(max_lookback, dtype=dt)

                if count < max_lookback:
                    for col in cols_list:
                        buf_cols[col][count] = bar_dict[col]
                    count += 1
                    col_views = {col: buf_cols[col][:count].copy() for col in cols_list}
                else:
                    for col in cols_list:
                        arr = buf_cols[col]
                        arr[:-1] = arr[1:]
                        arr[-1] = bar_dict[col]
                    col_views = {col: buf_cols[col].copy() for col in cols_list}

                # Construct DataFrame from fresh buffer copies (no views into parent)
                bars_copy = pd.DataFrame(col_views)

                sig = strategy_instance.on_bar(bars_copy)

                if sig is None:
                    conn.send({"status": "ok", "signal": None})
                else:
                    raw_dir = getattr(sig, "direction", None)
                    dir_val = None
                    if raw_dir is not None:
                        dir_val = raw_dir.value if hasattr(raw_dir, "value") else int(raw_dir)

                    sig_payload = {
                        "action": sig.action.value if hasattr(sig.action, "value") else int(sig.action),
                        "direction": dir_val,
                        "stop_loss": getattr(sig, "stop_loss", np.nan),
                        "take_profit": getattr(sig, "take_profit", np.nan),
                        "sl_distance": getattr(sig, "sl_distance", np.nan),
                        "tp_distance": getattr(sig, "tp_distance", np.nan),
                        "tag": getattr(sig, "tag", ""),
                    }
                    conn.send({"status": "ok", "signal": sig_payload})

            except Exception as exc:
                # Strategy exception inside on_bar (BT-12)
                tb_str = traceback.format_exc()
                conn.send({
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": tb_str,
                })

        elif cmd == "close":
            break

    try:
        conn.close()
    except Exception as ex:
        _ = str(ex)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
    addr_str = sys.argv[1]
    host_part, port_part = addr_str.split(":")
    auth_bytes = bytes.fromhex(sys.argv[2])
    child_main(host_part, int(port_part), auth_bytes)
