"""
sandbox/policy.py: AST policy scanner, restricted builtins, and audit hook.

Enforces:
1. Static AST Policy (validate_ast_policy):
   - Rejects dunder attributes: __class__, __base__, __subclasses__, __globals__, etc.
   - Rejects sensitive attributes: .base, .ctypes, .data, __array_interface__, f_back, tb_frame, .environ
   - Rejects dangerous calls: eval, exec, compile, open, getattr, setattr, delattr, __import__, globals, locals, vars, dir
   - Rejects forbidden imports outside allow-list {numpy, pandas, math, numba, etc.}
   - Rejects lookahead patterns: shift(-N), shift(periods=-1), shift(var), rolling(center=True), iloc[i+1], iloc[var]
   - Rejects star imports and nested functions declaring global
2. Audit hook (audit_hook):
   - Runtime enforcement via sys.addaudithook
   - Blocks socket.*, subprocess.*, os.system/exec/spawn/fork/remove/rename/mkdir/chdir, ctypes.*,
     file writes, unauthorized file reads, pickle/marshal loads, banned imports.
3. Safe Builtins & Restricted Namespaces.
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolicyViolation:
    line: int
    col: int
    pattern: str
    detail: str
    severity: str = "error"  # "error" = hard rejection, "warning" = advisory


@dataclass
class PolicyResult:
    passed: bool
    violations: list[PolicyViolation] = field(default_factory=list)
    banned_imports: list[str] = field(default_factory=list)
    summary: str = ""

    def to_doc(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [
                {
                    "line": v.line,
                    "col": v.col,
                    "pattern": v.pattern,
                    "detail": v.detail,
                    "severity": v.severity,
                }
                for v in self.violations
            ],
            "banned_imports": self.banned_imports,
            "summary": self.summary,
        }


ALLOWED_MODULES = {
    "numpy",
    "pandas",
    "math",
    "random",
    "numba",
    "np",
    "pd",
    "core.indicators",
    "core.signals",
    "core.backtester",
    "collections",
    "itertools",
    "functools",
    "dataclasses",
    "typing",
    "enum",
    "abc",
}

BANNED_CALL_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "getattr",
    "setattr",
    "delattr",
    "__import__",
    "globals",
    "locals",
    "vars",
    "dir",
    "breakpoint",
    "input",
    "exit",
    "quit",
}

BANNED_ATTR_NAMES = {
    "base",
    "ctypes",
    "data",
    "__array_interface__",
    "f_back",
    "tb_frame",
    "gi_frame",
    "cr_frame",
    "environ",
}


class _ASTPolicyVisitor(ast.NodeVisitor):
    def __init__(self, allowed_imports: set[str] | None = None):
        self.allowed = set(allowed_imports or ALLOWED_MODULES) | {
            "core.signals", "core.indicators", "core.backtester",
        }
        self.violations: list[PolicyViolation] = []
        self.banned_imports: list[str] = []
        self._fn_depth = 0
        self._assigned_vars: dict[str, Any] = {}

    def _add_error(self, node: ast.AST, pattern: str, detail: str):
        self.violations.append(
            PolicyViolation(
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                pattern=pattern,
                detail=detail,
                severity="error",
            )
        )

    def _add_warning(self, node: ast.AST, pattern: str, detail: str):
        self.violations.append(
            PolicyViolation(
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                pattern=pattern,
                detail=detail,
                severity="warning",
            )
        )

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if alias.name in ("core.signals", "core.indicators", "core.backtester") or alias.name in self.allowed or root in self.allowed:
                continue
            self.banned_imports.append(alias.name)
            self._add_error(
                node,
                "banned_import",
                f"Import of '{alias.name}' is forbidden in strategy code. Allowed: {sorted(self.allowed)}",
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        root = mod.split(".")[0]
        if not (mod in ("core.signals", "core.indicators", "core.backtester") or mod in self.allowed or root in self.allowed):
            self.banned_imports.append(mod)
            self._add_error(
                node,
                "banned_import",
                f"Import from '{mod}' is forbidden. Allowed: {sorted(self.allowed)}",
            )
        # Disallow star imports
        for alias in node.names:
            if alias.name == "*":
                self._add_error(
                    node,
                    "star_import",
                    "Star imports (from ... import *) are strictly forbidden in strategy code.",
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        attr = node.attr
        # Reject dunder attribute access: e.g. obj.__class__, obj.__base__, obj.__subclasses__
        if attr.startswith("__") and attr.endswith("__"):
            self._add_error(
                node,
                "dunder_attribute",
                f"Access to dunder attribute '{attr}' is forbidden.",
            )
        # Reject specific security/lookahead attributes
        elif attr in BANNED_ATTR_NAMES:
            self._add_error(
                node,
                f"banned_attr_{attr}",
                f"Access to attribute '.{attr}' is forbidden (potential lookahead or sandbox escape).",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # 1. Check direct function calls: eval(), exec(), open(), etc.
        if isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CALL_NAMES:
                self._add_error(
                    node,
                    f"banned_call_{node.func.id}",
                    f"Call to '{node.func.id}()' is forbidden in strategy code.",
                )

        # 2. Check method calls
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method in BANNED_CALL_NAMES:
                self._add_error(
                    node,
                    f"banned_call_{method}",
                    f"Call to '{method}()' is forbidden.",
                )

            # Check shift, diff, pct_change
            if method in ("shift", "diff", "pct_change"):
                # Positional argument check
                if node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                        self._add_error(
                            node,
                            f"{method}_negative",
                            f"{method}(-N) accesses future data. Use positive offsets for past data.",
                        )
                    elif isinstance(arg, ast.Constant):
                        if isinstance(arg.value, (int, float)) and arg.value < 0:
                            self._add_error(
                                node,
                                f"{method}_negative",
                                f"{method}({arg.value}) accesses future data.",
                            )
                    elif isinstance(arg, (ast.Name, ast.BinOp, ast.Call)):
                        # Non-literal or variable offset
                        self._add_error(
                            node,
                            f"{method}_non_literal",
                            f"{method}() with non-literal or variable offset is forbidden.",
                        )
                # Keyword argument check: e.g. periods=-1 or periods=var
                for kw in node.keywords:
                    if kw.arg == "periods":
                        if isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.op, ast.USub):
                            self._add_error(
                                node,
                                f"{method}_negative",
                                f"{method}(periods=-N) accesses future data.",
                            )
                        elif isinstance(kw.value, ast.Constant):
                            if isinstance(kw.value.value, (int, float)) and kw.value.value < 0:
                                self._add_error(
                                node,
                                f"{method}_negative",
                                f"{method}(periods={kw.value.value}) accesses future data.",
                            )
                        elif isinstance(kw.value, (ast.Name, ast.BinOp, ast.Call)):
                            self._add_error(
                                node,
                                f"{method}_non_literal",
                                f"{method}(periods=...) with non-literal or variable offset is forbidden.",
                            )

            # Check rolling, expanding, ewm
            if method in ("rolling", "expanding", "ewm"):
                for kw in node.keywords:
                    if kw.arg == "center":
                        if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            self._add_error(
                                node,
                                "rolling_center",
                                "rolling(..., center=True) centers the window using future data.",
                            )
                        elif not isinstance(kw.value, ast.Constant):
                            self._add_error(
                                node,
                                "rolling_center_variable",
                                "rolling(..., center=...) with non-literal argument is forbidden.",
                            )

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        # Check .iloc, .iat, .loc, .values for forward or non-literal offsets
        if isinstance(node.value, ast.Attribute) and node.value.attr in ("iloc", "iat", "loc", "values"):
            attr = node.value.attr
            s = node.slice
            # e.g., iloc[i+1], values[i+2]
            if isinstance(s, ast.BinOp) and isinstance(s.op, ast.Add):
                if isinstance(s.right, ast.Constant) and isinstance(s.right.value, (int, float)) and s.right.value > 0:
                    self._add_error(
                        node,
                        "index_forward",
                        f"{attr}[i+{s.right.value}] accesses future data.",
                    )
            # e.g., iloc[j] where j is a variable
            elif isinstance(s, ast.Name):
                self._add_error(
                    node,
                    f"{attr}_variable",
                    f"{attr}[{s.id}] with variable index is forbidden (could access future data).",
                )
            # Check slice step: e.g. [::-1]
            elif isinstance(s, ast.Slice):
                if s.step is not None:
                    if isinstance(s.step, ast.UnaryOp) and isinstance(s.step.op, ast.USub):
                        self._add_error(
                            node,
                            f"{attr}_reverse_slice",
                            f"{attr}[::-N] reversed slice is forbidden (potential lookahead).",
                        )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._fn_depth += 1
        self.generic_visit(node)
        self._fn_depth -= 1

    def visit_Global(self, node: ast.Global):
        # Disallow nested functions assigning global
        if self._fn_depth > 1:
            self._add_error(
                node,
                "nested_global",
                f"Nested function assigning global variables {node.names} is forbidden.",
            )
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        # Warn on while True without explicit break
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
            if not has_break:
                self._add_warning(
                    node,
                    "unbounded_while",
                    "while True without break detected. Process isolation timeout protects engine.",
                )
        self.generic_visit(node)


def validate_ast_policy(
    source: str,
    allowed_imports: set[str] | None = None,
    max_code_len: int = 15000,
) -> PolicyResult:
    """Validate strategy source code against the strict AST policy.

    Returns PolicyResult. Code is only approved if passed is True.
    """
    if len(source) > max_code_len:
        return PolicyResult(
            passed=False,
            violations=[
                PolicyViolation(
                    line=0,
                    col=0,
                    pattern="code_too_long",
                    detail=f"Code length {len(source)} exceeds maximum allowed {max_code_len}.",
                )
            ],
            summary=f"Code too long ({len(source)} > {max_code_len})",
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return PolicyResult(
            passed=False,
            violations=[
                PolicyViolation(
                    line=e.lineno or 0,
                    col=e.offset or 0,
                    pattern="syntax_error",
                    detail=f"Syntax error: {e}",
                )
            ],
            summary="Code has syntax errors",
        )

    visitor = _ASTPolicyVisitor(allowed_imports)
    visitor.visit(tree)

    errors = [v for v in visitor.violations if v.severity == "error"]
    passed = len(errors) == 0
    summary = (
        "AST policy passed."
        if passed
        else f"AST policy failed with {len(errors)} error(s): {'; '.join(f'L{v.line}: {v.detail}' for v in errors[:3])}"
    )

    return PolicyResult(
        passed=passed,
        violations=visitor.violations,
        banned_imports=visitor.banned_imports,
        summary=summary,
    )


def static_scan(source: str, allowed_imports: set[str] | None = None) -> PolicyResult:
    """Compatibility alias for static_scan."""
    return validate_ast_policy(source, allowed_imports)


# ═══════════════════════════════════════════════════════════════════════════
# Audit Hook
# ═══════════════════════════════════════════════════════════════════════════

_ALLOWED_READ_PREFIXES: list[str] = []


def _init_allowed_prefixes():
    global _ALLOWED_READ_PREFIXES
    if not _ALLOWED_READ_PREFIXES:
        pfxs = [
            os.path.abspath(sys.prefix),
            os.path.abspath(sys.base_prefix),
        ]
        # Include site-packages / lib
        for p in sys.path:
            if p and os.path.exists(p):
                pfxs.append(os.path.abspath(p))
        _ALLOWED_READ_PREFIXES = pfxs


def audit_hook(event: str, args: tuple):
    """sys.addaudithook callback installed in the child process.

    Blocks network, subprocess, filesystem mutations, unauthorised file reads,
    ctypes, and forbidden imports.
    """
    # 1. Sockets / Network
    if event.startswith("socket."):
        # socket.__new__ creates the object, but connect/bind/send/recv do IO
        if event not in ("socket.__new__",):
            raise PermissionError(f"Network access blocked by sandbox: {event}")

    # 2. Subprocess
    if event.startswith("subprocess."):
        raise PermissionError(f"Subprocess execution blocked by sandbox: {event}")

    # 3. OS System & Process execution
    if event in (
        "os.system",
        "os.posix_spawn",
        "os.spawn",
        "os.fork",
        "os.exec",
        "os.kill",
    ) or event.startswith("os.spawn") or event.startswith("os.exec"):
        raise PermissionError(f"OS process execution blocked by sandbox: {event}")

    # 4. Filesystem mutations
    if event in (
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "os.rename",
        "os.replace",
        "os.mkdir",
        "os.chdir",
        "shutil.rmtree",
        "shutil.copyfile",
        "shutil.copytree",
    ):
        raise PermissionError(f"Filesystem modification blocked by sandbox: {event}")

    # 5. Ctypes
    if event.startswith("ctypes."):
        raise PermissionError(f"Ctypes access blocked by sandbox: {event}")

    # 6. Open calls
    if event == "open":
        path = args[0] if len(args) > 0 else ""
        mode = args[1] if len(args) > 1 else "r"
        # Disallow any write / append / creation mode
        if any(m in str(mode) for m in ("w", "a", "+", "x")):
            raise PermissionError(f"File write blocked by sandbox: path={path!r} mode={mode!r}")

        # Check read path: only allow within python standard library / site-packages
        _init_allowed_prefixes()
        if isinstance(path, (str, bytes)):
            try:
                abs_p = os.path.abspath(str(path))
                allowed = any(abs_p.startswith(pfx) for pfx in _ALLOWED_READ_PREFIXES)
                if not allowed:
                    raise PermissionError(f"File read outside python environment blocked by sandbox: {path!r}")
            except Exception as e:
                if isinstance(e, PermissionError):
                    raise
                raise PermissionError(f"Invalid file path in sandbox open: {path!r}")

    # 7. Unsafe deserialization
    if event == "pickle.find_class":
        raise PermissionError(f"Unsafe deserialization blocked by sandbox: {event}")
    if event == "marshal.loads":
        try:
            frame = sys._getframe(1)
            filename = frame.f_code.co_filename
            if "importlib" not in filename:
                raise PermissionError(f"Unsafe deserialization blocked by sandbox: {event}")
        except Exception:
            raise PermissionError(f"Unsafe deserialization blocked by sandbox: {event}")

    # 8. Imports
    if event == "import":
        mod_name = args[0] if len(args) > 0 else ""
        root = mod_name.split(".")[0]
        # Allow internal python bootstrap and allowed modules
        if root and root not in ALLOWED_MODULES:
            # Check if internal standard python submodule required by interpreter
            _allowed_internals = {
                "sys", "builtins", "encodings", "codecs", "io", "abc", "site",
                "_signal", "_thread", "time", "_weakref", "zipimport", "importlib",
                "_collections", "operator", "itertools", "functools", "math",
                "_operator", "errno", "signal", "posix", "nt", "_locale",
                "json", "typing", "collections", "types", "enum", "re",
                "multiprocessing", "_multiprocessing", "select", "socket",
                "threading", "weakref", "copy", "struct", "traceback", "logging",
            }
            if root not in _allowed_internals:
                raise PermissionError(f"Import of '{mod_name}' blocked by sandbox audit hook.")


# ═══════════════════════════════════════════════════════════════════════════
# Restricted Namespace Wrappers
# ═══════════════════════════════════════════════════════════════════════════

def get_restricted_pandas():
    """Wrap pandas to strip all file IO and dangerous modules."""
    import pandas as pd
    # We can create a module proxy or shallow copy
    wrapper = type(pd)("pandas")
    # Copy attributes except banned ones
    banned_pd = {
        "read_csv", "read_parquet", "read_sql", "read_feather", "read_json",
        "read_html", "read_excel", "read_pickle", "read_table", "read_clipboard",
        "read_stata", "read_sas", "read_spss", "read_orc", "read_xml",
        "to_csv", "to_parquet", "to_sql", "to_feather", "to_json",
        "to_html", "to_excel", "to_pickle", "to_clipboard",
        "io", "testing", "compat", "util",
    }
    for k, v in pd.__dict__.items():
        if k not in banned_pd:
            setattr(wrapper, k, v)
    return wrapper


def get_restricted_numpy():
    """Wrap numpy to strip file IO and ctypes access."""
    import numpy as np
    wrapper = type(np)("numpy")
    banned_np = {
        "load", "save", "savez", "savez_compressed", "savetxt",
        "fromfile", "fromregex", "tofile", "memmap", "ctypeslib",
        "testing", "compat", "DataSource",
    }
    for k, v in np.__dict__.items():
        if k not in banned_np:
            setattr(wrapper, k, v)
    return wrapper


SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "complex", "dict", "divmod",
    "enumerate", "filter", "float", "format", "frozenset",
    "hasattr", "hash", "hex", "id", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "oct", "ord", "pow", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple",
    "zip", "True", "False", "None",
    "__build_class__",  # Needed for class definitions
    "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration",
    "ZeroDivisionError", "OverflowError", "ArithmeticError",
    "Exception", "BaseException",
    "property", "staticmethod", "classmethod", "super",
}


def make_safe_builtins(allowed_modules: set[str] | None = None) -> dict[str, Any]:
    """Return a dict containing only safe builtins (no type, object, open, exec, eval)."""
    import builtins
    safe = {}
    for name in SAFE_BUILTINS:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)

    allowed = allowed_modules or ALLOWED_MODULES

    def _safe_import(name, *args, **kwargs):
        root = name.split(".")[0]
        if root not in allowed and name not in allowed and not (root == "core" and name in {"core.signals", "core.indicators", "core.backtester"}):
            from sandbox.runner import SandboxImportError
            raise SandboxImportError(f"Import of '{name}' is forbidden in strategy code.")
        return __import__(name, *args, **kwargs)

    safe["__import__"] = _safe_import
    return safe
