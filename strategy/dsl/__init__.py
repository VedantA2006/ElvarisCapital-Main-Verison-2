"""
strategy/dsl – Structured Quantitative Strategy Domain-Specific Language (DSL).

Includes:
- schema.py: Pydantic & YAML/JSON Strategy DSL specifications
- compiler.py: Compiles Strategy DSL into verified, sandboxed Strategy code
- validator.py: Deterministic rule verification & indicator contract validation
- fingerprint.py: 3-tier duplicate detection (exact, logical, behavioral)
"""

from strategy.dsl.schema import StrategyDSL, EntryRule, ExitRule, RiskRule
from strategy.dsl.compiler import DSLCompiler
from strategy.dsl.validator import DSLValidator
from strategy.dsl.fingerprint import StrategyFingerprinter

__all__ = [
    "StrategyDSL",
    "EntryRule",
    "ExitRule",
    "RiskRule",
    "DSLCompiler",
    "DSLValidator",
    "StrategyFingerprinter",
]
