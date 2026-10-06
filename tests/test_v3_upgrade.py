"""
tests/test_v3_upgrade.py – Master V3 Autonomous Engine Unit & Integration Tests.

Validates all Master V3 components:
1. Strategy DSL (Schema, Compiler, Validator)
2. Fingerprinter (Level 1 exact, Level 2 logical, Level 3 behavioral)
3. Prompt Firewall (Sanitization, Holdout secrecy)
4. XKiro / Qwen 3.8 Max Provider & Key Handling
5. Failure Analyzer & Strategy Autopsy (18 Failure Classes)
6. Research Director & Research Memory
7. Evolution Engine (Genome, Mutation, Crossover, Pareto Ranking)
8. Strategy Library State Machine & Frozen Invariance
9. Portfolio Engine (Correlation, Risk Parity, Max Sharpe)
"""

import os
import pytest
import numpy as np
import pandas as pd

from strategy.dsl.schema import StrategyDSL, IndicatorDef, EntryRule, ExitRule, TimeframeConfig
from strategy.dsl.compiler import DSLCompiler
from strategy.dsl.validator import DSLValidator
from strategy.dsl.fingerprint import StrategyFingerprinter
from strategy.families import StrategyFamily, DiversityManager
from strategy.library import StrategyLibrary, StrategyState
from llm.prompt_firewall import PromptFirewall
from llm.providers.xkiro import XKiroProvider
from llm.failure_analyzer import FailureAnalyzer, FailureCategory, StrategyAutopsy
from llm.research_memory import ResearchMemory, ResearchHypothesis
from llm.research_director import ResearchDirector, DirectorActionType
from evolution.genome import Genome, Gene, ParameterGene
from evolution.mutation import mutate_genome
from evolution.crossover import crossover_genomes
from evolution.fitness import evaluate_candidate_fitness, pareto_rank
from portfolio.correlation import calculate_correlation_matrix
from portfolio.risk import calculate_risk_contributions
from portfolio.allocator import allocate_portfolio, AllocationMethod


# ─── 1. DSL Schema, Compiler & Validator ─────────────────────────────────────

def test_dsl_compilation_and_ast_scan():
    """Verify StrategyDSL compiles into valid, sandboxed Python code."""
    dsl = StrategyDSL(
        name="Test_EMA_RSI_Gold",
        market="XAUUSD",
        family=StrategyFamily.TREND.value,
        hypothesis="Trend continuation after RSI pullback",
        timeframes=TimeframeConfig(primary="1h", context="4h"),
        indicators=[
            IndicatorDef(id="ema_trend", type="ema", params={"period": 50}),
            IndicatorDef(id="rsi_pullback", type="rsi", params={"period": 14}),
            IndicatorDef(id="atr_vol", type="atr", params={"period": 14}),
        ],
        entry_rules=[
            EntryRule(
                direction="LONG",
                all_conditions=[
                    {"left": "close", "operator": ">", "right": "ema_trend", "offset": 0},
                    {"left": "rsi_pullback", "operator": "<", "right": 45, "offset": 0},
                ],
                session_filter="london_ny",
            )
        ],
        exit=ExitRule(stop_loss_multiplier=1.5, take_profit_ratio=2.5),
        parameters={"ema_period": {"default": 50, "min": 10, "max": 100}},
    )

    validator = DSLValidator()
    val_res = validator.validate(dsl)
    assert val_res.is_valid, f"Validation errors: {val_res.errors}"

    compiler = DSLCompiler()
    py_code = compiler.compile(dsl)
    assert "class Strategy:" in py_code
    assert "def on_bar(" in py_code
    assert "Signal.enter_long(" in py_code or "Signal.enter_short(" in py_code

    # Verify code passes Gate 2 policy scan
    from validation.gates import gate_policy_scan
    scan_res = gate_policy_scan(py_code)
    assert scan_res.passed, f"Security scan failed: {scan_res}"


def test_dsl_validator_rejects_negative_lookahead_offset():
    """Verify validator strictly rejects negative offset referencing future bars."""
    dsl = StrategyDSL(
        name="Malicious_Lookahead_DSL",
        market="XAUUSD",
        family=StrategyFamily.TREND.value,
        hypothesis="Future lookahead cheat",
        timeframes=TimeframeConfig(primary="1h"),
        indicators=[IndicatorDef(id="ema", type="ema", params={"period": 20})],
        entry_rules=[
            EntryRule(
                direction="LONG",
                all_conditions=[{"left": "close", "operator": ">", "right": "ema", "offset": -1}],
            )
        ],
        exit=ExitRule(stop_loss_multiplier=1.5, take_profit_ratio=2.0),
    )
    val_res = DSLValidator().validate(dsl)
    assert not val_res.is_valid
    assert any("offset must be non-negative" in err for err in val_res.errors)


# ─── 2. Strategy Fingerprinting & Dedup ──────────────────────────────────────

def test_fingerprinting_l1_l2_l3():
    """Verify L1 (exact), L2 (logical), and L3 (behavioral) duplicate detection."""
    fingerprinter = StrategyFingerprinter()

    code_a = "def on_bar(bars): return Signal.from_prices(close, 1, 10, 20)"
    code_b = "def on_bar(bars):\n    # formatting comment\n    return Signal.from_prices(close, 1, 10, 20)"
    code_c = "def on_bar(bars): return Signal.from_prices(close, -1, 10, 20)"

    # L1 Exact hash
    h_a = fingerprinter.exact_hash(code_a)
    h_b = fingerprinter.exact_hash(code_b)
    h_c = fingerprinter.exact_hash(code_c)
    assert h_a == h_b, "L1 should normalize whitespace and comments"
    assert h_a != h_c

    # L2 Logical hash
    log_a = fingerprinter.logical_hash({"indicators": ["ema_50", "rsi_14"], "entry": ["long_c1"]})
    log_b = fingerprinter.logical_hash({"entry": ["long_c1"], "indicators": ["rsi_14", "ema_50"]})
    assert log_a == log_b, "L2 should be invariant to key/item ordering"

    # L3 Behavioral similarity
    pos1 = np.array([0, 1, 1, 1, 0, -1, -1, 0, 0, 1], dtype=np.int8)
    pos2 = np.array([0, 1, 1, 1, 0, -1, -1, 0, 0, 1], dtype=np.int8)
    pos3 = np.array([0, -1, -1, 0, 0, 1, 1, 1, 0, 0], dtype=np.int8)

    sim_identical = fingerprinter.behavioral_similarity(pos1, pos2)
    sim_different = fingerprinter.behavioral_similarity(pos1, pos3)
    assert sim_identical == 1.0
    assert sim_different < 0.5


# ─── 3. Prompt Firewall ─────────────────────────────────────────────────────

def test_prompt_firewall_sanitizes_holdout_and_secrets():
    """Prompt Firewall must strip holdout statistics and credential leaks."""
    firewall = PromptFirewall()

    dirty_context = {
        "status": "FAIL",
        "primary_failure": "LOW_PROFIT_FACTOR",
        "gross_pnl": 5000.0,
        "holdout_sharpe": 1.45,
        "holdout_trades": 34,
        "holdout_pnl": 12400.0,
        "api_key": "xkiro_live_secret_key_12345",
        "train_pf": 1.18,
    }

    clean = firewall.sanitize_context(dirty_context)
    assert "holdout_sharpe" not in clean
    assert "holdout_trades" not in clean
    assert "holdout_pnl" not in clean
    assert "api_key" not in clean
    assert clean["train_pf"] == 1.18
    assert clean["status"] == "FAIL"

    prompt_text = "Here is holdout performance: Sharpe 1.45, key xkiro_abc123"
    sanitized_text = firewall.sanitize_text(prompt_text)
    assert "[REDACTED_HOLDOUT]" in sanitized_text or "holdout" not in sanitized_text.lower()
    assert "xkiro_abc123" not in sanitized_text


# ─── 4. XKiro / Qwen 3.8 Max Provider ────────────────────────────────────────

def test_xkiro_provider_strips_thinking_blocks():
    """Ensure XKiro provider removes internal reasoning blocks (<think>...)."""
    provider = XKiroProvider(api_key="mock_key")
    raw_response = "<think>\nLet us analyze XAUUSD market structure...\n</think>\n{\n  \"action\": \"BUY\"\n}"
    cleaned = provider.strip_reasoning(raw_response)
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    assert "Let us analyze" not in cleaned
    assert "{\n  \"action\": \"BUY\"\n}" in cleaned


# ─── 5. Failure Analyzer & Strategy Autopsy ──────────────────────────────────

def test_failure_analyzer_classifies_repairable_vs_fatal():
    """FailureAnalyzer must distinguish repairable edge issues from fatal errors."""
    analyzer = FailureAnalyzer()

    # Case A: Low PF (repairable edge)
    res_low_pf = {
        "metrics": {
            "total_trades": 120,
            "profit_factor": 1.15,
            "sharpe": 0.65,
            "max_drawdown": 0.12,
            "cost_gross_ratio": 0.35,
        }
    }
    autopsy_a = analyzer.analyze("S001", "TREND", res_low_pf)
    assert autopsy_a.status == "FAIL_REPAIRABLE"
    assert autopsy_a.primary_failure == FailureCategory.LOW_PROFIT_FACTOR.value
    assert autopsy_a.is_repairable is True
    assert len(autopsy_a.improvement_guidance) > 10

    # Case B: Severe cost drag (> 70% of gross)
    res_cost = {
        "metrics": {
            "total_trades": 250,
            "profit_factor": 0.95,
            "sharpe": 0.10,
            "max_drawdown": 0.18,
            "cost_gross_ratio": 0.78,
        }
    }
    autopsy_b = analyzer.analyze("S002", "MOMENTUM", res_cost)
    assert autopsy_b.primary_failure == FailureCategory.COST_SENSITIVE.value
    assert autopsy_b.is_repairable is True

    # Case C: Zero trades (non-repairable signal deadlock)
    res_no_trades = {"metrics": {"total_trades": 0}}
    autopsy_c = analyzer.analyze("S003", "BREAKOUT", res_no_trades)
    assert autopsy_c.primary_failure == FailureCategory.NO_TRADES.value
    assert autopsy_c.is_repairable is False


# ─── 6. Research Director & Family Diversity ─────────────────────────────────

def test_research_director_maintains_diversity():
    """Research Director must allocate research across taxonomy without winner-take-all loops."""
    div_mgr = DiversityManager()
    allocations = div_mgr.get_target_allocations()
    assert sum(allocations.values()) == pytest.approx(1.0, 0.01)
    assert StrategyFamily.LIQUIDITY in allocations
    assert StrategyFamily.MARKET_STRUCTURE in allocations

    # Register multiple trend experiments
    for _ in range(15):
        div_mgr.record_experiment(StrategyFamily.TREND, passed=False)

    # Next recommendation should explore an under-represented family
    next_fam = div_mgr.select_next_family()
    assert next_fam != StrategyFamily.TREND, "Should favor under-represented family"


# ─── 7. Evolution Engine (Genome, Mutation, Crossover, Pareto) ───────────────

def test_evolution_mutation_and_crossover():
    """Genome mutations and crossovers must maintain valid schema and bounds."""
    genome_a = Genome(
        genome_id="g_a",
        family=StrategyFamily.TREND.value,
        timeframe="1h",
        parameters={
            "ema_fast": ParameterGene(name="ema_fast", value=20, min_val=10, max_val=50, step=2),
            "sl_mult": ParameterGene(name="sl_mult", value=1.5, min_val=1.0, max_val=3.0, step=0.1),
        },
        genes=[Gene(gene_type="indicator", name="ema_20", data={"period": 20})],
    )

    # Mutation
    mutated = mutate_genome(genome_a, mutation_rate=1.0)
    assert mutated.genome_id != genome_a.genome_id
    assert mutated.generation == genome_a.generation + 1
    assert 10 <= mutated.parameters["ema_fast"].value <= 50

    # Crossover
    genome_b = Genome(
        genome_id="g_b",
        family=StrategyFamily.TREND.value,
        timeframe="1h",
        parameters={
            "ema_fast": ParameterGene(name="ema_fast", value=40, min_val=10, max_val=50, step=2),
            "sl_mult": ParameterGene(name="sl_mult", value=2.2, min_val=1.0, max_val=3.0, step=0.1),
        },
        genes=[Gene(gene_type="indicator", name="ema_40", data={"period": 40})],
    )
    child = crossover_genomes(genome_a, genome_b)
    assert child.generation == max(genome_a.generation, genome_b.generation) + 1
    assert child.parameters["ema_fast"].value in (20, 40)


def test_pareto_ranking():
    """Multi-objective Pareto ranking must assign rank 1 to non-dominated candidates."""
    candidates = [
        {"id": "A", "sharpe": 1.5, "profit_factor": 1.6, "max_drawdown": 0.08},  # Dominant
        {"id": "B", "sharpe": 0.8, "profit_factor": 1.2, "max_drawdown": 0.15},  # Dominated by A
        {"id": "C", "sharpe": 1.7, "profit_factor": 1.3, "max_drawdown": 0.10},  # Non-dominated tradeoff
    ]
    ranked = pareto_rank(candidates)
    assert ranked["A"] == 1
    assert ranked["C"] == 1
    assert ranked["B"] > 1


# ─── 8. Strategy Library State Machine ───────────────────────────────────────

def test_strategy_lifecycle_state_machine():
    """State machine transitions must follow strict ordering; frozen strategies immutable."""
    library = StrategyLibrary()
    strat_id = "Gold_OrderBlock_15m"

    library.register_strategy(strat_id, spec_hash="hash_abc123")
    assert library.get_state(strat_id) == StrategyState.GENERATED

    library.transition(strat_id, StrategyState.COMPILED)
    library.transition(strat_id, StrategyState.SECURITY_PASS)
    library.transition(strat_id, StrategyState.BACKTESTED)
    library.transition(strat_id, StrategyState.SCREENED)
    library.transition(strat_id, StrategyState.ROBUST)
    library.transition(strat_id, StrategyState.FROZEN)

    assert library.is_frozen(strat_id) is True

    # Mutating code after frozen must raise PermissionError
    with pytest.raises(PermissionError):
        library.modify_spec(strat_id, new_spec_hash="tampered_hash")


# ─── 9. Portfolio Engine ────────────────────────────────────────────────────

def test_portfolio_allocation_and_correlation():
    """Portfolio engine must calculate correlation and allocate weights correctly."""
    np.random.seed(42)
    returns = {
        "S1": pd.Series(np.random.normal(0.001, 0.01, 100)),
        "S2": pd.Series(np.random.normal(0.001, 0.01, 100)),
        "S3": pd.Series(np.random.normal(0.001, 0.02, 100)),
    }

    corr_df = calculate_correlation_matrix(returns)
    assert corr_df.shape == (3, 3)
    assert corr_df.loc["S1", "S1"] == 1.0

    # Test Equal Weight
    w_eq = allocate_portfolio(returns, method=AllocationMethod.EQUAL_WEIGHT)
    assert sum(w_eq.values()) == pytest.approx(1.0, 0.001)
    assert w_eq["S1"] == pytest.approx(1.0 / 3.0, 0.001)

    # Test Inverse Volatility
    w_inv = allocate_portfolio(returns, method=AllocationMethod.INVERSE_VOLATILITY)
    assert sum(w_inv.values()) == pytest.approx(1.0, 0.001)
    # S3 has twice the vol of S1/S2, so its weight must be lower
    assert w_inv["S3"] < w_inv["S1"]
