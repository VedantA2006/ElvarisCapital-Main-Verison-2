# ELVARIS CAPITAL — EVOLUTIONARY SEARCH ENGINE SPECIFICATION (V3)

---

## 1. Evolutionary Paradigm

The Evolution Engine (`evolution/`) provides an automated search mechanism that refines, mutates, and recombines candidate strategies. Rather than optimizing blindly against net profit, it conducts multi-objective Pareto optimization across risk-adjusted return, downside protection, cost resilience, and structural parsimony.

```
                      POPULATION (Generation g)
                                 │
                                 ▼
                     TIERED EVALUATION PIPELINE
             ┌───────────────────┬───────────────────┐
             │ Tier 1: Cheap     │ Basic Simulation  │
             │ Tier 2: Medium    │ Cost Stress       │
             │ Tier 3: Expensive │ Walk-Forward / MC │
             └───────────────────┴───────────────────┘
                                 │
                                 ▼
                    MULTI-OBJECTIVE PARETO RANKING
                 (Sharpe, PF, MaxDD, Cost Resilience)
                                 │
                                 ▼
                       SELECTION & ELITISM
                  (Non-dominated frontier kept)
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
          MUTATION OPERATORS          CROSSOVER OPERATORS
        (Param, SL/TP, Ind Swap)      (Recombine Setups)
                   │                           │
                   └─────────────┬─────────────┘
                                 ▼
                     OFFSPRING (Generation g+1)
```

---

## 2. Genetic Representation (`evolution/genome.py`)

A strategy is represented as a `StrategyGenome` wrapping its underlying `StrategyDSL`:
- **Genes**: Indicator definitions, entry rules, exit parameters, and parameter ranges.
- **Structural Complexity Penalty**:
  $$Complexity = 2 \times N_{indicators} + 1 \times N_{parameters} + 3 \times N_{conditions}$$
  Strategies with high complexity receive a fitness discount, ensuring that simpler models dominate equivalent complex models.

---

## 3. Genetic Operators

### Mutation (`evolution/mutation.py`)
- **Parameter Perturbation**: Modifies numerical thresholds by $\pm 10\%$ to $\pm 20\%$.
- **Stop Loss / Take Profit Tuning**: Adjusts ATR stop-loss multiplier and target risk-reward ratios.
- **Indicator Swapping**: Exchanges equivalent indicators (e.g., EMA $\leftrightarrow$ SMA $\leftrightarrow$ WMA, or Bollinger $\leftrightarrow$ Keltner) preserving type contracts.
- **Session Filter Tuning**: Modifies session active hours (`london_ny`, `ny`, `london`, `all`).

### Crossover (`evolution/crossover.py`)
- Recombines distinct parent strategies:
  - Takes Long entry setup from Parent A.
  - Takes Short entry setup from Parent B.
  - Inherits exit parameters from the parent with superior risk metrics.
  - Merges parameter spaces, strictly capping combined parameters to maximum 6.

---

## 4. Multi-Objective Fitness & Pareto Ranking (`evolution/fitness.py`)

Candidates are ranked using Non-Dominated Sorting across 4 core objectives:
1. **Sharpe Ratio** (Maximize)
2. **Profit Factor** (Maximize)
3. **Maximum Drawdown** (Minimize)
4. **Cost Resilience** ($1 - Cost/Gross$, Maximize)

Candidate $A$ dominates candidate $B$ if $A$ is greater than or equal to $B$ in all four metrics and strictly superior in at least one. Candidates on the non-dominated Pareto frontier receive Rank 1 and are preserved for subsequent generations.
