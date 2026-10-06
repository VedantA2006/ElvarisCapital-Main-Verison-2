# ELVARIS CAPITAL — STATISTICAL ROBUSTNESS SPECIFICATION (V3)

---

## 1. Robustness Philosophy

The objective of systematic quant research is not to discover high historical returns, but to **discover hypotheses that survive rigorous adversarial attempts to disprove them**. Overfitting is the default outcome of unconstrained optimization.

QuantForge V3 evaluates candidates through an adversarial 19-gate validation sieve:

```
┌─────────────────────────────────────────────────────────────┐
│                    ADVERSARIAL GATES                        │
├───────────────────┬─────────────────────────────────────────┤
│ Gates 1–3         │ Schema integrity, AST security,         │
│ Foundations       │ Determinism verification                │
├───────────────────┼─────────────────────────────────────────┤
│ Gates 4–6         │ Execution realism: Min trades (>=80),   │
│ Performance       │ Profit Factor (>=1.25), Sharpe (>=0.80) │
├───────────────────┼─────────────────────────────────────────┤
│ Gates 7–10        │ Downside protection: Max DD (<=20%),    │
│ Risk & Stability  │ Profit concentration, Time consistency  │
├───────────────────┼─────────────────────────────────────────┤
│ Gate 11–12        │ Cost Resilience: Net PnL must remain    │
│ Friction Drag     │ positive at 1.5x and 2.0x spread stress │
├───────────────────┼─────────────────────────────────────────┤
│ Gate 13           │ 1-Bar Execution Latency Delay           │
├───────────────────┼─────────────────────────────────────────┤
│ Gates 14–16       │ Rolling Walk-Forward Analysis (WFA)     │
│ Overfit Defense   │ Fold consistency and efficiency ratio   │
├───────────────────┼─────────────────────────────────────────┤
│ Gates 17–18       │ Monte Carlo Reshuffling & Bootstrap     │
│ Uncertainty       │ 95th percentile DD <= 30%               │
├───────────────────┼─────────────────────────────────────────┤
│ Gate 19           │ Deflated Sharpe Ratio (DSR >= 0.95)     │
│ Multiple Testing  │ Probability of Backtest Overfit (PBO)   │
└───────────────────┴─────────────────────────────────────────┘
```

---

## 2. Adversarial Cost Stress (Gate 12)

Because gold spot trading experiences liquidity shocks, news spikes, and rollover spread expansion, strategies cannot rely on razor-thin execution margins.

The cost stress gate re-simulates the exact trade sequence under 3 stress regimes:
1. **Spread Stress ($1.5\times$ and $2.0\times$)**: Base spread ($0.25$ USD) scaled to $0.375$ and $0.50$ USD.
2. **Slippage Expansion ($2.0\times$)**: Slippage increased to $0.10 \times ATR$.
3. **Execution Delay ($1$ bar)**: Orders executed with 1-bar latency delay.

A strategy that collapses under a 1-bar delay or $1.5\times$ spread is classified as `COST_SENSITIVE` and failed.

---

## 3. Rolling Walk-Forward Analysis (WFA)

Evaluates out-of-sample persistence across sliding chronological folds:
- **Anchored / Rolling Folds**: 5 rolling train/validation splits.
- **Walk-Forward Efficiency (WFE)**:
  $$WFE = \frac{Sharpe_{Out-of-Sample}}{Sharpe_{In-Sample}}$$
- A strategy must demonstrate positive returns in at least $70\%$ of folds and maintain $WFE \ge 0.50$.

---

## 4. Monte Carlo Simulation Suite

Simulates 1,000 synthetic trajectory paths:
1. **Trade Reshuffling**: Random permutation of trade order without replacement (tests path-dependency of drawdowns).
2. **Bootstrap Sampling**: Resampling trades with replacement (tests statistical distribution of expectancy).
3. **Slippage Perturbation**: Injects stochastic slippage shocks up to $3\times$ base.

Reported metrics: $5\text{th}$, $25\text{th}$, $50\text{th}$ (median), $75\text{th}$, and $95\text{th}$ percentile drawdowns. The $95\text{th}$ percentile drawdown must not exceed $30.0\%$.

---

## 5. Deflated Sharpe Ratio (DSR) & PBO

Accounts for data mining bias and multiple testing across $N$ trials:
- **DSR**: Tests whether the observed Sharpe ratio exceeds the expected maximum Sharpe ratio of $N$ independent random strategies, adjusting for skewness and kurtosis. Requires $DSR \ge 0.95$.
- **PBO**: Combinatorially Symmetric Cross-Validation (CSCV) estimating the probability that the in-sample selected strategy performs below median out-of-sample. Requires $PBO \le 0.30$.
