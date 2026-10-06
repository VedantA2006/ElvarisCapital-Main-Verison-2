# ELVARIS CAPITAL — PORTFOLIO ALLOCATION ENGINE SPECIFICATION (V3)

---

## 1. Overview & Objectives

Once individual strategies have survived the 19-gate adversarial gauntlet and reached `PORTFOLIO_ELIGIBLE` status, the Portfolio Engine (`portfolio/`) aggregates them into an institutional multi-strategy portfolio.

The portfolio engine's mandate is to maximize Sharpe and minimize maximum drawdown through genuine mathematical diversification, strictly penalizing strategies that exhibit high return correlation or simultaneous position overlap.

---

## 2. Multi-Strategy Correlation (`portfolio/correlation.py`)

The engine computes three dimensions of portfolio relationship:
1. **Daily Return Correlation**: Pearson correlation of daily marked-to-market returns.
2. **Position Overlap**: Fraction of bars where two strategies simultaneously maintain open positions in the same direction.
3. **Drawdown Coincidence**: Overlap of peak-to-trough drawdown periods.

Two strategies with different code but $>0.70$ return correlation are grouped into the same risk cluster, preventing double allocation to identical risk factors.

---

## 3. Allocation Algorithms (`portfolio/allocator.py`)

The engine implements 4 institutional weighting models:

### 1. Equal Weight (`equal_weight`)
$$w_i = \frac{1}{N}$$
Simple naive diversification baseline.

### 2. Inverse Volatility (`inverse_vol`)
$$w_i = \frac{\frac{1}{\sigma_i}}{\sum_{j=1}^N \frac{1}{\sigma_j}}$$
Normalizes risk by scaling down high-volatility strategies.

### 3. Risk Parity (`risk_parity`)
Equalizes each strategy's marginal risk contribution to total portfolio variance:
$$RC_i = w_i \times \frac{(\Sigma w)_i}{\sigma_{port}} = \frac{1}{N} \times \sigma_{port}$$

### 4. Maximum Sharpe Optimization (`max_sharpe`)
Mean-variance optimization subject to non-negativity and institutional exposure constraints:
$$\max_{w} \frac{w^T \mu}{\sqrt{w^T \Sigma w}} \quad \text{s.t.} \quad \sum w_i = 1, \quad 0 \le w_i \le w_{max}$$

---

## 4. Institutional Risk Budget & Constraints (`portfolio/risk.py`)

All allocations are validated against strict institutional caps:
- **Max Strategy Weight**: No single strategy may exceed $35.0\%$ of portfolio capital.
- **Max Correlated Cluster Weight**: Total exposure to strategies with pairwise correlation $>0.60$ is capped at $50.0\%$.
- **Max Portfolio Drawdown Ceiling**: Sizing scales down dynamically if portfolio drawdown approaches $15.0\%$.
