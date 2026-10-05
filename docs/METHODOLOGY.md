# QuantForge Quantitative Methodology & Mathematical Specification

## 1. Overview and Core Philosophy

QuantForge is an autonomous quantitative discovery and verification engine designed specifically for the spot gold market (**XAUUSD**). In quantitative trading, empirical backtest results are notoriously vulnerable to false discoveries resulting from:
1. **Lookahead bias and subtle information leakage** (peeking at future bars, same-bar execution ambiguity).
2. **Execution under-specification** (unrealistic fills, ignoring spreads, slippage, and overnight swap costs).
3. **Overfitting and p-hacking** (evaluating hundreds of parameters or hypotheses without adjusting statistical significance).
4. **Behavioral redundancy** (accumulating multiple strategies that trade the exact same economic edge under different cosmetic code variations).

To eradicate these vulnerabilities, QuantForge enforces an uncompromising, non-negotiable **19-Stage Gate Architecture**, followed by **Strict One-Time Holdout Verification**, **Pluggable Forward Paper Trading**, and **Rolling Bootstrapped Decay Monitoring**.

---

## 2. Market Model, Pricing, and Cost Model

### 2.1 Bid/Ask Price Basis
All market data series ingested by QuantForge are bid-basis. Bid prices represent the actual price at which market participants sell, while ask prices represent the price at which they buy:
$$\text{Ask}_t = \text{Bid}_t + \text{Spread}_t$$

- **Long Entries:** Fill at the current Ask price ($\text{Bid}_t + \text{Spread}_t + \text{Slippage}$).
- **Long Exits:** Fill at the current Bid price ($\text{Bid}_t - \text{Slippage}$).
- **Short Entries:** Fill at the current Bid price ($\text{Bid}_t - \text{Slippage}$).
- **Short Exits:** Fill at the current Ask price ($\text{Bid}_t + \text{Spread}_t + \text{Slippage}$).

### 2.2 Spread Regimes
Spread is modeled dynamically per bar based on market session and time of day (New York time):
- **London / New York Session (Normal):** $\$0.25$ per ounce ($2.5$ pips).
- **Asia Session:** $\$0.35$ per ounce ($3.5$ pips).
- **Daily Rollover Window (16:00 – 19:00 NY):** Widened to $\$0.50 – \$0.60$ per ounce to reflect illiquidity during the end-of-day clearing transition.

### 2.3 Slippage Model
Slippage is dynamic and proportional to current market volatility, parameterized by the 20-period Average True Range ($\text{ATR}_{20}$):
$$\text{Slippage}_{\text{base}} = \max\left(0, \text{slip\_atr\_fraction} \times \text{ATR}_{20}\right)$$
For stop-loss executions (which execute into liquidity deficits during adverse price movement), slippage is magnified:
$$\text{Slippage}_{\text{stop}} = 1.5 \times \text{Slippage}_{\text{base}}$$

### 2.4 Overnight Swap Costs
Overnight financing costs (swap) are charged on positions open at **17:00 New York time** on trading days:
- **Long Swap:** $-\$3.50$ per lot per night (cost).
- **Short Swap:** $+\$0.80$ per lot per night (credit).
- **Triple Swap Wednesday:** On Wednesday rollovers, swap is multiplied by $3\times$ to account for weekend settlement. Weekend rollovers are not charged as markets are closed.

### 2.5 Cost Reconciliation Invariant
For every closed trade $i$, gross profit and net profit must reconcile with zero discrepancy:
$$\text{Gross P\&L}_i - \text{Spread Cost}_i - \text{Slippage Cost}_i - \text{Commission}_i - \text{Swap Cost}_i \equiv \text{Net P\&L}_i$$

---

## 3. Signal Representation and Invariants

Strategies output signals as structured `Signal` instances rather than unverified order requests:
```python
@dataclass
class Signal:
    action: Action          # ENTER_LONG, ENTER_SHORT, CLOSE, NONE
    sl_distance: float     # Stop-loss distance in USD (> 0)
    tp_distance: float     # Take-profit distance in USD (> 0)
    trail_distance: float  # Trailing stop distance in USD (optional)
    time_stop_bars: int    # Maximum holding period in bars
    breakeven_after_r: float
```

### Critical Signal Invariants:
1. **Relative Distances, Not Absolute Prices:** Strategies must specify `sl_distance > 0` and `tp_distance > 0`. Absolute prices are strictly forbidden to eliminate inverted-stop exploits.
2. **Minimum SL Distance Floor:**
   $$\text{sl\_distance} \ge \max\left(\$3.00, 1.0 \times \text{ATR}_{20}\right)$$
3. **Minimum TP Distance Floor:**
   $$\text{tp\_distance} \ge \max\left(\$5.00, 3.0 \times \text{Spread}\right)$$
   This prevents unrealistic micro-scalping where strategies attempt to harvest bid-ask bounce.
4. **Execution Precedence on Same Bar:** If both Stop-Loss and Take-Profit price levels are touched within the high-low range of the same bar, **Stop-Loss is always hit first** (conservative worst-case assumption).

---

## 4. The 19-Stage Gate Suite

Every candidate strategy must sequentially survive 19 rigorous, independent quantitative rejection stages:

### Stage 1: Static Policy Check
AST analysis inspecting the strategy code for forbidden imports, system access, disk I/O, network sockets, environment variables, or global mutation.

### Stage 2: Compilation & Sandbox Smoke Test
Compiles the strategy in an isolated sandbox subprocess, executing 5 warmup bars to verify syntax, execution time ($< 50\text{ms}$ per bar), and zero unhandled exceptions.

### Stage 3: Determinism Invariance
Executes the strategy twice over identical data slices. The resulting trade sequence and equity curves must be mathematically identical (correlation $\equiv 1.000000$).

### Stage 4: Truncation Invariance
Prepends 50 random historical bars prior to the evaluation slice. The generated signals on the evaluation slice must remain identical, proving the strategy does not depend on bar history length anomalies.

### Stage 5: 1-Bar Delay Shift Resilience
Artificially delays strategy signal execution by 1 closed bar. Strategies relying on fleeting, unexecutable micro-edges or lookahead leakage immediately collapse under this delay. Required delay Sharpe retention: $\ge 60\%$.

### Stage 6: Canary Exploits Verification
Executes the canary test suite (planted deliberate lookahead cheats such as reading future highs, tomorrow's close, or unshifted moving averages). All canaries must be caught and rejected.

### Stage 7: Basic Quality Filters
- Total completed trades: $N \ge 50$
- Win Rate: $\text{WR} \ge 25\%$
- Expectancy: $\mathbb{E}[R] > 0$

### Stage 8: In-Sample Sharpe & Profit Factor
- In-sample annualized Sharpe Ratio: $\text{Sharpe} \ge 0.80$
- Profit Factor: $\text{PF} \ge 1.20$
- Maximum Drawdown: $\text{MDD} \le 20.0\%$

### Stage 9: Deflated Sharpe Ratio (DSR)
Calculates the Deflated Sharpe Ratio per Bailey & López de Prado (2014), accounting for multiple testing, non-normal returns, skewness ($\gamma_3$), and kurtosis ($\gamma_4$):
$$\text{DSR} \equiv \Phi\left(\frac{(\widehat{\text{SR}} - \text{SR}^*)\sqrt{T-1}}{\sqrt{1 - \widehat{\gamma}_3 \widehat{\text{SR}} + \frac{\widehat{\gamma}_4 - 1}{4}\widehat{\text{SR}}^2}}\right)$$
where $\text{SR}^* = \sqrt{2\ln(N_{\text{trials}})}\left(1 - \frac{\gamma}{\sqrt{2\ln(N_{\text{trials}})}}\right)$.
**Requirement:** $\text{DSR Probability} \ge 0.95$ (tightened to $0.99$ if population overfitting is detected).

### Stage 10: Combinatorially Symmetric Cross-Validation (CSCV) & PBO
Constructs an aligned daily return matrix across candidate strategies, partitioning into 16 blocks to generate $\binom{16}{8} = 12,870$ combinatorial training/validation splits.
Computes Probability of Backtest Overfitting (PBO):
$$\text{PBO} = \frac{1}{K}\sum_{k=1}^K \mathbb{I}\left[\lambda_k \le 0\right], \quad \lambda_k = \ln\left(\frac{\omega_k}{1 - \omega_k}\right)$$
**Requirement:** $\text{PBO} \le 0.30$. If $\text{PBO} > 0.50$, flags a global `population_overfit_warning`.

### Stage 11: Monte Carlo Trade Shuffling
Performs $N=1,000$ permutations of historical trade orders to compute empirical drawdown distributions:
$$\text{Drawdown}_{95\text{th percentile}} \le 25.0\%$$
$$\text{P5 Return} > 0$$

### Stage 12: Parameter Plateau Sensitivity Stress
Perturbs strategy parameters by $\pm 10\%$ and $\pm 20\%$. Sharp cliffs in performance indicate curve-fitting; performance must remain robust across the parameter neighborhood.

### Stage 13: Cost Multiplier Stress
Multiplies spreads and slippages by $1.5\times$ and $2.0\times$. Strategy net profit must remain positive under double-cost execution.

### Stage 14: Walk-Forward Efficiency (WFE)
Partitions data into rolling out-of-sample segments.
$$\text{WFE} = \frac{\text{Sharpe}_{\text{OOS}}}{\text{Sharpe}_{\text{IS}}} \ge 0.50$$

### Stage 15: Regime & Yearly Breadth
Evaluates performance separately across volatility regimes (high ATR vs low ATR) and calendar years. A strategy must not derive its entire lifetime profit from a single outlier year or crisis event.

### Stage 16: Random-Baseline Null Distribution
Constructs 1,000 random entry strategies with identical holding time distributions. The candidate strategy must outperform $\ge 95\%$ of random baselines ($p < 0.05$).

### Stage 17: Validation Split Non-Decline
Evaluates the strategy on the un-optimized Validation split.
$$\text{Sharpe}_{\text{val}} \ge 0.60 \times \text{Sharpe}_{\text{train}}$$

### Stage 18: Diversity & Redundancy Filtering
- **AST Jaccard Similarity:** Compares abstract syntax tree tokens against existing pool ($< 0.85$).
- **TF-IDF Logic Cosine:** Compares indicator keyword signatures against existing pool ($< 0.80$).
- **Return Stream Correlation:**
  $$\rho\left(R_{\text{candidate}}, R_{\text{existing}}\right) \le 0.70$$

### Stage 19: Candidate Promotion
Promotes candidate to the candidate pool with status `CANDIDATE (unproven)`.

---

## 5. Holdout and Forward Testing Protocols

### 5.1 One-Time Holdout Evaluation
- **Access Rule:** Strict single-access guarantee. The holdout dataset is accessed exclusively through `DataStore.get_data(tf, "holdout", strategy_id=id)` with mandatory atomic logging.
- **Pass Requirement:** Must produce $\text{Net Profit} > 0$ AND $\text{Sharpe}_{\text{holdout}} \ge 0.50 \times \text{Sharpe}_{\text{train}}$.
- **Failure:** Any failure permanently marks `holdout_failed = True`. The holdout dataset is burned for that strategy forever.
- **LLM Isolation:** Holdout numbers are strictly excluded from LLM context and prompts.

### 5.2 Pluggable Forward Paper Trading
Promoted strategies that pass holdout proceed to forward paper trading on incoming closed bars via `PaperTrader`.
- Full execution simulation (spread, slippage, commission, swap).
- Trades written in real-time to `forward_trades`.

### 5.3 Statistical Decay Monitoring
Bootstraps $N=1,000$ resamples of in-sample trade returns to construct empirical $(5\%, 95\%)$ confidence bands for mean return.
If rolling forward performance breaches the lower $5\text{th}$ percentile bound for 10 consecutive trades, the strategy is marked `DEGRADED`.

### 5.4 Non-Negotiable LIVE-READY Gate
A strategy may transition from `CANDIDATE (unproven)` to `LIVE_READY` **ONLY IF**:
1. Forward paper testing duration $\ge 60$ calendar days.
2. Total completed forward trades $\ge 50$.
3. Decay monitor confirms strategy is not degraded.
4. Forward Profit Factor $> 1.0$ and forward Sharpe $> 0$.

**Hard Floor Guarantee:** These minimums ($60$ days, $50$ trades) are hardcoded into Python source code (`MIN_FORWARD_DAYS = 60`, `MIN_FORWARD_TRADES = 50`). No configuration setting, UI button, command flag, or manual override can lower them.
