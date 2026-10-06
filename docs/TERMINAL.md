# ELVARIS CAPITAL — TERMINAL USER INTERFACE SPECIFICATION (V3)

---

## 1. Overview & Institutional Aesthetic

The QuantForge Terminal (`terminal/console.py`) provides an institutional quantitative research console built using Python's `Rich` framework. It eschews generic consumer styling in favor of high-information-density quantitative telemetry, monospaced numerical tables, and real-time state visualization.

---

## 2. Terminal Layout Specification (Section 54)

```
╔══════════════════════════════════════════════════════════════════════╗
║                 ELVARIS CAPITAL — QUANT LAB                         ║
║                     XAUUSD AUTONOMOUS R&D                           ║
╠══════════════════════════════════════════════════════════════════════╣
║ MODEL       QWEN 3.8 MAX     STATUS       AUTONOMOUS               ║
║ MARKET      XAUUSD           SESSION      #000142                  ║
║ TIMEFRAMES  5M 15M 1H 4H     UPTIME       04:31:27                ║
╠══════════════════════════════════════════════════════════════════════╣
║ RESEARCH DIRECTOR                                                  ║
║                                                                    ║
║ Current family : LIQUIDITY + MARKET STRUCTURE                     ║
║ Hypothesis     : NY liquidity sweep + displacement                ║
║ Generation     : 7                                                 ║
║ Lineage        : S00142 → S00142-7                                ║
╠══════════════════════════════════════════════════════════════════════╣
║ CURRENT STRATEGY                                                   ║
║                                                                    ║
║ Timeframe       15M                                               ║
║ Context         1H                                                ║
║ Trades          184                                               ║
║ PF              1.43                                              ║
║ Sharpe          1.12                                              ║
║ Max DD          8.7%                                              ║
║ Net P&L         +$18,420.00                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║ VALIDATION                                                         ║
║                                                                    ║
║ Lookahead       PASS       Determinism      PASS                  ║
║ Repainting      PASS       Cost Stress      PASS                  ║
║ Parameters      PASS       Walk Forward     PASS                  ║
║ Monte Carlo     PASS       DSR/PBO          PASS                  ║
╠══════════════════════════════════════════════════════════════════════╣
║ AUTONOMOUS LOOP TELEMETRY                                          ║
║                                                                    ║
║ [18:42:01] Formulated institutional liquidity sweep hypothesis     ║
║ [18:42:04] Strategy compiled into sandboxed AST representation    ║
║ [18:42:05] Gate 2 policy and lookahead scan PASSED                ║
║ [18:42:09] Event backtest completed across historical bars        ║
║ [18:42:10] Autopsy created: Edge confirmed, PF 1.43               ║
║ [18:42:15] Adversarial robustness testing initiated               ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 3. Comprehensive CLI Commands Reference (Section 56)

| Command | Action |
|:---|:---|
| `python main.py system health` | Runs complete system health audit (DB, LLM, Sandbox, Data, Resources) |
| `python main.py data verify` | Validates OHLC integrity, session boundaries, and split hashes |
| `python main.py db verify` | Verifies run/trial counter alignment with source-of-truth records |
| `python main.py db repair-counters` | Atomically recalculates global counters from immutable run records |
| `python main.py autonomous start` | Launches continuous autonomous quantitative research session |
| `python main.py autonomous start --dry-run` | Executes dry-run proving end-to-end loop without mutating state |
| `python main.py autonomous pause` | Gracefully pauses autonomous execution after current cycle |
| `python main.py autonomous resume` | Resumes paused research loop |
| `python main.py autonomous stop` | Safely halts autonomous research workers |
| `python main.py research dry-run` | Proves single autonomous research cycle in simulated mode |
| `python main.py strategy list` | Displays promoted and candidate strategies |
| `python main.py strategy show <id>` | Inspects complete strategy specification, metrics, and autopsy |
| `python main.py portfolio build` | Computes correlation matrix and optimal allocation weights |
