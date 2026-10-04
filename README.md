# QuantForge

Autonomous strategy discovery engine for XAUUSD.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up secrets
cp .env.example .env
# Edit .env with your MONGO_URL and LLM_API_KEY_1

# Validate data and freeze splits
python main.py verify-data

# Run discovery loop (10 trials)
python main.py run --trials 10

# View the leaderboard
python main.py leaderboard

# Launch the dashboard
python main.py dashboard
```

## Architecture

```
quantforge/
├── core/           # backtester, sandbox, gates, lookahead, splits, config
├── llm/            # LLM client, prompts, orchestrator
├── storage/        # MongoDB helpers, structured logger
├── dashboard/      # single-page web dashboard
├── scripts/        # one-off utilities (patch_hole.py, etc.)
├── tests/          # pytest suite (runs offline with mongomock by default)
│   ├── regression/ # one test per audit defect (regression-first)
│   └── repro/      # executable exploit reproductions
├── docs/           # AUDIT.md and specifications
├── config.yaml     # all thresholds, costs, and operational settings
└── main.py         # CLI entry point
```

## Testing

```bash
# Default: offline, in-memory (no MongoDB needed)
pytest

# Against a real MongoDB
QF_TEST_MONGO=real pytest
```

## Audit Status

See [docs/AUDIT.md](docs/AUDIT.md) for the full defect register and
`tests/repro/repro_exploits.py` for executable reproductions.
