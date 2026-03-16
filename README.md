# Deribit Options-Implied Probability Engine

A production-grade Python system that uses **Deribit options prices** to infer market-implied
future price distributions for BTC and ETH, then maps those distributions into
prediction-market-style contract probabilities.

## Core Idea

Options prices embed the market's risk-neutral expectations about future price movements.
By extracting the implied volatility surface and computing the Breeden-Litzenberger
risk-neutral density, we can answer questions like:

- **P(BTC > 100,000 at expiry)** — fair price for a binary "above" contract
- **P(ETH between 2,400–2,550)** — fair price for a range contract
- **P(BTC outside [95,000–105,000])** — fair price for an "outside" contract

## Important: Risk-Neutral vs Real-World Probabilities

This system produces **risk-neutral (Q-measure) probabilities** implied by option prices,
**not** real-world (P-measure) probabilities. Key differences:

- Risk-neutral probabilities incorporate risk premia — they overweight adverse outcomes
- For assets with positive risk premium (like BTC), risk-neutral probabilities typically
  assign *more* weight to downside scenarios than physical probabilities
- These are "fair prices" for contracts in a no-arbitrage sense, not forecasts
- A simple heuristic adjustment layer is included but the core output is Q-measure

For prediction markets, risk-neutral probabilities are actually the right pricing tool
if you view the market as pricing contingent claims.

## Architecture

```
src/
  data/           # Deribit API fetching, caching, raw snapshots
  cleaning/       # Quote filtering, parsing, sanity checks
  surface/        # IV surface construction (spline + SVI)
  distribution/   # Risk-neutral density extraction (Breeden-Litzenberger)
  pricing/        # Probability queries, contract pricing, digitals
  backtest/       # Historical probability vs realized outcome evaluation
  evaluation/     # Brier score, log loss, calibration metrics
  cli/            # Typer CLI for all operations
  utils/          # Black-Scholes, time helpers
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Fetch current options data
python -m src.cli.main fetch-data --asset BTC

# Build IV surface from latest snapshot
python -m src.cli.main build-surface --asset BTC

# Query: probability BTC > 100000 at a future settlement
python -m src.cli.main query-prob \
  --asset BTC \
  --contract-type above \
  --strike 100000 \
  --settlement "2026-03-28T08:00:00Z"

# Query: probability BTC in range
python -m src.cli.main query-prob \
  --asset BTC \
  --contract-type between \
  --lower 95000 \
  --upper 105000 \
  --settlement "2026-03-28T08:00:00Z"

# Run backtest
python -m src.cli.main backtest \
  --asset BTC \
  --start "2025-06-01T00:00:00Z" \
  --end "2025-12-31T00:00:00Z" \
  --horizon daily

# Generate report
python -m src.cli.main make-report --asset BTC
```

## Design Choices

### IV Surface Construction
Two methods are supported:
1. **Cubic spline** interpolation in (log-moneyness, sqrt-time) space — fast, stable
2. **SVI parametric fit** (Gatheral's Stochastic Volatility Inspired) — better extrapolation

The spline method is default for robustness. SVI is available when enough liquid strikes exist.

### Density Extraction
The primary method is **Breeden-Litzenberger**: the risk-neutral density equals the
second derivative of call prices with respect to strike. We use:
- Smoothed call prices from the IV surface (not raw quotes) to avoid noise
- Central finite differences on a fine strike grid
- Density repair: clip negatives, re-normalize to integrate to 1
- Validation: check mean ≈ forward, variance is reasonable

### Expiry Mapping
Prediction markets settle at arbitrary timestamps. Options have fixed expiries.
The system:
1. Finds the two nearest Deribit expiries bracketing the target
2. Uses variance-linear interpolation (IV² scales linearly in time) between them
3. Falls back to nearest-expiry if only one side is available
4. Documents the approximation quality in output

### Quote Filtering
- Reject quotes with zero bid or ask
- Reject crossed markets (bid > ask)
- Reject extreme spreads (> 50% of mid)
- Reject deep OTM options with negligible premium
- Reject options violating basic no-arbitrage (call > S, put > K·exp(-rT))
- Prefer OTM options (calls for K > F, puts for K < F) to avoid early-exercise noise

## Configuration

Copy `.env.example` to `.env` and set your Deribit API credentials (optional for
public endpoints). Edit `configs/default.yaml` for model parameters.

## Testing

```bash
pytest tests/ -v
```

## Limitations

1. Deribit options are European-style — no early exercise complications
2. Short-dated options (< 1 day) have very wide spreads and unstable IVs
3. Deep OTM tails are poorly constrained by market data
4. The density in the far tails is extrapolated, not observed
5. Risk-neutral ≠ real-world: systematic bias exists
6. Liquidity varies enormously by strike and expiry
