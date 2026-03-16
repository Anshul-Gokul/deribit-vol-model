"""
Backtesting framework for options-implied probabilities.

Simulates the full pipeline historically:
1. Load historical options snapshots at forecast time
2. Build IV surface and extract density
3. Compute implied probabilities for target contracts
4. Compare against realized settlement outcomes
5. Evaluate forecast quality using proper scoring rules

NOTE: For a real backtest, you need historical options snapshots saved over time.
This module provides the framework; data collection over time is a separate concern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fetch_deribit import get_latest_snapshot, load_snapshot
from src.cleaning.options_cleaner import clean_options_data
from src.surface.iv_surface import IVSurface
from src.distribution.risk_neutral_density import extract_density
from src.evaluation.metrics import full_evaluation, calibration_bins
from src.utils.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""

    asset: str
    contract_type: str
    horizon: str
    start: datetime
    end: datetime
    n_forecasts: int = 0
    forecasts: list[dict] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecasts to DataFrame."""
        return pd.DataFrame(self.forecasts)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Backtest: {self.asset} {self.contract_type} ({self.horizon})",
            f"Period: {self.start.date()} to {self.end.date()}",
            f"Forecasts: {self.n_forecasts}",
        ]
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)


def _horizon_to_timedelta(horizon: str) -> timedelta:
    """Convert horizon string to timedelta."""
    mapping = {
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "daily": timedelta(days=1),
        "1d": timedelta(days=1),
        "weekly": timedelta(weeks=1),
        "1w": timedelta(weeks=1),
    }
    if horizon in mapping:
        return mapping[horizon]
    raise ValueError(f"Unknown horizon: {horizon}")


def run_backtest_from_snapshots(
    asset: str,
    snapshot_dir: str | Path,
    start: datetime,
    end: datetime,
    horizon: str = "daily",
    contract_type: str = "above",
    strike_offset_pct: float = 0.0,
    config: dict | None = None,
) -> BacktestResult:
    """
    Run backtest using saved historical snapshots.

    For each snapshot in the time range:
    1. Build IV surface
    2. Extract density for the target horizon
    3. Compute P(above strike) where strike = forward * (1 + offset)
    4. Record the probability and later check realized outcome

    Parameters
    ----------
    asset : str
        "BTC" or "ETH".
    snapshot_dir : Path
        Directory containing timestamped Parquet snapshots.
    start, end : datetime
        Backtest period.
    horizon : str
        Forecast horizon ("1h", "4h", "daily", "weekly").
    contract_type : str
        "above" for binary above-strike.
    strike_offset_pct : float
        Strike as offset from forward (0.0 = ATM, 0.05 = 5% OTM call).
    config : dict, optional
        Configuration overrides.
    """
    cfg = config or get_config()
    snapshot_dir = Path(snapshot_dir) / "raw" / asset.upper()

    if not snapshot_dir.exists():
        logger.error(f"Snapshot directory not found: {snapshot_dir}")
        return BacktestResult(
            asset=asset, contract_type=contract_type, horizon=horizon,
            start=start, end=end,
        )

    # Find all snapshot files in range
    snapshot_files = sorted(snapshot_dir.glob("options_*.parquet"))
    logger.info(f"Found {len(snapshot_files)} snapshot files")

    dt = _horizon_to_timedelta(horizon)
    forecasts = []

    for snap_path in snapshot_files:
        # Parse timestamp from filename
        try:
            ts_str = snap_path.stem.replace("options_", "")
            snap_time = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        if snap_time < start or snap_time > end:
            continue

        settlement_time = snap_time + dt

        try:
            # Load and clean
            df_raw = load_snapshot(snap_path)
            df_clean = clean_options_data(df_raw, as_of=snap_time, config=cfg)

            if df_clean.empty:
                continue

            # Build surface
            surface = IVSurface(config=cfg)
            surface.fit(df_clean)

            if not surface.slices:
                continue

            # Find appropriate T for the target settlement
            T_target = (settlement_time - snap_time).total_seconds() / (365.25 * 24 * 3600)
            if T_target <= 0:
                continue

            # Get forward for this T
            forward = surface._interpolate_forward(T_target)
            strike = forward * (1.0 + strike_offset_pct)

            # Extract density
            density = extract_density(surface, T_target, forward, config=cfg)

            # Compute probability
            if contract_type == "above":
                prob = density.prob_above(strike)
            elif contract_type == "below":
                prob = density.prob_below(strike)
            else:
                prob = density.prob_above(strike)

            forecasts.append({
                "forecast_time": snap_time.isoformat(),
                "settlement_time": settlement_time.isoformat(),
                "forward": forward,
                "strike": strike,
                "T": T_target,
                "implied_prob": prob,
                "realized_outcome": None,  # To be filled when settlement data is available
            })

        except Exception as e:
            logger.warning(f"Error processing snapshot {snap_path}: {e}")
            continue

    result = BacktestResult(
        asset=asset,
        contract_type=contract_type,
        horizon=horizon,
        start=start,
        end=end,
        n_forecasts=len(forecasts),
        forecasts=forecasts,
    )

    logger.info(f"Backtest produced {len(forecasts)} forecasts")
    return result


def evaluate_backtest(result: BacktestResult) -> BacktestResult:
    """
    Evaluate backtest results by comparing implied probs to realized outcomes.

    This requires the 'realized_outcome' field to be populated (1 if event occurred, 0 if not).
    """
    df = result.to_dataframe()
    if df.empty or "realized_outcome" not in df.columns:
        logger.warning("No forecasts to evaluate")
        return result

    # Filter to rows with realized outcomes
    df_eval = df.dropna(subset=["realized_outcome"])
    if df_eval.empty:
        logger.warning("No realized outcomes available for evaluation")
        return result

    probs = df_eval["implied_prob"].values
    outcomes = df_eval["realized_outcome"].values.astype(float)

    result.metrics = full_evaluation(probs, outcomes)
    return result


def simulate_backtest_from_single_snapshot(
    asset: str,
    data_dir: str | Path = "data",
    n_strikes: int = 20,
    config: dict | None = None,
) -> BacktestResult:
    """
    Simplified backtest simulation using a single current snapshot.

    This demonstrates the backtest framework by evaluating probability
    estimates across multiple strike levels from a single point in time.
    Not a proper time-series backtest — for illustration and testing only.
    """
    cfg = config or get_config()
    data_dir = Path(data_dir)

    snap_path = get_latest_snapshot(asset, "options", data_dir)
    if snap_path is None:
        raise FileNotFoundError(f"No snapshot found for {asset}")

    df_raw = load_snapshot(snap_path)
    now = datetime.now(timezone.utc)
    df_clean = clean_options_data(df_raw, as_of=now, config=cfg)

    if df_clean.empty:
        raise ValueError("No valid options after cleaning")

    surface = IVSurface(config=cfg)
    surface.fit(df_clean)

    # Use the nearest expiry
    T = surface.expiry_times[0]
    forward = surface._interpolate_forward(T)

    density = extract_density(surface, T, forward, config=cfg)

    # Generate strike grid
    strikes = np.linspace(forward * 0.7, forward * 1.3, n_strikes)

    forecasts = []
    for K in strikes:
        prob = density.prob_above(K)
        forecasts.append({
            "strike": K,
            "strike_pct": K / forward,
            "forward": forward,
            "T": T,
            "implied_prob": prob,
            "contract": f"{asset} > {K:.0f}",
        })

    return BacktestResult(
        asset=asset,
        contract_type="above",
        horizon=f"T={T:.4f}y",
        start=now,
        end=now,
        n_forecasts=len(forecasts),
        forecasts=forecasts,
    )
