"""
Spot price forecasting model from Deribit derivatives data.

Combines three layers of information:

Layer 1 — Forward Curve (the anchor)
    The futures-implied forward F(T) is the risk-neutral expected price.
    Under no-arbitrage this is the best unbiased estimator under Q.
    For crypto, the basis is small at short horizons so F ≈ S.

Layer 2 — Confidence Intervals from Options
    The risk-neutral density gives us a full distribution at each horizon.
    We extract quantiles (5th, 25th, 50th, 75th, 95th) to produce
    fan-chart style confidence bands around the forward.

Layer 3 — Directional Tilt from Skew & Flow
    Risk reversals, put-call ratios, and basis momentum can tilt the
    forecast slightly away from the pure forward.  This is the only
    component that moves us from Q to a P-like estimate.

The output is a SpotForecast object with:
    - point forecast (tilted forward)
    - confidence intervals at each horizon
    - the underlying signals
    - the full risk-neutral distribution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from src.models.forward_curve import ForwardCurve, build_forward_curve_from_futures, merge_forward_curves, build_forward_curve_from_options
from src.models.signals import DerivativesSignals, extract_signals
from src.surface.iv_surface import IVSurface
from src.distribution.risk_neutral_density import RiskNeutralDensity, extract_density
from src.data.fetch_deribit import get_latest_snapshot, load_snapshot
from src.cleaning.options_cleaner import clean_options_data

logger = logging.getLogger(__name__)


@dataclass
class HorizonForecast:
    """Forecast for a single time horizon."""
    label: str  # e.g. "1d", "7d", "30d"
    T: float  # years
    target_time: datetime

    # Point estimates
    forward: float  # risk-neutral forward (anchor)
    point_forecast: float  # tilted forecast (our best guess)
    tilt_pct: float  # (point - forward) / forward

    # Confidence intervals from risk-neutral density
    q05: float
    q25: float
    q50: float  # median
    q75: float
    q95: float

    # Volatility
    implied_vol: float  # ATM IV at this tenor
    expected_move_1sd: float  # forward * IV * sqrt(T)
    expected_move_pct: float

    # Density available for detailed queries
    density: RiskNeutralDensity | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "density"}
        d["target_time"] = self.target_time.isoformat()
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in d.items()}


@dataclass
class SpotForecast:
    """
    Complete spot price forecast from derivatives data.

    Contains forecasts at multiple horizons plus the signals and
    curve data that produced them.
    """
    asset: str
    spot: float
    as_of: datetime
    horizons: list[HorizonForecast] = field(default_factory=list)
    signals: DerivativesSignals | None = None
    forward_curve: ForwardCurve | None = None

    def get_horizon(self, label: str) -> HorizonForecast | None:
        """Get forecast for a specific horizon label."""
        for h in self.horizons:
            if h.label == label:
                return h
        return None

    def table(self) -> pd.DataFrame:
        """Return forecasts as a DataFrame."""
        rows = []
        for h in self.horizons:
            rows.append({
                "horizon": h.label,
                "T_days": round(h.T * 365.25, 1),
                "forward": round(h.forward, 2),
                "forecast": round(h.point_forecast, 2),
                "tilt": f"{h.tilt_pct:+.3f}%",
                "q05": round(h.q05, 0),
                "q25": round(h.q25, 0),
                "median": round(h.q50, 0),
                "q75": round(h.q75, 0),
                "q95": round(h.q95, 0),
                "IV": f"{h.implied_vol:.1%}",
                "1SD_move": f"±{h.expected_move_pct:.1f}%",
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Human-readable forecast summary."""
        lines = [
            f"{'='*70}",
            f"  SPOT PRICE FORECAST — {self.asset}",
            f"  As of: {self.as_of.strftime('%Y-%m-%d %H:%M UTC')}",
            f"  Spot:  {self.spot:,.2f}",
            f"{'='*70}",
        ]

        if self.signals:
            direction = "BULLISH" if self.signals.directional_score > 0.1 else \
                        "BEARISH" if self.signals.directional_score < -0.1 else "NEUTRAL"
            lines.append(f"  Directional Bias: {direction} "
                         f"(score={self.signals.directional_score:+.3f}, "
                         f"conf={self.signals.confidence:.0%})")
            lines.append("")

        lines.append(f"  {'Horizon':<8} {'Forward':>10} {'Forecast':>10} {'Tilt':>8} "
                     f"{'5%':>9} {'25%':>9} {'Median':>9} {'75%':>9} {'95%':>9} "
                     f"{'IV':>7} {'±1SD':>8}")
        lines.append(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8} "
                     f"{'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} "
                     f"{'-'*7} {'-'*8}")

        for h in self.horizons:
            lines.append(
                f"  {h.label:<8} {h.forward:>10,.0f} {h.point_forecast:>10,.0f} "
                f"{h.tilt_pct:>+7.3f}% "
                f"{h.q05:>9,.0f} {h.q25:>9,.0f} {h.q50:>9,.0f} "
                f"{h.q75:>9,.0f} {h.q95:>9,.0f} "
                f"{h.implied_vol:>6.1%} {h.expected_move_pct:>+7.1f}%"
            )

        lines.append("")
        lines.append("  Note: Forward = risk-neutral expected price (from futures/options)")
        lines.append("  Forecast = forward + directional tilt from skew/flow signals")
        lines.append("  Quantiles = risk-neutral confidence intervals from options density")

        if self.forward_curve:
            lines.append("")
            lines.append("  --- Forward Curve ---")
            for p in sorted(self.forward_curve.points, key=lambda x: x.T):
                lines.append(
                    f"    T={p.T*365.25:>5.0f}d  F={p.forward:>10,.2f}  "
                    f"basis={p.basis_pct:>+.3f}%  carry={p.annualized_basis:>+.2f}%/yr  "
                    f"[{p.source}]"
                )

        return "\n".join(lines)


def _apply_directional_tilt(
    forward: float,
    score: float,
    confidence: float,
    T: float,
    max_tilt_annual_pct: float = 5.0,
) -> float:
    """
    Apply a small directional tilt to the forward based on skew signals.

    The tilt is bounded and decays with sqrt(T) to avoid wild long-horizon shifts.

    Parameters
    ----------
    forward : float
        Risk-neutral forward price.
    score : float
        Directional score in [-1, 1].
    confidence : float
        Signal confidence in [0, 1].
    T : float
        Time horizon in years.
    max_tilt_annual_pct : float
        Maximum annualized tilt in percent (caps how far we deviate from forward).
    """
    # Scale tilt: max_tilt * score * confidence * sqrt(T)
    # sqrt(T) scaling means tilt grows sub-linearly with horizon
    tilt_annual = max_tilt_annual_pct / 100.0 * score * confidence
    tilt = tilt_annual * np.sqrt(T)
    return forward * (1.0 + tilt)


def build_forecast(
    asset: str,
    data_dir: str = "data",
    horizons: list[tuple[str, float]] | None = None,
    apply_tilt: bool = True,
    max_tilt_pct: float = 5.0,
) -> SpotForecast:
    """
    Build a complete spot price forecast from Deribit derivatives data.

    Parameters
    ----------
    asset : str
        "BTC" or "ETH".
    data_dir : str
        Path to data directory.
    horizons : list of (label, T_years) tuples
        Forecast horizons. Default: 1h, 4h, 1d, 3d, 7d, 14d, 30d.
    apply_tilt : bool
        Whether to apply directional tilt on top of forward.
    max_tilt_pct : float
        Maximum annualized tilt percentage.
    """
    if horizons is None:
        horizons = [
            ("1h", 1 / (365.25 * 24)),
            ("4h", 4 / (365.25 * 24)),
            ("1d", 1 / 365.25),
            ("3d", 3 / 365.25),
            ("7d", 7 / 365.25),
            ("14d", 14 / 365.25),
            ("30d", 30 / 365.25),
        ]

    now = datetime.now(timezone.utc)

    # Load data
    opt_snap = get_latest_snapshot(asset, "options", data_dir)
    fut_snap = get_latest_snapshot(asset, "futures", data_dir)

    if opt_snap is None:
        raise FileNotFoundError(f"No options snapshot for {asset}. Run fetch-data first.")

    df_opt_raw = load_snapshot(opt_snap)
    df_clean = clean_options_data(df_opt_raw, as_of=now)

    if df_clean.empty:
        raise ValueError(f"No valid options after cleaning for {asset}")

    spot = float(df_clean["spot"].iloc[0])

    # Build forward curve
    if fut_snap is not None:
        df_fut = load_snapshot(fut_snap)
        fwd_curve = build_forward_curve_from_futures(df_fut, spot, now)
    else:
        fwd_curve = build_forward_curve_from_options(df_clean, spot, now)

    # Also get options-implied forwards and merge
    opt_curve = build_forward_curve_from_options(df_clean, spot, now)
    if fwd_curve.points:
        fwd_curve = merge_forward_curves(fwd_curve, opt_curve)
    else:
        fwd_curve = opt_curve

    # Build IV surface
    surface = IVSurface()
    surface.fit(df_clean)

    # Extract signals
    signals = extract_signals(surface, fwd_curve, df_clean)

    # Build forecasts at each horizon
    horizon_forecasts = []
    for label, T in horizons:
        target_time = now + timedelta(days=T * 365.25)
        fwd = fwd_curve.forward(T)

        # Point forecast with optional tilt
        if apply_tilt:
            point = _apply_directional_tilt(
                fwd, signals.directional_score, signals.confidence,
                T, max_tilt_pct
            )
        else:
            point = fwd

        tilt_pct = (point - fwd) / fwd * 100

        # ATM IV
        atm_iv = _safe_atm_iv(surface, T)
        move_1sd = fwd * atm_iv * np.sqrt(T) if atm_iv > 0 else 0
        move_pct = move_1sd / fwd * 100 if fwd > 0 else 0

        # Extract density for quantiles
        try:
            density = extract_density(surface, max(T, 0.0005), fwd)
            q05 = density.quantile(0.05)
            q25 = density.quantile(0.25)
            q50 = density.quantile(0.50)
            q75 = density.quantile(0.75)
            q95 = density.quantile(0.95)
        except Exception as e:
            logger.warning(f"Could not extract density for {label}: {e}")
            density = None
            # Lognormal fallback
            if atm_iv > 0:
                s = atm_iv * np.sqrt(T)
                q05 = fwd * np.exp(-1.645 * s)
                q25 = fwd * np.exp(-0.674 * s)
                q50 = fwd
                q75 = fwd * np.exp(0.674 * s)
                q95 = fwd * np.exp(1.645 * s)
            else:
                q05 = q25 = q50 = q75 = q95 = fwd

        horizon_forecasts.append(HorizonForecast(
            label=label,
            T=T,
            target_time=target_time,
            forward=fwd,
            point_forecast=point,
            tilt_pct=tilt_pct,
            q05=q05,
            q25=q25,
            q50=q50,
            q75=q75,
            q95=q95,
            implied_vol=atm_iv,
            expected_move_1sd=move_1sd,
            expected_move_pct=move_pct,
            density=density,
        ))

    return SpotForecast(
        asset=asset,
        spot=spot,
        as_of=now,
        horizons=horizon_forecasts,
        signals=signals,
        forward_curve=fwd_curve,
    )


def _safe_atm_iv(surface: IVSurface, T: float) -> float:
    """Get ATM IV with fallback."""
    try:
        fwd = surface._interpolate_forward(T)
        iv = surface.iv(fwd, T, fwd)
        return iv if 0.01 < iv < 5.0 else 0.5
    except Exception:
        return 0.5


def forecast_at_timestamp(
    asset: str,
    target: datetime,
    data_dir: str = "data",
    apply_tilt: bool = True,
) -> HorizonForecast:
    """
    Produce a forecast for a specific future timestamp.

    Convenience function that builds the full model and extracts
    the single relevant horizon.
    """
    now = datetime.now(timezone.utc)
    T = max((target - now).total_seconds() / (365.25 * 24 * 3600), 0.0001)
    delta_hours = (target - now).total_seconds() / 3600
    label = f"{delta_hours:.0f}h"

    forecast = build_forecast(
        asset=asset,
        data_dir=data_dir,
        horizons=[(label, T)],
        apply_tilt=apply_tilt,
    )
    return forecast.horizons[0]
