"""
Forward curve extraction and interpolation from Deribit futures & options.

The forward price F(T) at each maturity T is the market's risk-neutral
expectation of the spot price at time T.  We build the curve from:

1. Futures mark prices (direct observation)
2. Put-call parity implied forwards where futures are missing
3. Cubic interpolation + flat extrapolation for arbitrary query times

The curve itself IS the market-implied future spot price under Q.
Under no-arbitrage:  F(T) = S * exp((r - q) * T)
but in crypto r ≈ 0 and there is no dividend, so any deviation from spot
reflects funding/basis/convenience-yield effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from src.utils.time_utils import parse_deribit_instrument, time_to_expiry_years

logger = logging.getLogger(__name__)


@dataclass
class ForwardPoint:
    """Single observed forward price at a maturity."""
    expiry: datetime
    T: float  # years to expiry
    forward: float  # USD
    source: str  # "futures" or "put_call_parity"
    basis_pct: float = 0.0  # (F - S) / S
    annualized_basis: float = 0.0  # basis_pct / T (annualized carry)


@dataclass
class ForwardCurve:
    """
    Fitted forward curve F(T) across all available maturities.

    Provides interpolation for any future timestamp and extracts
    the term structure of basis (implied financing/carry rate).
    """
    spot: float
    as_of: datetime
    points: list[ForwardPoint] = field(default_factory=list)
    _spline: CubicSpline | None = field(default=None, repr=False)

    def fit(self) -> None:
        """Fit cubic spline to the forward points."""
        if len(self.points) < 2:
            logger.warning(f"Only {len(self.points)} forward points — linear fallback")
            self._spline = None
            return

        pts = sorted(self.points, key=lambda p: p.T)
        # Prepend spot at T=0
        Ts = np.array([0.0] + [p.T for p in pts])
        Fs = np.array([self.spot] + [p.forward for p in pts])

        # Remove duplicates
        _, idx = np.unique(Ts, return_index=True)
        Ts, Fs = Ts[idx], Fs[idx]

        if len(Ts) < 2:
            self._spline = None
            return

        self._spline = CubicSpline(Ts, Fs, bc_type="natural", extrapolate=True)

    def forward(self, T: float) -> float:
        """Query forward price at time T (years)."""
        if T <= 0:
            return self.spot
        if self._spline is not None:
            return float(np.clip(self._spline(T), self.spot * 0.3, self.spot * 5.0))
        # Fallback: linear interpolation from nearest points
        if not self.points:
            return self.spot
        pts = sorted(self.points, key=lambda p: p.T)
        if T <= pts[0].T:
            alpha = T / pts[0].T if pts[0].T > 0 else 0
            return self.spot + alpha * (pts[0].forward - self.spot)
        if T >= pts[-1].T:
            return pts[-1].forward
        for i in range(len(pts) - 1):
            if pts[i].T <= T <= pts[i + 1].T:
                alpha = (T - pts[i].T) / (pts[i + 1].T - pts[i].T)
                return pts[i].forward + alpha * (pts[i + 1].forward - pts[i].forward)
        return self.spot

    def forward_at(self, target: datetime) -> float:
        """Query forward price at an absolute timestamp."""
        T = time_to_expiry_years(target, self.as_of)
        return self.forward(max(T, 0.0))

    def basis(self, T: float) -> float:
        """Percentage basis at time T: (F(T) - S) / S."""
        return (self.forward(T) - self.spot) / self.spot

    def annualized_carry(self, T: float) -> float:
        """Annualized carry rate implied by the basis."""
        if T <= 0:
            return 0.0
        b = self.basis(T)
        return b / T

    def implied_rate(self, T: float) -> float:
        """Continuously compounded implied rate: r = ln(F/S) / T."""
        if T <= 0:
            return 0.0
        F = self.forward(T)
        if F <= 0 or self.spot <= 0:
            return 0.0
        return np.log(F / self.spot) / T

    def curve_table(self, tenors: list[float] | None = None) -> pd.DataFrame:
        """Return the forward curve as a DataFrame."""
        if tenors is None:
            tenors = [0] + sorted([p.T for p in self.points])
        rows = []
        for T in tenors:
            F = self.forward(T)
            rows.append({
                "T_years": round(T, 4),
                "days": round(T * 365.25),
                "forward": round(F, 2),
                "basis_pct": round(self.basis(T) * 100, 3),
                "annualized_carry_pct": round(self.annualized_carry(T) * 100, 2),
                "implied_rate_pct": round(self.implied_rate(T) * 100, 2),
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Forward Curve: spot={self.spot:.2f}, {len(self.points)} points"]
        for p in sorted(self.points, key=lambda x: x.T):
            lines.append(
                f"  T={p.T:.4f}y ({p.T*365.25:.0f}d) | F={p.forward:.2f} | "
                f"basis={p.basis_pct:+.3f}% | carry={p.annualized_basis:+.2f}%/yr | "
                f"src={p.source}"
            )
        return "\n".join(lines)


def build_forward_curve_from_futures(
    df_futures: pd.DataFrame,
    spot: float,
    as_of: datetime | None = None,
) -> ForwardCurve:
    """
    Build forward curve from Deribit futures data.

    Parameters
    ----------
    df_futures : pd.DataFrame
        Raw futures book summary from Deribit.
    spot : float
        Current spot/index price.
    as_of : datetime
        Reference time.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)

    points = []
    for _, row in df_futures.iterrows():
        name = row.get("instrument_name", "")
        parsed = parse_deribit_instrument(name)

        # Skip perpetuals
        if parsed.get("instrument_type") == "perpetual":
            continue
        if "expiry_dt" not in parsed:
            continue

        expiry = parsed["expiry_dt"]
        T = time_to_expiry_years(expiry, as_of)
        if T <= 0:
            continue

        # Mark price is the forward
        mark = row.get("mark_price")
        if mark is None or float(mark) <= 0:
            # Try mid
            bid = row.get("bid_price", 0)
            ask = row.get("ask_price", 0)
            if bid and ask and float(bid) > 0 and float(ask) > 0:
                mark = (float(bid) + float(ask)) / 2
            else:
                continue
        else:
            mark = float(mark)

        basis_pct = (mark - spot) / spot
        ann_basis = basis_pct / T if T > 0 else 0.0

        points.append(ForwardPoint(
            expiry=expiry,
            T=T,
            forward=mark,
            source="futures",
            basis_pct=basis_pct * 100,
            annualized_basis=ann_basis * 100,
        ))

    curve = ForwardCurve(spot=spot, as_of=as_of, points=points)
    curve.fit()
    logger.info(f"Forward curve built with {len(points)} futures points")
    return curve


def build_forward_curve_from_options(
    df_options: pd.DataFrame,
    spot: float,
    as_of: datetime | None = None,
) -> ForwardCurve:
    """
    Build forward curve from put-call parity on options data.

    For each expiry, finds ATM call/put pairs and infers:
        F = K + exp(rT) * (C - P)
    where C, P are the call/put mid prices at strike K.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)

    points = []
    for expiry in df_options["expiry_dt"].unique():
        slice_df = df_options[df_options["expiry_dt"] == expiry]
        T = slice_df["T"].iloc[0] if "T" in slice_df.columns else time_to_expiry_years(expiry, as_of)
        if T <= 0:
            continue

        # Find strikes with both call and put
        calls = slice_df[slice_df["option_type"] == "C"].set_index("strike")
        puts = slice_df[slice_df["option_type"] == "P"].set_index("strike")
        common_strikes = calls.index.intersection(puts.index)

        if len(common_strikes) == 0:
            continue

        # Pick the strike nearest to spot
        best_K = min(common_strikes, key=lambda k: abs(k - spot))
        C = calls.loc[best_K, "mid_usd"]
        P = puts.loc[best_K, "mid_usd"]

        if pd.isna(C) or pd.isna(P) or C <= 0 or P <= 0:
            continue

        # Put-call parity: F = K + (C - P) * exp(rT), assume r=0
        F = best_K + (C - P)

        # Sanity check
        if F <= 0 or abs(F / spot - 1) > 0.5:
            logger.debug(f"Skipping put-call parity forward {F:.0f} at T={T:.4f}")
            continue

        basis_pct = (F - spot) / spot
        ann_basis = basis_pct / T if T > 0 else 0.0

        points.append(ForwardPoint(
            expiry=expiry,
            T=T,
            forward=F,
            source="put_call_parity",
            basis_pct=basis_pct * 100,
            annualized_basis=ann_basis * 100,
        ))

    curve = ForwardCurve(spot=spot, as_of=as_of, points=points)
    curve.fit()
    logger.info(f"Forward curve from options with {len(points)} put-call parity points")
    return curve


def merge_forward_curves(
    futures_curve: ForwardCurve,
    options_curve: ForwardCurve,
    prefer: str = "futures",
) -> ForwardCurve:
    """
    Merge forward curves from futures and options.

    Futures are preferred where available; options fill gaps.
    """
    spot = futures_curve.spot
    as_of = futures_curve.as_of

    # Index existing futures tenors
    fut_tenors = {round(p.T, 4) for p in futures_curve.points}

    merged_points = list(futures_curve.points)

    for p in options_curve.points:
        # Add options-derived points that don't overlap futures
        T_rounded = round(p.T, 4)
        if T_rounded not in fut_tenors:
            merged_points.append(p)

    curve = ForwardCurve(spot=spot, as_of=as_of, points=merged_points)
    curve.fit()
    logger.info(f"Merged forward curve: {len(merged_points)} points "
                f"({len(futures_curve.points)} futures + "
                f"{len(merged_points) - len(futures_curve.points)} options)")
    return curve
