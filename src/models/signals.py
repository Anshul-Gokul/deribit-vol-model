"""
Derivatives-implied directional and volatility signals.

Extracts market microstructure signals from the options/futures data:

1. **Risk Reversal (skew)** — difference in IV between OTM calls and puts
   at matched delta.  Positive = market bids for upside.
2. **Butterfly** — curvature of the smile. High = fat tails expected.
3. **Term Structure Slope** — ATM IV slope across tenors.
   Steep contango = calm now, fear later.  Backwardation = stress.
4. **Put-Call Volume/OI Ratio** — demand for protection vs upside.
5. **Futures Basis Momentum** — rate of change of the basis.
6. **Variance Risk Premium proxy** — implied vs recent realized vol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.surface.iv_surface import IVSurface
from src.models.forward_curve import ForwardCurve

logger = logging.getLogger(__name__)


@dataclass
class DerivativesSignals:
    """Collection of derivatives-implied signals at a point in time."""

    # Spot / forward
    spot: float
    forward_1d: float  # 1-day forward
    forward_7d: float  # 7-day forward
    forward_30d: float  # 30-day forward

    # Basis signals
    basis_1d_pct: float  # (F_1d - S) / S
    basis_7d_pct: float
    basis_30d_pct: float
    annualized_carry_7d: float  # annualized basis 7d

    # Volatility signals
    atm_iv_1d: float  # ATM IV for ~1 day
    atm_iv_7d: float  # ATM IV for ~7 days
    atm_iv_30d: float  # ATM IV for ~30 days
    iv_term_slope: float  # (IV_30d - IV_7d) — positive = contango

    # Skew signals (25-delta risk reversal)
    rr25_7d: float  # IV(25d call) - IV(25d put) for 7d
    rr25_30d: float  # same for 30d
    butterfly_25_7d: float  # 0.5*(IV(25d call) + IV(25d put)) - IV(ATM) for 7d

    # Flow signals
    put_call_oi_ratio: float  # total put OI / call OI
    put_call_volume_ratio: float

    # Composite
    directional_score: float  # -1 to +1, positive = bullish
    confidence: float  # 0 to 1

    def to_dict(self) -> dict:
        """Serialize all signals."""
        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}

    def summary(self) -> str:
        """Human-readable signal summary."""
        direction = "BULLISH" if self.directional_score > 0.1 else \
                    "BEARISH" if self.directional_score < -0.1 else "NEUTRAL"

        lines = [
            f"=== Derivatives Signals ({direction}) ===",
            f"  Spot:          {self.spot:>12,.2f}",
            f"  Forward 1d:    {self.forward_1d:>12,.2f}  ({self.basis_1d_pct:+.3f}%)",
            f"  Forward 7d:    {self.forward_7d:>12,.2f}  ({self.basis_7d_pct:+.3f}%)",
            f"  Forward 30d:   {self.forward_30d:>12,.2f}  ({self.basis_30d_pct:+.3f}%)",
            f"  Carry (7d ann): {self.annualized_carry_7d:+.2f}%",
            "",
            f"  ATM IV 1d:     {self.atm_iv_1d:>8.1%}",
            f"  ATM IV 7d:     {self.atm_iv_7d:>8.1%}",
            f"  ATM IV 30d:    {self.atm_iv_30d:>8.1%}",
            f"  IV Term Slope: {self.iv_term_slope:+.1%}",
            "",
            f"  RR25 7d:       {self.rr25_7d:+.1%}  ({'calls bid' if self.rr25_7d > 0 else 'puts bid'})",
            f"  RR25 30d:      {self.rr25_30d:+.1%}",
            f"  Butterfly 7d:  {self.butterfly_25_7d:.1%}",
            "",
            f"  P/C OI Ratio:  {self.put_call_oi_ratio:.2f}",
            f"  P/C Vol Ratio: {self.put_call_volume_ratio:.2f}",
            "",
            f"  Direction:     {self.directional_score:+.3f}  ({direction})",
            f"  Confidence:    {self.confidence:.2f}",
        ]
        return "\n".join(lines)


def _safe_iv(surface: IVSurface, K: float, T: float, forward: float) -> float:
    """Query IV with fallback."""
    try:
        iv = surface.iv(K, T, forward)
        return iv if 0.01 < iv < 5.0 else np.nan
    except Exception:
        return np.nan


def _atm_iv(surface: IVSurface, T: float) -> float:
    """ATM implied vol at tenor T."""
    fwd = surface._interpolate_forward(T)
    return _safe_iv(surface, fwd, T, fwd)


def _risk_reversal_25d(surface: IVSurface, T: float) -> float:
    """
    25-delta risk reversal: IV(25d call) - IV(25d put).

    Approximation: 25-delta corresponds roughly to K/F = exp(±0.674 * σ * √T)
    where 0.674 is the 75th percentile of N(0,1).
    We iterate once to refine.
    """
    fwd = surface._interpolate_forward(T)
    atm_iv = _safe_iv(surface, fwd, T, fwd)
    if np.isnan(atm_iv) or atm_iv <= 0:
        return np.nan

    # 25-delta strike approximation
    d = 0.674 * atm_iv * np.sqrt(T)
    K_call = fwd * np.exp(d)  # OTM call
    K_put = fwd * np.exp(-d)  # OTM put

    iv_call = _safe_iv(surface, K_call, T, fwd)
    iv_put = _safe_iv(surface, K_put, T, fwd)

    if np.isnan(iv_call) or np.isnan(iv_put):
        return np.nan
    return iv_call - iv_put


def _butterfly_25d(surface: IVSurface, T: float) -> float:
    """25-delta butterfly: 0.5*(IV_call + IV_put) - IV_ATM."""
    fwd = surface._interpolate_forward(T)
    atm_iv = _safe_iv(surface, fwd, T, fwd)
    if np.isnan(atm_iv):
        return np.nan

    d = 0.674 * atm_iv * np.sqrt(T)
    K_call = fwd * np.exp(d)
    K_put = fwd * np.exp(-d)

    iv_call = _safe_iv(surface, K_call, T, fwd)
    iv_put = _safe_iv(surface, K_put, T, fwd)

    if np.isnan(iv_call) or np.isnan(iv_put):
        return np.nan
    return 0.5 * (iv_call + iv_put) - atm_iv


def _put_call_ratios(df_options: pd.DataFrame) -> tuple[float, float]:
    """Compute put/call OI and volume ratios."""
    calls = df_options[df_options["option_type"] == "C"]
    puts = df_options[df_options["option_type"] == "P"]

    call_oi = calls["open_interest"].astype(float).sum() if "open_interest" in calls.columns else 0
    put_oi = puts["open_interest"].astype(float).sum() if "open_interest" in puts.columns else 0

    call_vol = calls["volume"].astype(float).sum() if "volume" in calls.columns else 0
    put_vol = puts["volume"].astype(float).sum() if "volume" in puts.columns else 0

    oi_ratio = put_oi / call_oi if call_oi > 0 else 1.0
    vol_ratio = put_vol / call_vol if call_vol > 0 else 1.0

    return oi_ratio, vol_ratio


def _compute_directional_score(
    basis_7d: float,
    rr25_7d: float,
    rr25_30d: float,
    iv_slope: float,
    pc_oi_ratio: float,
) -> tuple[float, float]:
    """
    Combine signals into a directional score in [-1, +1].

    Signal logic:
    - Positive basis → market willing to pay premium for future delivery → bullish
    - Positive risk reversal → call IV > put IV → upside demand → bullish
    - IV contango (positive slope) → calm near-term → slightly bullish
    - Low P/C OI ratio → less hedging demand → bullish

    Each signal is z-scored and combined with heuristic weights.
    Returns (score, confidence).
    """
    signals = []
    weights = []

    # Basis signal: positive basis is bullish
    if not np.isnan(basis_7d):
        # Typical crypto basis range: -5% to +20% annualized
        s = np.clip(basis_7d / 10.0, -1, 1)  # normalize
        signals.append(s)
        weights.append(0.20)

    # Risk reversal: positive = calls bid = bullish
    for rr, w in [(rr25_7d, 0.25), (rr25_30d, 0.15)]:
        if not np.isnan(rr):
            # Typical RR range: -0.10 to +0.10
            s = np.clip(rr / 0.05, -1, 1)
            signals.append(s)
            weights.append(w)

    # IV term structure: contango = calm = slightly bullish
    if not np.isnan(iv_slope):
        s = np.clip(iv_slope / 0.05, -1, 1)
        signals.append(s)
        weights.append(0.15)

    # Put-call ratio: < 1 is bullish, > 1 is bearish
    if not np.isnan(pc_oi_ratio) and pc_oi_ratio > 0:
        s = np.clip(-(pc_oi_ratio - 1.0) / 0.5, -1, 1)
        signals.append(s)
        weights.append(0.25)

    if not signals:
        return 0.0, 0.0

    weights = np.array(weights)
    weights /= weights.sum()
    score = float(np.dot(signals, weights))
    confidence = min(len(signals) / 5.0, 1.0)

    return np.clip(score, -1, 1), confidence


def extract_signals(
    surface: IVSurface,
    forward_curve: ForwardCurve,
    df_clean: pd.DataFrame,
) -> DerivativesSignals:
    """
    Extract all derivatives-implied signals from fitted surface and forward curve.

    Parameters
    ----------
    surface : IVSurface
        Fitted IV surface.
    forward_curve : ForwardCurve
        Fitted forward curve from futures/options.
    df_clean : pd.DataFrame
        Cleaned options data (for OI/volume stats).
    """
    spot = forward_curve.spot
    T_1d = 1 / 365.25
    T_7d = 7 / 365.25
    T_30d = 30 / 365.25

    # Forwards
    fwd_1d = forward_curve.forward(T_1d)
    fwd_7d = forward_curve.forward(T_7d)
    fwd_30d = forward_curve.forward(T_30d)

    basis_1d = (fwd_1d - spot) / spot * 100
    basis_7d = (fwd_7d - spot) / spot * 100
    basis_30d = (fwd_30d - spot) / spot * 100
    carry_7d = forward_curve.annualized_carry(T_7d) * 100

    # ATM IVs
    atm_1d = _atm_iv(surface, T_1d)
    atm_7d = _atm_iv(surface, T_7d)
    atm_30d = _atm_iv(surface, T_30d)
    iv_slope = atm_30d - atm_7d if not (np.isnan(atm_30d) or np.isnan(atm_7d)) else np.nan

    # Skew
    rr_7d = _risk_reversal_25d(surface, T_7d)
    rr_30d = _risk_reversal_25d(surface, T_30d)
    bf_7d = _butterfly_25d(surface, T_7d)

    # P/C ratios
    oi_ratio, vol_ratio = _put_call_ratios(df_clean)

    # Composite
    score, conf = _compute_directional_score(
        basis_7d, rr_7d, rr_30d,
        iv_slope if not np.isnan(iv_slope) else 0.0,
        oi_ratio,
    )

    return DerivativesSignals(
        spot=spot,
        forward_1d=fwd_1d,
        forward_7d=fwd_7d,
        forward_30d=fwd_30d,
        basis_1d_pct=basis_1d,
        basis_7d_pct=basis_7d,
        basis_30d_pct=basis_30d,
        annualized_carry_7d=carry_7d,
        atm_iv_1d=atm_1d,
        atm_iv_7d=atm_7d,
        atm_iv_30d=atm_30d,
        iv_term_slope=iv_slope if not np.isnan(iv_slope) else 0.0,
        rr25_7d=rr_7d if not np.isnan(rr_7d) else 0.0,
        rr25_30d=rr_30d if not np.isnan(rr_30d) else 0.0,
        butterfly_25_7d=bf_7d if not np.isnan(bf_7d) else 0.0,
        put_call_oi_ratio=oi_ratio,
        put_call_volume_ratio=vol_ratio,
        directional_score=score,
        confidence=conf,
    )
