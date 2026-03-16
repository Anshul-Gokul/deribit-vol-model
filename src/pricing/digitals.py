"""
Digital option pricing using vertical spread approximation.

A digital call (paying 1 if S_T > K) can be approximated by a tight call spread:
    Digital_Call ≈ (C(K - ε) - C(K + ε)) / (2ε)

This provides an independent estimate of P(S_T > K) that can be compared
against the Breeden-Litzenberger density-based probability.

The spread width ε controls the tradeoff:
- Smaller ε: more accurate but more sensitive to IV noise
- Larger ε: smoother but less precise

For liquid strikes, the two methods should agree closely.
"""

from __future__ import annotations

import logging

import numpy as np

from src.surface.iv_surface import IVSurface
from src.utils.black_scholes import bs_call, bs_put

logger = logging.getLogger(__name__)


def digital_call_spread(
    surface: IVSurface,
    K: float,
    T: float,
    forward: float | None = None,
    r: float = 0.0,
    spread_pct: float = 0.005,
) -> float:
    """
    Approximate digital call P(S_T > K) using a call spread.

    Digital ≈ (C(K - ε) - C(K + ε)) / (2ε)

    Parameters
    ----------
    surface : IVSurface
        Fitted IV surface.
    K : float
        Strike.
    T : float
        Time to expiry in years.
    forward : float, optional
        Forward price.
    r : float
        Risk-free rate.
    spread_pct : float
        Spread width as fraction of K.

    Returns
    -------
    float
        Estimated probability P(S_T > K).
    """
    if forward is None:
        forward = surface._interpolate_forward(T)

    eps = K * spread_pct
    C_lo = surface.call_price(K - eps, T, forward, r)
    C_hi = surface.call_price(K + eps, T, forward, r)

    digital = (C_lo - C_hi) / (2 * eps)

    # Undiscount to get probability
    digital *= np.exp(r * T)

    return float(np.clip(digital, 0.0, 1.0))


def digital_put_spread(
    surface: IVSurface,
    K: float,
    T: float,
    forward: float | None = None,
    r: float = 0.0,
    spread_pct: float = 0.005,
) -> float:
    """
    Approximate digital put P(S_T < K) using a put spread.

    Digital ≈ (P(K + ε) - P(K - ε)) / (2ε)
    """
    if forward is None:
        forward = surface._interpolate_forward(T)

    eps = K * spread_pct
    P_hi = surface.put_price(K + eps, T, forward, r)
    P_lo = surface.put_price(K - eps, T, forward, r)

    digital = (P_hi - P_lo) / (2 * eps)
    digital *= np.exp(r * T)

    return float(np.clip(digital, 0.0, 1.0))


def digital_profile(
    surface: IVSurface,
    T: float,
    strikes: np.ndarray | None = None,
    forward: float | None = None,
    r: float = 0.0,
    spread_pct: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute digital call probabilities across a range of strikes.

    Returns (strikes, probabilities).
    """
    if forward is None:
        forward = surface._interpolate_forward(T)

    if strikes is None:
        strikes = np.linspace(forward * 0.5, forward * 1.5, 100)

    probs = np.array([
        digital_call_spread(surface, K, T, forward, r, spread_pct)
        for K in strikes
    ])

    return strikes, probs


def compare_digital_vs_density(
    surface: IVSurface,
    density,  # RiskNeutralDensity
    strikes: np.ndarray,
    T: float,
    forward: float | None = None,
    r: float = 0.0,
    spread_pct: float = 0.005,
) -> dict[str, np.ndarray]:
    """
    Compare digital spread approximation against density-based probabilities.

    Returns dict with strikes, spread probs, density probs, and differences.
    """
    spread_probs = np.array([
        digital_call_spread(surface, K, T, forward, r, spread_pct)
        for K in strikes
    ])

    density_probs = np.array([density.prob_above(K) for K in strikes])

    diff = spread_probs - density_probs

    return {
        "strikes": strikes,
        "spread_probs": spread_probs,
        "density_probs": density_probs,
        "difference": diff,
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
    }
