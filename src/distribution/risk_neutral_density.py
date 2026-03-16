"""
Risk-neutral density extraction from options prices.

Primary method: Breeden-Litzenberger (1978)
The risk-neutral probability density f(K) at expiry T satisfies:
    f(K) = e^(rT) * d²C/dK²
where C(K) is the undiscounted call price as a function of strike.

In practice we:
1. Evaluate call prices on a fine strike grid using the fitted IV surface
2. Take the second derivative numerically using central finite differences
3. Smooth and normalize the resulting density

This gives us the market-implied terminal distribution of the underlying price,
under the risk-neutral measure Q.

IMPORTANT: This is NOT the real-world probability distribution. It incorporates
risk premia. For crypto (positive risk premium), the Q-measure typically assigns
more weight to the downside than the physical P-measure would.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import trapezoid
from scipy.ndimage import uniform_filter1d

from src.surface.iv_surface import IVSurface
from src.utils.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class RiskNeutralDensity:
    """
    Risk-neutral density for a specific time to expiry.

    Attributes
    ----------
    T : float
        Time to expiry in years.
    forward : float
        Forward price used.
    strikes : np.ndarray
        Strike grid.
    pdf : np.ndarray
        Probability density function values.
    cdf : np.ndarray
        Cumulative distribution function values.
    call_prices : np.ndarray
        Call prices on the strike grid (used for extraction).
    r : float
        Risk-free rate used.
    method : str
        Extraction method used.
    """

    T: float
    forward: float
    strikes: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray
    call_prices: np.ndarray = field(default_factory=lambda: np.array([]))
    r: float = 0.0
    method: str = "breeden_litzenberger"

    @property
    def mean(self) -> float:
        """Expected value under the risk-neutral measure."""
        return float(trapezoid(self.strikes * self.pdf, self.strikes))

    @property
    def variance(self) -> float:
        """Variance under the risk-neutral measure."""
        mu = self.mean
        return float(trapezoid((self.strikes - mu) ** 2 * self.pdf, self.strikes))

    @property
    def std(self) -> float:
        """Standard deviation under the risk-neutral measure."""
        return float(np.sqrt(max(self.variance, 0)))

    def prob_above(self, K: float) -> float:
        """P(S_T > K) under risk-neutral measure."""
        if K <= self.strikes[0]:
            return 1.0
        if K >= self.strikes[-1]:
            return 0.0
        idx = np.searchsorted(self.strikes, K)
        return float(1.0 - np.interp(K, self.strikes, self.cdf))

    def prob_below(self, K: float) -> float:
        """P(S_T < K) under risk-neutral measure."""
        return 1.0 - self.prob_above(K)

    def prob_between(self, K_low: float, K_high: float) -> float:
        """P(K_low <= S_T <= K_high) under risk-neutral measure."""
        return self.prob_above(K_low) - self.prob_above(K_high)

    def prob_outside(self, K_low: float, K_high: float) -> float:
        """P(S_T < K_low or S_T > K_high)."""
        return 1.0 - self.prob_between(K_low, K_high)

    def quantile(self, q: float) -> float:
        """Quantile function (inverse CDF)."""
        return float(np.interp(q, self.cdf, self.strikes))

    def bucket_probabilities(
        self, boundaries: np.ndarray
    ) -> list[tuple[float, float, float]]:
        """
        Compute probabilities for adjacent buckets defined by boundaries.

        Returns list of (lower, upper, probability) tuples.
        """
        buckets = []
        for i in range(len(boundaries) - 1):
            lo, hi = boundaries[i], boundaries[i + 1]
            p = self.prob_between(lo, hi)
            buckets.append((lo, hi, p))
        return buckets

    def validate(self) -> dict[str, float]:
        """Run validation checks on the density."""
        total_mass = float(trapezoid(self.pdf, self.strikes))
        mean = self.mean
        mean_vs_forward = abs(mean / self.forward - 1.0)
        max_negative = float(np.min(self.pdf))
        pct_negative = float(np.mean(self.pdf < 0))

        checks = {
            "total_mass": total_mass,
            "mean": mean,
            "forward": self.forward,
            "mean_vs_forward_pct": mean_vs_forward,
            "max_negative_density": max_negative,
            "pct_negative_density": pct_negative,
            "std": self.std,
        }

        if abs(total_mass - 1.0) > 0.05:
            logger.warning(f"Density total mass = {total_mass:.4f} (expected ~1.0)")
        if mean_vs_forward > 0.05:
            logger.warning(
                f"Density mean ({mean:.0f}) deviates from forward ({self.forward:.0f}) "
                f"by {mean_vs_forward:.1%}"
            )
        if max_negative < -1e-6:
            logger.warning(f"Negative density detected: min = {max_negative:.6f}")

        return checks


def extract_density_breeden_litzenberger(
    surface: IVSurface,
    T: float,
    forward: float | None = None,
    r: float = 0.0,
    config: dict | None = None,
) -> RiskNeutralDensity:
    """
    Extract risk-neutral density using Breeden-Litzenberger method.

    The density is the second derivative of the call price with respect to strike:
        f(K) = e^(rT) * d²C/dK²

    We compute this numerically on a fine strike grid using the fitted IV surface.

    Parameters
    ----------
    surface : IVSurface
        Fitted IV surface.
    T : float
        Target time to expiry in years.
    forward : float, optional
        Forward price. If None, interpolated from surface.
    r : float
        Risk-free rate.
    config : dict, optional
        Distribution config.
    """
    cfg = (config or get_config()).get("distribution", {})
    n_points = cfg.get("num_strike_points", 500)
    fd_step_pct = cfg.get("fd_step_pct", 0.002)
    smooth = cfg.get("smooth_density", True)
    smooth_window = cfg.get("smoothing_window", 5)

    if forward is None:
        forward = surface._interpolate_forward(T)

    # Build strike grid: centered on forward, spanning reasonable range
    # Use log-space for better coverage of tails
    log_spread = 1.0  # covers roughly ±2.7x the forward price
    log_K_min = np.log(forward) - log_spread
    log_K_max = np.log(forward) + log_spread
    log_strikes = np.linspace(log_K_min, log_K_max, n_points)
    strikes = np.exp(log_strikes)

    # Evaluate call prices on the grid
    call_prices = surface.call_prices_on_grid(strikes, T, forward, r)

    # Finite difference step
    dK = fd_step_pct * forward

    # Compute second derivative using central differences
    # f(K) = e^(rT) * (C(K+dK) - 2*C(K) + C(K-dK)) / dK^2
    pdf = np.zeros_like(strikes)
    discount = np.exp(r * T)

    for i in range(len(strikes)):
        K = strikes[i]
        C_up = surface.call_price(K + dK, T, forward, r)
        C_mid = call_prices[i]
        C_down = surface.call_price(K - dK, T, forward, r)
        pdf[i] = discount * (C_up - 2.0 * C_mid + C_down) / (dK**2)

    # Smooth the density
    if smooth and smooth_window > 1:
        pdf = uniform_filter1d(pdf, size=smooth_window)

    # Clip negative values
    n_negative = np.sum(pdf < 0)
    if n_negative > 0:
        logger.info(f"Clipping {n_negative}/{len(pdf)} negative density values")
        pdf = np.maximum(pdf, 0.0)

    # Normalize to integrate to 1
    total = trapezoid(pdf, strikes)
    if total > 0:
        pdf = pdf / total
    else:
        logger.error("Density integrates to zero — check surface fit")
        # Fallback: lognormal centered at forward
        sigma = 0.5  # crude default
        pdf = _lognormal_pdf(strikes, forward, sigma, T)

    # Compute CDF by cumulative integration
    cdf = np.zeros_like(strikes)
    for i in range(1, len(strikes)):
        cdf[i] = cdf[i - 1] + 0.5 * (pdf[i] + pdf[i - 1]) * (strikes[i] - strikes[i - 1])
    # Normalize CDF endpoint
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]

    density = RiskNeutralDensity(
        T=T,
        forward=forward,
        strikes=strikes,
        pdf=pdf,
        cdf=cdf,
        call_prices=call_prices,
        r=r,
        method="breeden_litzenberger",
    )

    # Validate
    density.validate()

    return density


def _lognormal_pdf(K: np.ndarray, F: float, sigma: float, T: float) -> np.ndarray:
    """Lognormal PDF as fallback density."""
    from scipy.stats import lognorm

    # Lognormal with mean = F
    s = sigma * np.sqrt(T)
    scale = F * np.exp(-0.5 * s**2)
    return lognorm.pdf(K, s=s, scale=scale)


def extract_density(
    surface: IVSurface,
    T: float,
    forward: float | None = None,
    r: float = 0.0,
    method: str = "breeden_litzenberger",
    config: dict | None = None,
) -> RiskNeutralDensity:
    """
    Extract risk-neutral density using the specified method.

    Parameters
    ----------
    surface : IVSurface
        Fitted IV surface.
    T : float
        Target time to expiry.
    forward : float, optional
        Forward price.
    r : float
        Risk-free rate.
    method : str
        "breeden_litzenberger" (default).
    config : dict, optional
        Configuration overrides.
    """
    if method == "breeden_litzenberger":
        return extract_density_breeden_litzenberger(surface, T, forward, r, config)
    else:
        raise ValueError(f"Unknown density extraction method: {method}")
