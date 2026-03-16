"""Tests for risk-neutral density extraction."""

import numpy as np
import pytest

from src.surface.iv_surface import IVSurface, SmileSlice
from src.distribution.risk_neutral_density import extract_density_breeden_litzenberger


def _make_flat_vol_surface(forward: float = 100000, sigma: float = 0.6, T: float = 0.1) -> IVSurface:
    """
    Create a synthetic IV surface with flat volatility.

    Under flat vol, the density should be lognormal, which we can verify analytically.
    """
    surface = IVSurface()

    # Create strikes around forward
    n_strikes = 50
    strikes = np.linspace(forward * 0.5, forward * 1.5, n_strikes)
    log_m = np.log(strikes / forward)
    ivs = np.full(n_strikes, sigma)

    from datetime import datetime, timezone, timedelta
    expiry = datetime.now(timezone.utc) + timedelta(days=T * 365.25)

    smile = SmileSlice(
        expiry=expiry,
        T=T,
        forward=forward,
        strikes=strikes,
        ivs=ivs,
        log_moneyness=log_m,
    )
    smile.fit_spline()
    surface.slices[T] = smile
    surface.forward_curve[T] = forward
    surface._sorted_T = np.array([T])

    return surface


class TestDensityExtraction:
    """Test Breeden-Litzenberger density extraction."""

    def test_density_integrates_to_one(self):
        """Extracted density should integrate to approximately 1."""
        surface = _make_flat_vol_surface()
        T = list(surface.slices.keys())[0]
        density = extract_density_breeden_litzenberger(surface, T)

        from scipy.integrate import trapezoid
        total = trapezoid(density.pdf, density.strikes)
        assert abs(total - 1.0) < 0.05, f"Density integrates to {total}"

    def test_density_no_negative(self):
        """Density should not have negative values (after repair)."""
        surface = _make_flat_vol_surface()
        T = list(surface.slices.keys())[0]
        density = extract_density_breeden_litzenberger(surface, T)
        assert np.all(density.pdf >= 0)

    def test_mean_near_forward(self):
        """Density mean should be near the forward price."""
        forward = 100000
        surface = _make_flat_vol_surface(forward=forward)
        T = list(surface.slices.keys())[0]
        density = extract_density_breeden_litzenberger(surface, T)

        # Mean should be within 5% of forward for flat vol
        assert abs(density.mean / forward - 1.0) < 0.10, \
            f"Mean={density.mean:.0f}, Forward={forward:.0f}"

    def test_cdf_monotone(self):
        """CDF should be monotonically non-decreasing."""
        surface = _make_flat_vol_surface()
        T = list(surface.slices.keys())[0]
        density = extract_density_breeden_litzenberger(surface, T)

        diffs = np.diff(density.cdf)
        assert np.all(diffs >= -1e-10), "CDF is not monotone"

    def test_cdf_endpoints(self):
        """CDF should go from ~0 to ~1."""
        surface = _make_flat_vol_surface()
        T = list(surface.slices.keys())[0]
        density = extract_density_breeden_litzenberger(surface, T)

        assert density.cdf[0] < 0.05
        assert density.cdf[-1] > 0.95

    def test_atm_prob_above_near_half(self):
        """P(S > F) should be near 0.5 for flat vol (slightly below due to lognormal skew)."""
        forward = 100000
        surface = _make_flat_vol_surface(forward=forward)
        T = list(surface.slices.keys())[0]
        density = extract_density_breeden_litzenberger(surface, T)

        prob = density.prob_above(forward)
        # For lognormal, P(S > F) is N(d2) which is slightly below 0.5
        assert 0.3 < prob < 0.7, f"P(above forward) = {prob}"

    def test_different_vol_levels(self):
        """Higher vol should produce wider distribution."""
        forward = 100000
        s_lo = _make_flat_vol_surface(forward=forward, sigma=0.3)
        s_hi = _make_flat_vol_surface(forward=forward, sigma=0.9)

        T = list(s_lo.slices.keys())[0]
        d_lo = extract_density_breeden_litzenberger(s_lo, T)
        d_hi = extract_density_breeden_litzenberger(s_hi, T)

        assert d_hi.std > d_lo.std, "Higher vol should give wider distribution"
