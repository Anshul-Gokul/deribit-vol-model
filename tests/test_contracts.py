"""Tests for contract pricing and probability queries."""

import numpy as np
import pytest

from src.distribution.risk_neutral_density import RiskNeutralDensity
from src.pricing.contracts import (
    price_above,
    price_below,
    price_between,
    price_outside,
    price_contract,
    price_bucket_ladder,
)


def _make_lognormal_density(forward: float = 100000, sigma: float = 0.5, T: float = 0.1) -> RiskNeutralDensity:
    """Create a synthetic lognormal density for testing."""
    from scipy.stats import lognorm

    # Lognormal parameters matching forward
    s = sigma * np.sqrt(T)
    scale = forward * np.exp(-0.5 * s**2)

    strikes = np.linspace(forward * 0.3, forward * 3.0, 1000)
    pdf = lognorm.pdf(strikes, s=s, scale=scale)

    # Normalize
    from scipy.integrate import trapezoid
    pdf = pdf / trapezoid(pdf, strikes)

    # CDF
    cdf = np.zeros_like(strikes)
    for i in range(1, len(strikes)):
        cdf[i] = cdf[i-1] + 0.5 * (pdf[i] + pdf[i-1]) * (strikes[i] - strikes[i-1])
    cdf = cdf / cdf[-1]

    return RiskNeutralDensity(
        T=T,
        forward=forward,
        strikes=strikes,
        pdf=pdf,
        cdf=cdf,
    )


class TestDensityProperties:
    """Test risk-neutral density object."""

    def test_probabilities_sum_to_one(self):
        d = _make_lognormal_density()
        p_above = d.prob_above(d.forward)
        p_below = d.prob_below(d.forward)
        assert abs(p_above + p_below - 1.0) < 0.01

    def test_between_plus_outside_equals_one(self):
        d = _make_lognormal_density()
        low, high = d.forward * 0.9, d.forward * 1.1
        p_between = d.prob_between(low, high)
        p_outside = d.prob_outside(low, high)
        assert abs(p_between + p_outside - 1.0) < 0.01

    def test_prob_above_decreasing(self):
        d = _make_lognormal_density()
        strikes = np.linspace(d.forward * 0.5, d.forward * 1.5, 50)
        probs = [d.prob_above(K) for K in strikes]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i+1] - 0.01  # monotonically decreasing

    def test_prob_above_extremes(self):
        d = _make_lognormal_density()
        assert d.prob_above(d.strikes[0]) > 0.99
        assert d.prob_above(d.strikes[-1]) < 0.01

    def test_mean_near_forward(self):
        d = _make_lognormal_density()
        assert abs(d.mean / d.forward - 1.0) < 0.05

    def test_quantile_median(self):
        d = _make_lognormal_density()
        median = d.quantile(0.5)
        # Median of lognormal is less than mean
        assert median < d.mean * 1.1
        assert median > d.forward * 0.5


class TestContractPricing:
    """Test prediction market contract pricing."""

    def test_above_contract(self):
        d = _make_lognormal_density()
        result = price_above(d, d.forward, asset="BTC")
        assert 0.0 < result.implied_probability < 1.0
        assert result.contract_type == "above"
        assert abs(result.fair_yes_price + result.fair_no_price - 1.0) < 1e-10

    def test_below_contract(self):
        d = _make_lognormal_density()
        result = price_below(d, d.forward, asset="BTC")
        assert 0.0 < result.implied_probability < 1.0
        assert result.contract_type == "below"

    def test_above_plus_below_equals_one(self):
        d = _make_lognormal_density()
        K = d.forward * 1.05
        above = price_above(d, K)
        below = price_below(d, K)
        assert abs(above.implied_probability + below.implied_probability - 1.0) < 0.01

    def test_between_contract(self):
        d = _make_lognormal_density()
        result = price_between(d, d.forward * 0.95, d.forward * 1.05)
        assert 0.0 < result.implied_probability < 1.0
        assert result.contract_type == "between"

    def test_outside_contract(self):
        d = _make_lognormal_density()
        result = price_outside(d, d.forward * 0.95, d.forward * 1.05)
        assert 0.0 < result.implied_probability < 1.0

    def test_odds_computation(self):
        d = _make_lognormal_density()
        result = price_above(d, d.forward)
        assert result.implied_decimal_odds_yes > 1.0
        assert result.implied_decimal_odds_no > 1.0

    def test_price_contract_dispatcher(self):
        d = _make_lognormal_density()
        r1 = price_contract(d, "above", strike=d.forward)
        assert r1.contract_type == "above"

        r2 = price_contract(d, "between", lower=d.forward*0.9, upper=d.forward*1.1)
        assert r2.contract_type == "between"

    def test_bucket_ladder(self):
        d = _make_lognormal_density()
        boundaries = [d.forward * f for f in [0.8, 0.9, 1.0, 1.1, 1.2]]
        buckets = price_bucket_ladder(d, boundaries)
        assert len(buckets) == 4
        total_prob = sum(b.implied_probability for b in buckets)
        # Won't sum to 1 since we don't cover entire range
        assert 0.5 < total_prob < 1.0
