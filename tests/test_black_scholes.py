"""Tests for Black-Scholes pricing and implied volatility."""

import numpy as np
import pytest

from src.utils.black_scholes import (
    bs_call,
    bs_put,
    bs_call_vec,
    bs_put_vec,
    implied_vol,
    bs_digital_call,
    bs_digital_put,
)


class TestBSPricing:
    """Test Black-Scholes call and put pricing."""

    def test_call_atm(self):
        """ATM call should be roughly 0.4 * S * sigma * sqrt(T) for small vol."""
        S, K, T, sigma = 100.0, 100.0, 1.0, 0.20
        price = bs_call(S, K, T, sigma)
        # Known approximate value for ATM
        assert 5.0 < price < 12.0

    def test_put_atm(self):
        """ATM put should be similar to call by put-call parity (r=0)."""
        S, K, T, sigma = 100.0, 100.0, 1.0, 0.20
        call = bs_call(S, K, T, sigma)
        put = bs_put(S, K, T, sigma)
        # With r=0: C - P = S - K = 0
        assert abs(call - put) < 1e-10

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT)."""
        S, K, T, sigma, r = 100.0, 105.0, 0.5, 0.30, 0.05
        call = bs_call(S, K, T, sigma, r)
        put = bs_put(S, K, T, sigma, r)
        parity = call - put - (S - K * np.exp(-r * T))
        assert abs(parity) < 1e-10

    def test_expired_call(self):
        """Expired call = max(S-K, 0)."""
        assert bs_call(110, 100, 0, 0.2) == 10.0
        assert bs_call(90, 100, 0, 0.2) == 0.0

    def test_expired_put(self):
        """Expired put = max(K-S, 0)."""
        assert bs_put(90, 100, 0, 0.2) == 10.0
        assert bs_put(110, 100, 0, 0.2) == 0.0

    def test_deep_itm_call(self):
        """Deep ITM call should be close to intrinsic."""
        price = bs_call(200, 100, 0.01, 0.2)
        assert abs(price - 100) < 1.0

    def test_deep_otm_call(self):
        """Deep OTM call should be near zero."""
        price = bs_call(50, 100, 0.1, 0.2)
        assert price < 0.01

    def test_vectorized_matches_scalar(self):
        """Vectorized pricing should match scalar."""
        S = np.array([100.0, 110.0, 90.0])
        K = np.array([100.0, 100.0, 100.0])
        T = np.array([1.0, 1.0, 1.0])
        sigma = np.array([0.2, 0.2, 0.2])

        vec_prices = bs_call_vec(S, K, T, sigma)
        for i in range(3):
            scalar_price = bs_call(S[i], K[i], T[i], sigma[i])
            assert abs(vec_prices[i] - scalar_price) < 1e-10


class TestImpliedVol:
    """Test implied volatility computation."""

    def test_roundtrip_call(self):
        """IV of a BS-priced call should recover the original vol."""
        S, K, T, sigma = 100.0, 105.0, 0.5, 0.25
        price = bs_call(S, K, T, sigma)
        iv = implied_vol(price, S, K, T, option_type="call")
        assert iv is not None
        assert abs(iv - sigma) < 1e-6

    def test_roundtrip_put(self):
        """IV of a BS-priced put should recover the original vol."""
        S, K, T, sigma = 100.0, 95.0, 0.5, 0.30
        price = bs_put(S, K, T, sigma)
        iv = implied_vol(price, S, K, T, option_type="put")
        assert iv is not None
        assert abs(iv - sigma) < 1e-6

    def test_zero_price_returns_none(self):
        """Zero price should return None."""
        assert implied_vol(0, 100, 100, 1.0) is None

    def test_negative_price_returns_none(self):
        """Negative price should return None."""
        assert implied_vol(-1, 100, 100, 1.0) is None

    def test_expired_returns_none(self):
        """Expired option should return None."""
        assert implied_vol(5, 100, 100, 0) is None


class TestDigitals:
    """Test digital option pricing."""

    def test_atm_digital_call(self):
        """ATM digital call should be roughly 0.5 (slightly less due to drift)."""
        prob = bs_digital_call(100, 100, 1.0, 0.20, r=0)
        assert 0.40 < prob < 0.60

    def test_digital_call_put_sum_to_pv1(self):
        """Digital call + digital put = exp(-rT)."""
        S, K, T, sigma, r = 100.0, 105.0, 0.5, 0.25, 0.05
        dc = bs_digital_call(S, K, T, sigma, r)
        dp = bs_digital_put(S, K, T, sigma, r)
        assert abs(dc + dp - np.exp(-r * T)) < 1e-10

    def test_deep_itm_digital(self):
        """Deep ITM digital should be close to exp(-rT)."""
        dc = bs_digital_call(200, 100, 0.5, 0.2, r=0)
        assert dc > 0.99

    def test_deep_otm_digital(self):
        """Deep OTM digital should be close to 0."""
        dc = bs_digital_call(50, 100, 0.5, 0.2, r=0)
        assert dc < 0.01
