"""
Integration test: end-to-end pipeline with synthetic data.

This test verifies the full flow from cleaned data through surface fitting,
density extraction, and contract pricing — without requiring live API access.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from src.cleaning.options_cleaner import clean_options_data
from src.surface.iv_surface import IVSurface
from src.distribution.risk_neutral_density import extract_density
from src.pricing.contracts import price_above, price_below, price_between, price_outside
from src.pricing.digitals import digital_call_spread


def _generate_synthetic_options(
    forward: float = 85000.0,
    base_iv: float = 0.55,
    n_expiries: int = 3,
    n_strikes_per: int = 20,
) -> pd.DataFrame:
    """
    Generate synthetic options data that mimics Deribit book summary format.

    Creates options across multiple expiries with a realistic-ish smile.
    """
    now = datetime.now(timezone.utc)
    rows = []

    for i in range(n_expiries):
        dte = [7, 30, 90][i] if i < 3 else 30 * (i + 1)
        expiry = now + timedelta(days=dte)
        day = expiry.day
        month = expiry.strftime("%b").upper()
        year = expiry.strftime("%y")
        expiry_str = f"{day}{month}{year}"

        T = dte / 365.25

        for j in range(n_strikes_per):
            # Strikes from 60% to 140% of forward
            strike_pct = 0.6 + 0.8 * j / (n_strikes_per - 1)
            strike = round(forward * strike_pct / 1000) * 1000  # Round to nearest 1000
            log_m = np.log(strike / forward)

            # Smile: quadratic in log-moneyness + slight skew
            iv = base_iv + 0.3 * log_m**2 - 0.1 * log_m + 0.02 * np.sqrt(T)
            iv = max(iv, 0.1)

            # Option type: OTM convention
            opt_type = "C" if strike >= forward else "P"
            instrument = f"BTC-{expiry_str}-{int(strike)}-{opt_type}"

            # BS price in BTC terms
            from src.utils.black_scholes import bs_call, bs_put
            if opt_type == "C":
                price_usd = bs_call(forward, strike, T, iv)
            else:
                price_usd = bs_put(forward, strike, T, iv)

            price_btc = price_usd / forward

            # Add some bid-ask spread
            spread = price_btc * 0.05
            bid = max(price_btc - spread / 2, 0.0001)
            ask = price_btc + spread / 2

            rows.append({
                "instrument_name": instrument,
                "bid_price": bid,
                "ask_price": ask,
                "mark_price": price_btc,
                "mark_iv": iv * 100,  # Deribit reports as percentage
                "underlying_price": forward,
                "open_interest": 100,
                "volume": 10,
            })

    return pd.DataFrame(rows)


class TestEndToEnd:
    """End-to-end integration test."""

    def test_full_pipeline(self):
        """Test the complete pipeline from synthetic data to contract prices."""
        forward = 85000.0

        # 1. Generate synthetic data
        df_raw = _generate_synthetic_options(forward=forward)
        assert len(df_raw) > 0

        # 2. Clean data
        df_clean = clean_options_data(df_raw)
        assert len(df_clean) > 0
        assert "T" in df_clean.columns
        assert "log_moneyness" in df_clean.columns

        # 3. Build surface
        surface = IVSurface()
        surface.fit(df_clean)
        assert len(surface.slices) > 0

        # 4. Query IV at various points
        T = surface.expiry_times[0]
        iv_atm = surface.iv(forward, T)
        assert 0.1 < iv_atm < 2.0

        # 5. Extract density
        density = extract_density(surface, T, forward)
        assert density.pdf is not None
        assert len(density.pdf) > 0
        assert np.all(density.pdf >= 0)

        # 6. Price contracts
        above = price_above(density, forward, asset="BTC")
        assert 0 < above.implied_probability < 1
        assert abs(above.fair_yes_price + above.fair_no_price - 1.0) < 1e-10

        below = price_below(density, forward, asset="BTC")
        assert abs(above.implied_probability + below.implied_probability - 1.0) < 0.02

        between = price_between(density, forward * 0.9, forward * 1.1, asset="BTC")
        assert 0 < between.implied_probability < 1

        outside = price_outside(density, forward * 0.9, forward * 1.1, asset="BTC")
        assert abs(between.implied_probability + outside.implied_probability - 1.0) < 0.02

        # 7. Digital approximation
        dig_prob = digital_call_spread(surface, forward, T)
        assert 0 < dig_prob < 1
        # Should be reasonably close to density-based probability
        assert abs(dig_prob - above.implied_probability) < 0.1

    def test_multiple_expiries(self):
        """Test that surface handles multiple expiries correctly."""
        df_raw = _generate_synthetic_options(n_expiries=3)
        df_clean = clean_options_data(df_raw)

        surface = IVSurface()
        surface.fit(df_clean)

        assert len(surface.slices) >= 2  # At least 2 expiries should survive cleaning

        # IV should be queryable at intermediate times
        T_mid = np.mean(surface.expiry_times[:2])
        forward = surface._interpolate_forward(T_mid)
        iv = surface.iv(forward, T_mid)
        assert 0.1 < iv < 2.0

    def test_density_validation(self):
        """Test density validation checks."""
        df_raw = _generate_synthetic_options()
        df_clean = clean_options_data(df_raw)

        surface = IVSurface()
        surface.fit(df_clean)

        T = surface.expiry_times[0]
        forward = surface._interpolate_forward(T)
        density = extract_density(surface, T, forward)

        checks = density.validate()
        assert 0.9 < checks["total_mass"] < 1.1
        assert checks["pct_negative_density"] == 0.0  # After repair
