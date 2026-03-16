"""Tests for time and expiry utilities."""

from datetime import datetime, timezone, timedelta

import pytest

from src.utils.time_utils import (
    parse_deribit_instrument,
    parse_deribit_expiry,
    time_to_expiry_years,
    find_bracketing_expiries,
    find_nearest_expiry,
    interpolation_weights,
)


class TestParseDeribitInstrument:
    def test_option(self):
        result = parse_deribit_instrument("BTC-28MAR26-100000-C")
        assert result["asset"] == "BTC"
        assert result["strike"] == 100000
        assert result["option_type"] == "C"
        assert result["instrument_type"] == "option"

    def test_put(self):
        result = parse_deribit_instrument("ETH-28MAR26-2500-P")
        assert result["asset"] == "ETH"
        assert result["strike"] == 2500
        assert result["option_type"] == "P"

    def test_perpetual(self):
        result = parse_deribit_instrument("BTC-PERPETUAL")
        assert result["instrument_type"] == "perpetual"

    def test_future(self):
        result = parse_deribit_instrument("BTC-28MAR26")
        assert result["instrument_type"] == "future"


class TestParseExpiry:
    def test_standard(self):
        dt = parse_deribit_expiry("28MAR26")
        assert dt.year == 2026
        assert dt.month == 3
        assert dt.day == 28
        assert dt.hour == 8  # Deribit settles at 08:00 UTC

    def test_single_digit_day(self):
        dt = parse_deribit_expiry("5JAN26")
        assert dt.day == 5
        assert dt.month == 1


class TestTimeToExpiry:
    def test_future_expiry(self):
        as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
        expiry = datetime(2026, 4, 1, tzinfo=timezone.utc)
        T = time_to_expiry_years(expiry, as_of)
        assert 0.24 < T < 0.26  # ~90 days

    def test_past_expiry(self):
        as_of = datetime(2026, 4, 1, tzinfo=timezone.utc)
        expiry = datetime(2026, 1, 1, tzinfo=timezone.utc)
        T = time_to_expiry_years(expiry, as_of)
        assert T == 0.0


class TestBracketingExpiries:
    def test_normal_bracket(self):
        expiries = [
            datetime(2026, 3, 28, 8, tzinfo=timezone.utc),
            datetime(2026, 4, 25, 8, tzinfo=timezone.utc),
            datetime(2026, 6, 27, 8, tzinfo=timezone.utc),
        ]
        target = datetime(2026, 4, 10, tzinfo=timezone.utc)
        before, after = find_bracketing_expiries(target, expiries)
        assert before == expiries[0]
        assert after == expiries[1]

    def test_before_all(self):
        expiries = [datetime(2026, 6, 1, tzinfo=timezone.utc)]
        target = datetime(2026, 3, 1, tzinfo=timezone.utc)
        before, after = find_bracketing_expiries(target, expiries)
        assert before is None
        assert after == expiries[0]


class TestInterpolationWeights:
    def test_midpoint(self):
        before = datetime(2026, 3, 1, tzinfo=timezone.utc)
        after = datetime(2026, 3, 31, tzinfo=timezone.utc)
        target = datetime(2026, 3, 16, tzinfo=timezone.utc)
        w_b, w_a = interpolation_weights(target, before, after)
        assert abs(w_b - 0.5) < 0.05
        assert abs(w_a - 0.5) < 0.05
        assert abs(w_b + w_a - 1.0) < 1e-10
