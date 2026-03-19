"""Tests for forward curve construction."""

import numpy as np
import pytest
from datetime import datetime, timezone, timedelta

from src.models.forward_curve import ForwardCurve, ForwardPoint


def _make_curve(spot: float = 75000.0) -> ForwardCurve:
    """Build a simple test forward curve."""
    now = datetime.now(timezone.utc)
    points = [
        ForwardPoint(
            expiry=now + timedelta(days=7),
            T=7 / 365.25,
            forward=spot * 1.001,
            source="futures",
            basis_pct=0.1,
            annualized_basis=0.1 / (7 / 365.25),
        ),
        ForwardPoint(
            expiry=now + timedelta(days=30),
            T=30 / 365.25,
            forward=spot * 1.005,
            source="futures",
            basis_pct=0.5,
            annualized_basis=0.5 / (30 / 365.25),
        ),
        ForwardPoint(
            expiry=now + timedelta(days=90),
            T=90 / 365.25,
            forward=spot * 1.02,
            source="futures",
            basis_pct=2.0,
            annualized_basis=2.0 / (90 / 365.25),
        ),
    ]
    curve = ForwardCurve(spot=spot, as_of=now, points=points)
    curve.fit()
    return curve


class TestForwardCurve:
    def test_spot_at_zero(self):
        curve = _make_curve()
        assert curve.forward(0) == 75000.0

    def test_forward_monotone_in_contango(self):
        curve = _make_curve()
        f1 = curve.forward(7 / 365.25)
        f2 = curve.forward(30 / 365.25)
        f3 = curve.forward(90 / 365.25)
        assert f1 < f2 < f3

    def test_forward_interpolation(self):
        curve = _make_curve()
        f15 = curve.forward(15 / 365.25)
        assert 75000 < f15 < 75000 * 1.005

    def test_basis_positive_in_contango(self):
        curve = _make_curve()
        assert curve.basis(30 / 365.25) > 0

    def test_annualized_carry(self):
        curve = _make_curve()
        carry = curve.annualized_carry(30 / 365.25)
        # Should be positive and reasonable
        assert 0 < carry < 1.0

    def test_implied_rate(self):
        curve = _make_curve()
        r = curve.implied_rate(90 / 365.25)
        assert 0 < r < 0.5

    def test_curve_table_returns_df(self):
        curve = _make_curve()
        df = curve.curve_table()
        assert len(df) > 0
        assert "forward" in df.columns
        assert "basis_pct" in df.columns

    def test_empty_curve_returns_spot(self):
        curve = ForwardCurve(spot=75000, as_of=datetime.now(timezone.utc))
        curve.fit()
        assert curve.forward(1 / 365.25) == 75000.0
