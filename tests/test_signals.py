"""Tests for signal extraction logic."""

import numpy as np
import pytest

from src.models.signals import _compute_directional_score


class TestDirectionalScore:
    def test_all_bullish_returns_positive(self):
        score, conf = _compute_directional_score(
            basis_7d=5.0,  # positive basis
            rr25_7d=0.03,  # calls bid
            rr25_30d=0.02,
            iv_slope=0.02,  # contango
            pc_oi_ratio=0.7,  # low put/call
        )
        assert score > 0
        assert conf > 0

    def test_all_bearish_returns_negative(self):
        score, conf = _compute_directional_score(
            basis_7d=-3.0,
            rr25_7d=-0.04,
            rr25_30d=-0.03,
            iv_slope=-0.03,
            pc_oi_ratio=1.5,
        )
        assert score < 0
        assert conf > 0

    def test_neutral_near_zero(self):
        score, conf = _compute_directional_score(
            basis_7d=0.0,
            rr25_7d=0.0,
            rr25_30d=0.0,
            iv_slope=0.0,
            pc_oi_ratio=1.0,
        )
        assert abs(score) < 0.1

    def test_nan_signals_reduce_confidence(self):
        _, conf_full = _compute_directional_score(1.0, 0.01, 0.01, 0.01, 1.0)
        _, conf_partial = _compute_directional_score(np.nan, np.nan, 0.01, 0.01, 1.0)
        assert conf_partial < conf_full

    def test_score_bounded(self):
        score, _ = _compute_directional_score(100, 1.0, 1.0, 1.0, 0.01)
        assert -1 <= score <= 1
