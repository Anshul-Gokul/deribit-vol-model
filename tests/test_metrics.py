"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    brier_score,
    log_loss,
    calibration_bins,
    calibration_error,
    sharpness,
    full_evaluation,
)


class TestBrierScore:
    def test_perfect_predictions(self):
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1, 0, 1, 0])
        assert brier_score(probs, outcomes) == 0.0

    def test_worst_predictions(self):
        probs = np.array([0.0, 1.0])
        outcomes = np.array([1, 0])
        assert brier_score(probs, outcomes) == 1.0

    def test_uniform_predictions(self):
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1, 0, 1, 0])
        assert abs(brier_score(probs, outcomes) - 0.25) < 1e-10


class TestLogLoss:
    def test_perfect_predictions(self):
        probs = np.array([0.999, 0.001, 0.999])
        outcomes = np.array([1, 0, 1])
        assert log_loss(probs, outcomes) < 0.01

    def test_bad_predictions(self):
        probs = np.array([0.1, 0.9])
        outcomes = np.array([1, 0])
        assert log_loss(probs, outcomes) > 1.0


class TestCalibration:
    def test_well_calibrated(self):
        np.random.seed(42)
        n = 10000
        probs = np.random.uniform(0, 1, n)
        outcomes = (np.random.uniform(0, 1, n) < probs).astype(float)
        ece = calibration_error(probs, outcomes)
        assert ece < 0.05  # Well-calibrated within bins

    def test_bins_output(self):
        probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        outcomes = np.array([0, 0, 0, 1, 1, 1])
        bins = calibration_bins(probs, outcomes, n_bins=5, min_obs=1)
        assert len(bins["bin_centers"]) > 0


class TestSharpness:
    def test_decisive(self):
        probs = np.array([0.01, 0.99, 0.02, 0.98])
        assert sharpness(probs) > 0.45

    def test_uncertain(self):
        probs = np.array([0.48, 0.52, 0.49, 0.51])
        assert sharpness(probs) < 0.05


class TestFullEvaluation:
    def test_returns_all_keys(self):
        probs = np.array([0.5, 0.6, 0.7])
        outcomes = np.array([0, 1, 1])
        result = full_evaluation(probs, outcomes)
        assert "brier_score" in result
        assert "log_loss" in result
        assert "calibration_error" in result
        assert "sharpness" in result
        assert "n_observations" in result
