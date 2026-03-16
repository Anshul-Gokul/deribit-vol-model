"""
Evaluation metrics for probability forecasts.

Implements standard proper scoring rules and calibration diagnostics:
- Brier score: mean squared error of probability forecasts
- Log loss: cross-entropy loss
- Calibration: binned reliability analysis
- Sharpness: how spread-out the forecasted probabilities are
"""

from __future__ import annotations

import numpy as np


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Brier score: mean( (p - o)^2 ).

    Parameters
    ----------
    probs : array of predicted probabilities in [0, 1]
    outcomes : array of binary outcomes (0 or 1)

    Returns
    -------
    float
        Brier score. Lower is better. Perfect = 0, worst = 1.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - outcomes) ** 2))


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """
    Log loss (binary cross-entropy).

    Parameters
    ----------
    probs : array of predicted probabilities in [0, 1]
    outcomes : array of binary outcomes (0 or 1)
    eps : float
        Clipping epsilon to avoid log(0).

    Returns
    -------
    float
        Log loss. Lower is better. Perfect = 0.
    """
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))


def calibration_bins(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    min_obs: int = 5,
) -> dict[str, np.ndarray]:
    """
    Compute calibration (reliability) data in bins.

    Parameters
    ----------
    probs : array of predicted probabilities
    outcomes : array of binary outcomes
    n_bins : int
        Number of bins.
    min_obs : int
        Minimum observations per bin to include.

    Returns
    -------
    dict with keys:
        bin_centers : array of bin center probabilities
        bin_freqs : array of observed frequencies in each bin
        bin_counts : array of observation counts per bin
        bin_avg_prob : array of average predicted probability in each bin
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_freqs = []
    bin_counts = []
    bin_avg_prob = []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])

        count = mask.sum()
        if count < min_obs:
            continue

        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        bin_freqs.append(outcomes[mask].mean())
        bin_counts.append(count)
        bin_avg_prob.append(probs[mask].mean())

    return {
        "bin_centers": np.array(bin_centers),
        "bin_freqs": np.array(bin_freqs),
        "bin_counts": np.array(bin_counts),
        "bin_avg_prob": np.array(bin_avg_prob),
    }


def calibration_error(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).

    Weighted average absolute difference between predicted and observed frequency.
    """
    bins = calibration_bins(probs, outcomes, n_bins, min_obs=1)
    if len(bins["bin_counts"]) == 0:
        return float("nan")

    weights = bins["bin_counts"] / bins["bin_counts"].sum()
    return float(np.sum(weights * np.abs(bins["bin_avg_prob"] - bins["bin_freqs"])))


def sharpness(probs: np.ndarray) -> float:
    """
    Sharpness: average distance of probabilities from 0.5.

    Higher = sharper/more decisive forecasts.
    """
    return float(np.mean(np.abs(np.asarray(probs) - 0.5)))


def resolution(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """
    Resolution: how much the predicted probabilities differ from the overall base rate.

    Higher = better ability to discriminate.
    """
    bins = calibration_bins(probs, outcomes, n_bins, min_obs=1)
    if len(bins["bin_counts"]) == 0:
        return 0.0

    base_rate = outcomes.mean()
    weights = bins["bin_counts"] / bins["bin_counts"].sum()
    return float(np.sum(weights * (bins["bin_freqs"] - base_rate) ** 2))


def full_evaluation(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute all evaluation metrics.

    Returns dict with all scores.
    """
    return {
        "brier_score": brier_score(probs, outcomes),
        "log_loss": log_loss(probs, outcomes),
        "calibration_error": calibration_error(probs, outcomes, n_bins),
        "sharpness": sharpness(probs),
        "resolution": resolution(probs, outcomes, n_bins),
        "n_observations": len(probs),
        "base_rate": float(np.mean(outcomes)),
    }
