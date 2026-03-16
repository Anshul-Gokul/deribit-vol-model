"""
Visualization utilities for IV surfaces, densities, and backtest results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.surface.iv_surface import IVSurface
from src.distribution.risk_neutral_density import RiskNeutralDensity


def plot_smile(
    surface: IVSurface,
    T: float | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot IV smile for a given expiry (or all fitted expiries)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    times = [T] if T is not None else surface.expiry_times

    for t in times:
        smile = surface.get_smile(t)
        if smile is None:
            continue
        # Plot raw data points
        ax.scatter(
            smile.strikes, smile.ivs * 100,
            s=20, alpha=0.6, label=f"T={t:.3f}y data",
        )
        # Plot fitted curve
        K_fine = np.linspace(smile.strikes.min(), smile.strikes.max(), 200)
        iv_fine = smile.iv_at_strike(K_fine) * 100
        ax.plot(K_fine, iv_fine, linewidth=1.5, label=f"T={t:.3f}y fit")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title("Implied Volatility Smile")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_surface_3d(
    surface: IVSurface,
    save_path: str | Path | None = None,
) -> None:
    """Plot IV surface as 3D surface."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for T in surface.expiry_times:
        smile = surface.get_smile(T)
        if smile is None:
            continue
        K_fine = np.linspace(smile.strikes.min(), smile.strikes.max(), 50)
        T_arr = np.full_like(K_fine, T * 365.25)
        iv_fine = smile.iv_at_strike(K_fine) * 100
        ax.plot(K_fine, T_arr, iv_fine, linewidth=1)

    ax.set_xlabel("Strike")
    ax.set_ylabel("Days to Expiry")
    ax.set_zlabel("IV (%)")
    ax.set_title("Implied Volatility Surface")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_density(
    density: RiskNeutralDensity,
    save_path: str | Path | None = None,
) -> None:
    """Plot risk-neutral PDF and CDF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # PDF
    ax1.plot(density.strikes, density.pdf, "b-", linewidth=1.5)
    ax1.axvline(density.forward, color="r", linestyle="--", label=f"Forward={density.forward:.0f}")
    ax1.fill_between(density.strikes, density.pdf, alpha=0.15)
    ax1.set_xlabel("Price at Expiry")
    ax1.set_ylabel("Probability Density")
    ax1.set_title(f"Risk-Neutral PDF (T={density.T:.3f}y)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # CDF
    ax2.plot(density.strikes, density.cdf, "b-", linewidth=1.5)
    ax2.axvline(density.forward, color="r", linestyle="--", label=f"Forward={density.forward:.0f}")
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Price at Expiry")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title(f"Risk-Neutral CDF (T={density.T:.3f}y)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_probability_by_strike(
    density: RiskNeutralDensity,
    contract_type: str = "above",
    save_path: str | Path | None = None,
) -> None:
    """Plot implied probability as a function of strike."""
    fig, ax = plt.subplots(figsize=(10, 6))

    strikes = density.strikes
    if contract_type == "above":
        probs = np.array([density.prob_above(K) for K in strikes])
        ax.set_ylabel("P(S_T > K)")
        title = "Probability of Settling Above Strike"
    else:
        probs = np.array([density.prob_below(K) for K in strikes])
        ax.set_ylabel("P(S_T < K)")
        title = "Probability of Settling Below Strike"

    ax.plot(strikes, probs, "b-", linewidth=1.5)
    ax.axvline(density.forward, color="r", linestyle="--", label=f"Forward={density.forward:.0f}")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Strike")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_calibration(
    cal_data: dict[str, np.ndarray],
    save_path: str | Path | None = None,
) -> None:
    """Plot calibration (reliability) diagram."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    # Calibration curve
    ax.plot(
        cal_data["bin_avg_prob"],
        cal_data["bin_freqs"],
        "bo-",
        markersize=8,
        label="Model",
    )

    # Shade confidence around each bin
    for i in range(len(cal_data["bin_counts"])):
        n = cal_data["bin_counts"][i]
        p = cal_data["bin_freqs"][i]
        se = np.sqrt(p * (1 - p) / max(n, 1))
        ax.errorbar(
            cal_data["bin_avg_prob"][i],
            p,
            yerr=1.96 * se,
            fmt="none",
            color="blue",
            alpha=0.3,
        )

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
