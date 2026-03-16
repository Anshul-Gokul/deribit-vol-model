"""
Implied volatility surface construction.

Builds a smooth IV surface across strike (log-moneyness) and maturity (sqrt-time).
Two methods supported:

1. Cubic spline interpolation — fast, stable, works with sparse data
2. SVI (Stochastic Volatility Inspired) parametric fit — better extrapolation

The surface exposes methods to query:
- iv(K, T) — implied vol at any strike and time to expiry
- call_price(K, T) — BS call price using interpolated IV
- put_price(K, T) — BS put price using interpolated IV
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.optimize import minimize
import pandas as pd

from src.utils.black_scholes import bs_call, bs_put, bs_call_vec, bs_put_vec
from src.utils.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class SmileSlice:
    """IV smile for a single expiry."""

    expiry: datetime
    T: float
    forward: float
    strikes: np.ndarray
    ivs: np.ndarray
    log_moneyness: np.ndarray
    _spline: CubicSpline | None = field(default=None, repr=False)

    def fit_spline(self, smoothing: float = 0.0) -> None:
        """Fit cubic spline to the smile."""
        # Sort by log-moneyness
        order = np.argsort(self.log_moneyness)
        x = self.log_moneyness[order]
        y = self.ivs[order]

        # Remove duplicates
        _, unique_idx = np.unique(x, return_index=True)
        x = x[unique_idx]
        y = y[unique_idx]

        if len(x) < 3:
            logger.warning(f"Only {len(x)} points for T={self.T:.4f} — linear interpolation")
            self._spline = CubicSpline(x, y, bc_type="natural", extrapolate=True)
        else:
            self._spline = CubicSpline(x, y, bc_type="natural", extrapolate=True)

    def iv_at_moneyness(self, log_m: float | np.ndarray) -> float | np.ndarray:
        """Query IV at given log-moneyness."""
        if self._spline is None:
            self.fit_spline()
        result = self._spline(log_m)
        # Clip to sensible range
        return np.clip(result, 0.01, 5.0)

    def iv_at_strike(self, K: float | np.ndarray) -> float | np.ndarray:
        """Query IV at given absolute strike."""
        log_m = np.log(np.asarray(K, dtype=float) / self.forward)
        return self.iv_at_moneyness(log_m)


def _svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """
    SVI total variance parametrization.

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    where k = log(K/F) is log-moneyness, w = sigma^2 * T is total variance.
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def fit_svi_slice(
    log_moneyness: np.ndarray,
    total_variance: np.ndarray,
    initial_guess: list[float] | None = None,
) -> dict[str, float]:
    """
    Fit SVI parameters to a single smile slice.

    Parameters
    ----------
    log_moneyness : array of log(K/F)
    total_variance : array of IV^2 * T (total variance)

    Returns dict with keys: a, b, rho, m, sigma
    """
    if initial_guess is None:
        # Heuristic initial guess
        avg_w = np.mean(total_variance)
        initial_guess = [avg_w, 0.3, -0.2, 0.0, 0.1]

    def objective(params):
        a, b, rho, m, sigma = params
        w_model = _svi_total_variance(log_moneyness, a, b, rho, m, sigma)
        return np.sum((w_model - total_variance) ** 2)

    # Bounds: a>0, b>0, -1<rho<1, sigma>0
    bounds = [
        (0.001, None),  # a
        (0.001, None),  # b
        (-0.999, 0.999),  # rho
        (-1.0, 1.0),  # m
        (0.001, None),  # sigma
    ]

    result = minimize(objective, initial_guess, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        logger.warning(f"SVI fit did not converge: {result.message}")

    a, b, rho, m, sigma = result.x
    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma}


class IVSurface:
    """
    Implied volatility surface.

    Supports querying IV at any (strike, time-to-expiry) pair by:
    1. Fitting per-expiry smile slices
    2. Interpolating across time using variance-linear interpolation
    """

    def __init__(self, config: dict | None = None):
        cfg = (config or get_config()).get("surface", {})
        self.method = cfg.get("method", "spline")
        self.smoothing = cfg.get("spline_smoothing", 0.01)
        self.min_strikes = cfg.get("min_strikes_per_expiry", 5)
        self.slices: dict[float, SmileSlice] = {}  # keyed by T
        self._sorted_T: np.ndarray | None = None
        self.forward_curve: dict[float, float] = {}  # T -> forward

    def fit(self, df_clean: pd.DataFrame) -> None:
        """
        Fit the IV surface from cleaned options data.

        The DataFrame must have columns: expiry_dt, T, forward, strike, log_moneyness,
        and either mark_iv_decimal or mid_usd.
        """
        self.slices.clear()
        self.forward_curve.clear()

        expiries = sorted(df_clean["expiry_dt"].unique())

        for expiry in expiries:
            slice_df = df_clean[df_clean["expiry_dt"] == expiry].copy()

            if len(slice_df) < self.min_strikes:
                logger.info(
                    f"Skipping expiry {expiry}: only {len(slice_df)} strikes "
                    f"(min {self.min_strikes})"
                )
                continue

            T = slice_df["T"].iloc[0]
            forward = slice_df["forward"].iloc[0]

            # Get IVs — prefer mark_iv from Deribit, fall back to computing from price
            if "mark_iv_decimal" in slice_df.columns and slice_df["mark_iv_decimal"].notna().sum() > 0:
                ivs = slice_df["mark_iv_decimal"].values.astype(float)
            else:
                logger.warning(f"No mark IV for expiry {expiry}, skipping")
                continue

            strikes = slice_df["strike"].values.astype(float)
            log_m = slice_df["log_moneyness"].values.astype(float)

            # Filter valid IVs
            valid = (ivs > 0.01) & (ivs < 5.0) & np.isfinite(ivs)
            if valid.sum() < 3:
                logger.info(f"Skipping expiry {expiry}: too few valid IVs")
                continue

            smile = SmileSlice(
                expiry=expiry,
                T=T,
                forward=forward,
                strikes=strikes[valid],
                ivs=ivs[valid],
                log_moneyness=log_m[valid],
            )
            smile.fit_spline(self.smoothing)
            self.slices[T] = smile
            self.forward_curve[T] = forward

        self._sorted_T = np.array(sorted(self.slices.keys()))
        logger.info(f"IV surface fitted with {len(self.slices)} expiry slices")

    def iv(self, K: float, T: float, forward: float | None = None) -> float:
        """
        Query implied volatility at strike K and time T.

        Uses variance-linear interpolation between the two nearest fitted slices.
        """
        if not self.slices:
            raise RuntimeError("Surface not fitted — call fit() first")

        if forward is None:
            forward = self._interpolate_forward(T)

        log_m = np.log(K / forward)

        # Find bracketing slices
        sorted_T = self._sorted_T

        if T <= sorted_T[0]:
            return float(self.slices[sorted_T[0]].iv_at_moneyness(log_m))

        if T >= sorted_T[-1]:
            return float(self.slices[sorted_T[-1]].iv_at_moneyness(log_m))

        # Find bracket
        idx = np.searchsorted(sorted_T, T)
        T_lo = sorted_T[idx - 1]
        T_hi = sorted_T[idx]

        iv_lo = float(self.slices[T_lo].iv_at_moneyness(log_m))
        iv_hi = float(self.slices[T_hi].iv_at_moneyness(log_m))

        # Variance-linear interpolation: IV^2 * T is linear in T
        w_lo = iv_lo**2 * T_lo
        w_hi = iv_hi**2 * T_hi

        # Linear interpolation in total variance
        alpha = (T - T_lo) / (T_hi - T_lo)
        w_interp = w_lo + alpha * (w_hi - w_lo)

        if w_interp <= 0 or T <= 0:
            return iv_lo  # fallback

        iv_interp = np.sqrt(w_interp / T)
        return float(np.clip(iv_interp, 0.01, 5.0))

    def call_price(self, K: float, T: float, forward: float | None = None, r: float = 0.0) -> float:
        """BS call price at (K, T) using surface IV."""
        if forward is None:
            forward = self._interpolate_forward(T)
        sigma = self.iv(K, T, forward)
        return bs_call(forward, K, T, sigma, r)

    def put_price(self, K: float, T: float, forward: float | None = None, r: float = 0.0) -> float:
        """BS put price at (K, T) using surface IV."""
        if forward is None:
            forward = self._interpolate_forward(T)
        sigma = self.iv(K, T, forward)
        return bs_put(forward, K, T, sigma, r)

    def call_prices_on_grid(
        self, strikes: np.ndarray, T: float, forward: float | None = None, r: float = 0.0
    ) -> np.ndarray:
        """Vectorized call prices across a strike grid for a single T."""
        if forward is None:
            forward = self._interpolate_forward(T)
        ivs = np.array([self.iv(K, T, forward) for K in strikes])
        return bs_call_vec(forward, strikes, T, ivs, r)

    def _interpolate_forward(self, T: float) -> float:
        """Interpolate forward price for a given T."""
        if not self.forward_curve:
            raise RuntimeError("No forward curve available")

        sorted_T = self._sorted_T
        if T <= sorted_T[0]:
            return self.forward_curve[sorted_T[0]]
        if T >= sorted_T[-1]:
            return self.forward_curve[sorted_T[-1]]

        idx = np.searchsorted(sorted_T, T)
        T_lo = sorted_T[idx - 1]
        T_hi = sorted_T[idx]
        alpha = (T - T_lo) / (T_hi - T_lo)
        return self.forward_curve[T_lo] + alpha * (self.forward_curve[T_hi] - self.forward_curve[T_lo])

    def get_smile(self, T: float) -> SmileSlice | None:
        """Get the nearest smile slice for a given T."""
        if not self.slices:
            return None
        nearest_T = min(self.slices.keys(), key=lambda t: abs(t - T))
        return self.slices[nearest_T]

    @property
    def expiry_times(self) -> list[float]:
        """List of fitted expiry times."""
        return sorted(self.slices.keys())

    def summary(self) -> str:
        """Human-readable summary of the surface."""
        lines = [f"IV Surface ({self.method}): {len(self.slices)} slices"]
        for T in sorted(self.slices.keys()):
            s = self.slices[T]
            lines.append(
                f"  T={T:.4f}y ({T*365.25:.0f}d) | F={s.forward:.0f} | "
                f"{len(s.strikes)} strikes | "
                f"IV range: [{s.ivs.min():.1%}, {s.ivs.max():.1%}]"
            )
        return "\n".join(lines)
