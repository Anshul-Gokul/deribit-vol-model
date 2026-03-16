"""
Black-Scholes pricing and implied volatility.

European options on a non-dividend-paying asset (appropriate for crypto options
on Deribit which are European-style and settled in the underlying).

All prices are in units of the underlying (e.g., BTC-denominated for BTC options).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def d1(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Compute d1 in Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Compute d2 in Black-Scholes formula."""
    return d1(S, K, T, sigma, r) - sigma * np.sqrt(T)


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes European call price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1_val = d1(S, K, T, sigma, r)
    d2_val = d1_val - sigma * np.sqrt(T)
    return S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)


def bs_put(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes European put price."""
    if T <= 0:
        return max(K - S, 0.0)
    d1_val = d1(S, K, T, sigma, r)
    d2_val = d1_val - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)


def bs_call_vec(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    sigma: np.ndarray | float,
    r: float = 0.0,
) -> np.ndarray:
    """Vectorized Black-Scholes call price."""
    S, K, T, sigma = np.broadcast_arrays(
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    result = np.zeros_like(S)
    valid = (T > 0) & (sigma > 0) & (K > 0) & (S > 0)
    if not valid.any():
        return np.maximum(S - K, 0.0)

    Sv, Kv, Tv, sv = S[valid], K[valid], T[valid], sigma[valid]
    d1v = (np.log(Sv / Kv) + (r + 0.5 * sv**2) * Tv) / (sv * np.sqrt(Tv))
    d2v = d1v - sv * np.sqrt(Tv)
    result[valid] = Sv * norm.cdf(d1v) - Kv * np.exp(-r * Tv) * norm.cdf(d2v)

    expired = ~valid & (T <= 0)
    result[expired] = np.maximum(S[expired] - K[expired], 0.0)
    return result


def bs_put_vec(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    sigma: np.ndarray | float,
    r: float = 0.0,
) -> np.ndarray:
    """Vectorized Black-Scholes put price."""
    call = bs_call_vec(S, K, T, sigma, r)
    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    return call - np.asarray(S, dtype=float) + K_arr * np.exp(-r * T_arr)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    option_type: str = "call",
    vol_bounds: tuple[float, float] = (0.01, 10.0),
) -> float | None:
    """
    Compute implied volatility using Brent's method.

    Returns None if no solution found.
    """
    if T <= 0 or price <= 0:
        return None

    pricer = bs_call if option_type == "call" else bs_put

    # Check bounds
    low_price = pricer(S, K, T, vol_bounds[0], r)
    high_price = pricer(S, K, T, vol_bounds[1], r)

    if price < low_price or price > high_price:
        return None

    try:
        iv = brentq(lambda sig: pricer(S, K, T, sig, r) - price, vol_bounds[0], vol_bounds[1])
        return iv
    except (ValueError, RuntimeError):
        return None


def implied_vol_from_mark_iv(mark_iv: float) -> float:
    """
    Convert Deribit mark IV (percentage) to decimal.

    Deribit reports IV as a percentage (e.g., 80 means 80% annualized vol).
    """
    return mark_iv / 100.0


def bs_digital_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """
    Price of a digital (binary) call that pays 1 if S_T > K.

    Under Black-Scholes: P(S_T > K) = N(d2) under risk-neutral measure.
    """
    if T <= 0:
        return 1.0 if S > K else 0.0
    d2_val = d2(S, K, T, sigma, r)
    return np.exp(-r * T) * norm.cdf(d2_val)


def bs_digital_put(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """
    Price of a digital (binary) put that pays 1 if S_T < K.
    """
    return np.exp(-r * T) - bs_digital_call(S, K, T, sigma, r)
