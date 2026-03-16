"""
Options data cleaning and normalization.

Takes raw Deribit book summary data and produces a clean DataFrame suitable
for surface fitting. Key operations:

1. Parse instrument names to extract strike, expiry, option type
2. Compute mid prices from bid/ask
3. Filter stale, crossed, illiquid, or nonsensical quotes
4. Reject quotes violating basic no-arbitrage bounds
5. Select OTM options relative to forward for surface fitting
6. Compute time to expiry and log-moneyness
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.utils.time_utils import parse_deribit_instrument, time_to_expiry_years
from src.utils.config import get_config

logger = logging.getLogger(__name__)


def clean_options_data(
    df_raw: pd.DataFrame,
    as_of: datetime | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Clean and normalize raw Deribit options data.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw book summary data from Deribit API.
    as_of : datetime, optional
        Reference time for computing time to expiry. Default: now.
    config : dict, optional
        Cleaning configuration. Default: from configs/default.yaml.

    Returns
    -------
    pd.DataFrame
        Cleaned options data with computed fields.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)

    cfg = (config or get_config()).get("cleaning", {})
    min_bid = cfg.get("min_bid", 0.0001)
    max_spread_pct = cfg.get("max_spread_pct", 0.50)
    min_oi = cfg.get("min_open_interest", 0)
    min_vol = cfg.get("min_volume", 0)
    otm_only = cfg.get("otm_only", True)
    min_dte = cfg.get("min_dte", 0.01)
    max_dte = cfg.get("max_dte", 365)

    df = df_raw.copy()

    # Parse instrument names
    parsed = df["instrument_name"].apply(parse_deribit_instrument)
    parsed_df = pd.DataFrame(parsed.tolist())
    df = pd.concat([df, parsed_df], axis=1)

    # Keep only options
    df = df[df["instrument_type"] == "option"].copy()
    if df.empty:
        logger.warning("No option instruments found after parsing")
        return df

    # Get underlying/index price
    if "underlying_price" in df.columns:
        df["spot"] = df["underlying_price"]
    elif "index_price" in df.columns:
        df["spot"] = df["index_price"]
    else:
        logger.error("No underlying or index price found in data")
        return pd.DataFrame()

    # Compute time to expiry
    df["T"] = df["expiry_dt"].apply(lambda x: time_to_expiry_years(x, as_of))
    df["dte"] = df["T"] * 365.25

    # Filter by time to expiry
    n_before = len(df)
    df = df[(df["dte"] >= min_dte) & (df["dte"] <= max_dte)].copy()
    logger.info(f"DTE filter: {n_before} -> {len(df)} options")

    # Extract prices — Deribit reports in BTC/ETH terms, multiply by underlying
    # bid_price, ask_price, mark_price are in underlying units (fraction of 1 BTC)
    # We convert to USD for pricing
    for col in ["bid_price", "ask_price", "mark_price"]:
        if col in df.columns:
            df[f"{col}_usd"] = df[col].astype(float) * df["spot"]
        else:
            df[f"{col}_usd"] = np.nan

    # Compute mid price
    df["mid_usd"] = (df["bid_price_usd"] + df["ask_price_usd"]) / 2.0
    # Fall back to mark price if bid/ask unavailable
    mask_no_mid = df["mid_usd"].isna() | (df["mid_usd"] <= 0)
    df.loc[mask_no_mid, "mid_usd"] = df.loc[mask_no_mid, "mark_price_usd"]

    # Get mark IV from Deribit (reported as percentage)
    if "mark_iv" in df.columns:
        df["mark_iv_decimal"] = df["mark_iv"].astype(float) / 100.0
    else:
        df["mark_iv_decimal"] = np.nan

    # === FILTERING ===

    # 1. Reject zero or negative bids
    n_before = len(df)
    df = df[df["bid_price"].astype(float) >= min_bid].copy()
    logger.info(f"Min bid filter: {n_before} -> {len(df)}")

    # 2. Reject crossed markets
    n_before = len(df)
    df = df[df["ask_price"].astype(float) > df["bid_price"].astype(float)].copy()
    logger.info(f"Crossed market filter: {n_before} -> {len(df)}")

    # 3. Reject wide spreads
    n_before = len(df)
    spread = df["ask_price_usd"] - df["bid_price_usd"]
    spread_pct = spread / df["mid_usd"]
    df = df[spread_pct <= max_spread_pct].copy()
    logger.info(f"Spread filter: {n_before} -> {len(df)}")

    # 4. Open interest filter
    if "open_interest" in df.columns and min_oi > 0:
        n_before = len(df)
        df = df[df["open_interest"].astype(float) >= min_oi].copy()
        logger.info(f"OI filter: {n_before} -> {len(df)}")

    # 5. Volume filter
    if "volume" in df.columns and min_vol > 0:
        n_before = len(df)
        df = df[df["volume"].astype(float) >= min_vol].copy()
        logger.info(f"Volume filter: {n_before} -> {len(df)}")

    # 6. Basic no-arbitrage checks
    n_before = len(df)
    calls = df["option_type"] == "C"
    puts = df["option_type"] == "P"

    # Call price should not exceed spot
    bad_calls = calls & (df["mid_usd"] > df["spot"])
    # Put price should not exceed strike * exp(-rT)
    bad_puts = puts & (df["mid_usd"] > df["strike"])
    # Option price must be positive
    bad_price = df["mid_usd"] <= 0

    df = df[~(bad_calls | bad_puts | bad_price)].copy()
    logger.info(f"No-arb filter: {n_before} -> {len(df)}")

    # 7. Select OTM options only (calls for K > F, puts for K < F)
    # Use spot as proxy for forward when futures data not available
    if otm_only:
        n_before = len(df)
        otm_calls = calls & (df["strike"] >= df["spot"])
        otm_puts = puts & (df["strike"] <= df["spot"])
        # Keep ATM for both
        atm_band = np.abs(df["strike"] / df["spot"] - 1.0) < 0.005
        df = df[otm_calls | otm_puts | atm_band].copy()
        logger.info(f"OTM filter: {n_before} -> {len(df)}")

    # Compute log-moneyness: ln(K/F) where F ≈ spot for now
    df["forward"] = df["spot"]  # Will be updated with futures-implied forward
    df["log_moneyness"] = np.log(df["strike"] / df["forward"])
    df["moneyness"] = df["strike"] / df["forward"]

    # Sort
    df = df.sort_values(["expiry_dt", "strike"]).reset_index(drop=True)

    logger.info(f"Clean data: {len(df)} options across {df['expiry_dt'].nunique()} expiries")
    return df


def extract_forwards_from_futures(
    df_futures: pd.DataFrame,
    as_of: datetime | None = None,
) -> dict[datetime, float]:
    """
    Extract forward prices from futures data.

    Returns dict mapping expiry datetime to forward price.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)

    forwards = {}
    for _, row in df_futures.iterrows():
        parsed = parse_deribit_instrument(row["instrument_name"])
        if parsed.get("instrument_type") == "perpetual":
            continue
        if "expiry_dt" not in parsed:
            continue
        expiry = parsed["expiry_dt"]
        # Use mark_price as forward, converted to USD
        if "mark_price" in row and row["mark_price"] is not None:
            price = float(row["mark_price"])
            if price > 0:
                forwards[expiry] = price

    return forwards


def update_forwards(df_options: pd.DataFrame, forwards: dict[datetime, float]) -> pd.DataFrame:
    """Update forward prices in cleaned options data using futures-implied forwards."""
    df = df_options.copy()
    for expiry, fwd in forwards.items():
        mask = df["expiry_dt"] == expiry
        df.loc[mask, "forward"] = fwd
        df.loc[mask, "log_moneyness"] = np.log(df.loc[mask, "strike"] / fwd)
        df.loc[mask, "moneyness"] = df.loc[mask, "strike"] / fwd
    return df


def get_available_expiries(df: pd.DataFrame) -> list[datetime]:
    """Get sorted list of unique expiry dates in the data."""
    return sorted(df["expiry_dt"].unique().tolist())


def get_expiry_slice(df: pd.DataFrame, expiry: datetime) -> pd.DataFrame:
    """Get all options for a specific expiry."""
    return df[df["expiry_dt"] == expiry].copy()
