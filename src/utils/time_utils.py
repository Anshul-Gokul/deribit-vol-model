"""
Time and expiry utilities.

Handles:
- Parsing Deribit instrument names to extract expiry dates
- Computing time to expiry in years
- Mapping arbitrary target timestamps to Deribit expiry dates
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta

# Deribit month codes
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_deribit_instrument(instrument_name: str) -> dict:
    """
    Parse a Deribit instrument name.

    Examples:
        BTC-28MAR26-100000-C -> {asset: BTC, expiry_str: 28MAR26, strike: 100000, option_type: C}
        ETH-PERPETUAL -> {asset: ETH, instrument_type: perpetual}
        BTC-28MAR26 -> {asset: BTC, expiry_str: 28MAR26, instrument_type: future}
    """
    parts = instrument_name.split("-")
    result = {"instrument_name": instrument_name, "asset": parts[0]}

    if len(parts) == 4:
        # Option: ASSET-EXPIRY-STRIKE-TYPE
        result["expiry_str"] = parts[1]
        result["strike"] = float(parts[2])
        result["option_type"] = parts[3].upper()  # C or P
        result["instrument_type"] = "option"
        result["expiry_dt"] = parse_deribit_expiry(parts[1])
    elif len(parts) == 2:
        if parts[1].upper() == "PERPETUAL":
            result["instrument_type"] = "perpetual"
        else:
            result["expiry_str"] = parts[1]
            result["instrument_type"] = "future"
            result["expiry_dt"] = parse_deribit_expiry(parts[1])
    elif len(parts) == 3:
        # Could be a combo or future spread — treat as future
        result["expiry_str"] = parts[1]
        result["instrument_type"] = "future"
        try:
            result["expiry_dt"] = parse_deribit_expiry(parts[1])
        except (ValueError, KeyError):
            pass

    return result


def parse_deribit_expiry(expiry_str: str) -> datetime:
    """
    Parse Deribit expiry string to datetime.

    Format: DDMMMYY, e.g., 28MAR26
    Deribit options expire at 08:00 UTC on the expiry date.
    """
    match = re.match(r"^(\d{1,2})([A-Z]{3})(\d{2})$", expiry_str.upper())
    if not match:
        raise ValueError(f"Cannot parse expiry: {expiry_str}")

    day = int(match.group(1))
    month = _MONTH_MAP[match.group(2)]
    year = 2000 + int(match.group(3))

    return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)


def time_to_expiry_years(expiry: datetime, as_of: datetime | None = None) -> float:
    """
    Compute time to expiry in years (365.25 day convention).

    Returns 0 if expiry has passed.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    delta = (expiry - as_of).total_seconds()
    return max(delta / (365.25 * 24 * 3600), 0.0)


def find_bracketing_expiries(
    target: datetime,
    available_expiries: list[datetime],
) -> tuple[datetime | None, datetime | None]:
    """
    Find the two expiries that bracket the target timestamp.

    Returns (before, after) where before <= target <= after.
    Either can be None if the target is outside the range.
    """
    sorted_expiries = sorted(available_expiries)
    before = None
    after = None

    for exp in sorted_expiries:
        if exp <= target:
            before = exp
        elif after is None:
            after = exp

    return before, after


def find_nearest_expiry(
    target: datetime,
    available_expiries: list[datetime],
) -> datetime:
    """Find the single nearest expiry to the target."""
    if not available_expiries:
        raise ValueError("No available expiries")
    return min(available_expiries, key=lambda e: abs((e - target).total_seconds()))


def interpolation_weights(
    target: datetime,
    before: datetime,
    after: datetime,
) -> tuple[float, float]:
    """
    Compute weights for variance-linear interpolation between two expiries.

    For IV surfaces, total variance (IV^2 * T) is approximately linear in time.
    Weight is based on time distance.
    """
    t_target = (target - before).total_seconds()
    t_total = (after - before).total_seconds()
    if t_total <= 0:
        return 1.0, 0.0
    w_after = t_target / t_total
    w_before = 1.0 - w_after
    return w_before, w_after
