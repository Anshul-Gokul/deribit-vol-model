"""
Prediction market contract pricing from options-implied probabilities.

Maps risk-neutral densities into fair prices for prediction-market-style contracts:
- ABOVE: P(S_T > K) — binary call
- BELOW: P(S_T < K) — binary put
- BETWEEN: P(A <= S_T <= B) — range contract
- OUTSIDE: P(S_T < A or S_T > B) — inverse range

Output format includes all relevant metadata for downstream consumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from src.distribution.risk_neutral_density import RiskNeutralDensity

logger = logging.getLogger(__name__)


@dataclass
class ContractPricing:
    """Result of pricing a prediction market contract."""

    asset: str
    contract_type: str  # "above", "below", "between", "outside"
    strike: float | None = None
    lower: float | None = None
    upper: float | None = None
    forecast_timestamp: str = ""
    settlement_timestamp: str = ""
    nearest_expiry_used: str = ""
    interpolation_method: str = ""
    forward_price: float = 0.0
    implied_probability: float = 0.0
    fair_yes_price: float = 0.0
    fair_no_price: float = 0.0
    implied_decimal_odds_yes: float = 0.0
    implied_decimal_odds_no: float = 0.0
    time_to_expiry_years: float = 0.0
    notes: str = ""

    def __post_init__(self):
        self.fair_yes_price = self.implied_probability
        self.fair_no_price = 1.0 - self.implied_probability
        if self.implied_probability > 0:
            self.implied_decimal_odds_yes = 1.0 / self.implied_probability
        if self.implied_probability < 1:
            self.implied_decimal_odds_no = 1.0 / (1.0 - self.implied_probability)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "asset": self.asset,
            "contract_type": self.contract_type,
            "strike": self.strike,
            "lower_bound": self.lower,
            "upper_bound": self.upper,
            "forecast_timestamp": self.forecast_timestamp,
            "settlement_timestamp": self.settlement_timestamp,
            "nearest_expiry_used": self.nearest_expiry_used,
            "interpolation_method": self.interpolation_method,
            "forward_price": round(self.forward_price, 2),
            "implied_probability": round(self.implied_probability, 6),
            "fair_yes_price": round(self.fair_yes_price, 6),
            "fair_no_price": round(self.fair_no_price, 6),
            "implied_decimal_odds_yes": round(self.implied_decimal_odds_yes, 4),
            "implied_decimal_odds_no": round(self.implied_decimal_odds_no, 4),
            "time_to_expiry_years": round(self.time_to_expiry_years, 6),
            "notes": self.notes,
        }

    def display(self) -> str:
        """Human-readable display string."""
        lines = []
        if self.contract_type == "above":
            desc = f"{self.asset} > {self.strike:.0f}"
        elif self.contract_type == "below":
            desc = f"{self.asset} < {self.strike:.0f}"
        elif self.contract_type == "between":
            desc = f"{self.lower:.0f} <= {self.asset} <= {self.upper:.0f}"
        elif self.contract_type == "outside":
            desc = f"{self.asset} < {self.lower:.0f} or {self.asset} > {self.upper:.0f}"
        else:
            desc = self.contract_type

        lines.append(f"Contract: {desc}")
        lines.append(f"Settlement: {self.settlement_timestamp}")
        lines.append(f"Forward: {self.forward_price:.2f}")
        lines.append(f"Implied Prob: {self.implied_probability:.4%}")
        lines.append(f"Fair YES: {self.fair_yes_price:.4f}")
        lines.append(f"Fair NO:  {self.fair_no_price:.4f}")
        lines.append(f"Odds YES: {self.implied_decimal_odds_yes:.2f}x")
        lines.append(f"Odds NO:  {self.implied_decimal_odds_no:.2f}x")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


def price_above(
    density: RiskNeutralDensity,
    strike: float,
    asset: str = "BTC",
    settlement: str = "",
    forecast_time: str = "",
    expiry_used: str = "",
    interp_method: str = "variance_linear",
) -> ContractPricing:
    """Price an ABOVE contract: P(S_T > K)."""
    prob = density.prob_above(strike)
    return ContractPricing(
        asset=asset,
        contract_type="above",
        strike=strike,
        forecast_timestamp=forecast_time,
        settlement_timestamp=settlement,
        nearest_expiry_used=expiry_used,
        interpolation_method=interp_method,
        forward_price=density.forward,
        implied_probability=prob,
        time_to_expiry_years=density.T,
        notes="Risk-neutral probability from Breeden-Litzenberger density",
    )


def price_below(
    density: RiskNeutralDensity,
    strike: float,
    asset: str = "BTC",
    settlement: str = "",
    forecast_time: str = "",
    expiry_used: str = "",
    interp_method: str = "variance_linear",
) -> ContractPricing:
    """Price a BELOW contract: P(S_T < K)."""
    prob = density.prob_below(strike)
    return ContractPricing(
        asset=asset,
        contract_type="below",
        strike=strike,
        forecast_timestamp=forecast_time,
        settlement_timestamp=settlement,
        nearest_expiry_used=expiry_used,
        interpolation_method=interp_method,
        forward_price=density.forward,
        implied_probability=prob,
        time_to_expiry_years=density.T,
        notes="Risk-neutral probability from Breeden-Litzenberger density",
    )


def price_between(
    density: RiskNeutralDensity,
    lower: float,
    upper: float,
    asset: str = "BTC",
    settlement: str = "",
    forecast_time: str = "",
    expiry_used: str = "",
    interp_method: str = "variance_linear",
) -> ContractPricing:
    """Price a BETWEEN contract: P(A <= S_T <= B)."""
    prob = density.prob_between(lower, upper)
    return ContractPricing(
        asset=asset,
        contract_type="between",
        lower=lower,
        upper=upper,
        forecast_timestamp=forecast_time,
        settlement_timestamp=settlement,
        nearest_expiry_used=expiry_used,
        interpolation_method=interp_method,
        forward_price=density.forward,
        implied_probability=prob,
        time_to_expiry_years=density.T,
        notes="Risk-neutral probability from Breeden-Litzenberger density",
    )


def price_outside(
    density: RiskNeutralDensity,
    lower: float,
    upper: float,
    asset: str = "BTC",
    settlement: str = "",
    forecast_time: str = "",
    expiry_used: str = "",
    interp_method: str = "variance_linear",
) -> ContractPricing:
    """Price an OUTSIDE contract: P(S_T < A or S_T > B)."""
    prob = density.prob_outside(lower, upper)
    return ContractPricing(
        asset=asset,
        contract_type="outside",
        lower=lower,
        upper=upper,
        forecast_timestamp=forecast_time,
        settlement_timestamp=settlement,
        nearest_expiry_used=expiry_used,
        interpolation_method=interp_method,
        forward_price=density.forward,
        implied_probability=prob,
        time_to_expiry_years=density.T,
        notes="Risk-neutral probability from Breeden-Litzenberger density",
    )


def price_contract(
    density: RiskNeutralDensity,
    contract_type: str,
    strike: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    asset: str = "BTC",
    settlement: str = "",
    forecast_time: str = "",
    expiry_used: str = "",
    interp_method: str = "variance_linear",
) -> ContractPricing:
    """
    Unified contract pricing dispatcher.

    Parameters
    ----------
    contract_type : str
        One of "above", "below", "between", "outside".
    strike : float, optional
        Required for "above" and "below".
    lower, upper : float, optional
        Required for "between" and "outside".
    """
    kwargs = dict(
        asset=asset,
        settlement=settlement,
        forecast_time=forecast_time,
        expiry_used=expiry_used,
        interp_method=interp_method,
    )

    if contract_type == "above":
        if strike is None:
            raise ValueError("strike required for 'above' contract")
        return price_above(density, strike, **kwargs)
    elif contract_type == "below":
        if strike is None:
            raise ValueError("strike required for 'below' contract")
        return price_below(density, strike, **kwargs)
    elif contract_type == "between":
        if lower is None or upper is None:
            raise ValueError("lower and upper required for 'between' contract")
        return price_between(density, lower, upper, **kwargs)
    elif contract_type == "outside":
        if lower is None or upper is None:
            raise ValueError("lower and upper required for 'outside' contract")
        return price_outside(density, lower, upper, **kwargs)
    else:
        raise ValueError(f"Unknown contract type: {contract_type}")


def price_bucket_ladder(
    density: RiskNeutralDensity,
    bucket_boundaries: list[float],
    asset: str = "BTC",
    settlement: str = "",
) -> list[ContractPricing]:
    """
    Price a ladder of adjacent bucket contracts.

    E.g., boundaries [90000, 95000, 100000, 105000, 110000] produces
    4 between contracts for each adjacent pair.
    """
    results = []
    for i in range(len(bucket_boundaries) - 1):
        lo = bucket_boundaries[i]
        hi = bucket_boundaries[i + 1]
        pricing = price_between(density, lo, hi, asset=asset, settlement=settlement)
        results.append(pricing)
    return results
