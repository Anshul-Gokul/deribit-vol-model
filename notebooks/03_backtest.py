"""
03_backtest.py — Backtest options-implied probabilities.

This script demonstrates:
1. Strike-ladder probability estimation from a single snapshot
2. Cross-expiry comparison
3. Digital spread vs density method comparison
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.fetch_deribit import get_latest_snapshot, load_snapshot
from src.cleaning.options_cleaner import clean_options_data
from src.surface.iv_surface import IVSurface
from src.distribution.risk_neutral_density import extract_density
from src.pricing.contracts import price_bucket_ladder
from src.pricing.digitals import compare_digital_vs_density


def main():
    asset = "BTC"
    snap = get_latest_snapshot(asset)
    if snap is None:
        print(f"No data. Run: python -m src.cli.main fetch-data --asset {asset}")
        return

    df_raw = load_snapshot(snap)
    df_clean = clean_options_data(df_raw)

    surface = IVSurface()
    surface.fit(df_clean)

    out = Path("reports")
    out.mkdir(exist_ok=True)

    T = surface.expiry_times[0]
    forward = surface._interpolate_forward(T)
    density = extract_density(surface, T, forward)

    print(f"Asset: {asset}")
    print(f"Forward: {forward:.0f}")
    print(f"T: {T:.4f}y ({T*365.25:.0f}d)")

    # === Bucket ladder ===
    step = forward * 0.02  # 2% buckets
    boundaries = [forward * (0.85 + 0.02 * i) for i in range(16)]
    buckets = price_bucket_ladder(density, boundaries, asset=asset)

    print("\n--- Bucket Probabilities ---")
    for b in buckets:
        print(f"  [{b.lower:.0f}, {b.upper:.0f}): {b.implied_probability:.4f} "
              f"(YES: {b.fair_yes_price:.4f})")

    # Plot bucket distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    centers = [(b.lower + b.upper) / 2 for b in buckets]
    probs = [b.implied_probability for b in buckets]
    widths = [b.upper - b.lower for b in buckets]
    ax.bar(centers, probs, width=[w * 0.9 for w in widths], alpha=0.7)
    ax.axvline(forward, color="r", linestyle="--", label=f"Forward={forward:.0f}")
    ax.set_xlabel("Price Bucket")
    ax.set_ylabel("Probability")
    ax.set_title(f"{asset} Bucket Probabilities (T={T*365.25:.0f}d)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out / f"{asset}_buckets.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / f'{asset}_buckets.png'}")

    # === Digital vs Density comparison ===
    strikes = np.linspace(forward * 0.8, forward * 1.2, 50)
    comparison = compare_digital_vs_density(surface, density, strikes, T, forward)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(strikes, comparison["density_probs"], "b-", label="Density-based", linewidth=1.5)
    ax1.plot(strikes, comparison["spread_probs"], "r--", label="Digital spread", linewidth=1.5)
    ax1.axvline(forward, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("P(above)")
    ax1.set_title("Density vs Digital Spread Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(strikes, comparison["difference"] * 100, "g-", linewidth=1.5)
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Difference (pp)")
    ax2.set_title(f"Difference (max={comparison['max_abs_diff']*100:.2f}pp)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / f"{asset}_digital_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / f'{asset}_digital_comparison.png'}")

    print(f"\nDigital vs Density:")
    print(f"  Max abs diff: {comparison['max_abs_diff']*100:.2f} pp")
    print(f"  Mean abs diff: {comparison['mean_abs_diff']*100:.2f} pp")


if __name__ == "__main__":
    main()
