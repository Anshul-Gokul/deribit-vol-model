"""
02_surface_fitting.py — IV surface construction and analysis.

This script:
1. Loads and cleans options data
2. Fits the IV surface
3. Visualizes smile fits and surface
4. Extracts risk-neutral densities
5. Shows probability profiles
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
from src.utils.visualization import plot_smile, plot_density, plot_probability_by_strike


def main():
    asset = "BTC"
    snap = get_latest_snapshot(asset)
    if snap is None:
        print(f"No data. Run: python -m src.cli.main fetch-data --asset {asset}")
        return

    df_raw = load_snapshot(snap)
    df_clean = clean_options_data(df_raw)

    if df_clean.empty:
        print("No valid options after cleaning")
        return

    # Fit surface
    surface = IVSurface()
    surface.fit(df_clean)
    print(surface.summary())

    out = Path("reports")
    out.mkdir(exist_ok=True)

    # Plot all smiles
    plot_smile(surface, save_path=out / f"{asset}_smiles.png")
    print(f"Saved: {out / f'{asset}_smiles.png'}")

    # Density for each expiry
    for i, T in enumerate(surface.expiry_times[:4]):
        forward = surface._interpolate_forward(T)
        density = extract_density(surface, T, forward)
        dte = int(T * 365.25)

        checks = density.validate()
        print(f"\nT={T:.4f}y ({dte}d): mean={checks['mean']:.0f}, "
              f"fwd={checks['forward']:.0f}, std={checks['std']:.0f}, "
              f"mass={checks['total_mass']:.4f}")

        plot_density(density, save_path=out / f"{asset}_density_{dte}d.png")
        plot_probability_by_strike(density, save_path=out / f"{asset}_prob_{dte}d.png")

        # Example probability queries
        print(f"  P(> {forward:.0f}) = {density.prob_above(forward):.4f}")
        print(f"  P(> {forward*1.1:.0f}) = {density.prob_above(forward*1.1):.4f}")
        print(f"  P({forward*0.9:.0f}-{forward*1.1:.0f}) = "
              f"{density.prob_between(forward*0.9, forward*1.1):.4f}")

    print(f"\nAll plots saved to {out}/")


if __name__ == "__main__":
    main()
