"""
01_exploration.py — Exploratory analysis of Deribit options data.

Run with: python notebooks/01_exploration.py
Or convert to notebook with jupytext.

This script:
1. Loads a saved options snapshot
2. Examines data structure and quality
3. Visualizes bid-ask spreads, open interest by strike
4. Identifies available expiries and their liquidity
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.fetch_deribit import get_latest_snapshot, load_snapshot
from src.cleaning.options_cleaner import clean_options_data
from src.utils.time_utils import parse_deribit_instrument


def main():
    asset = "BTC"
    snap = get_latest_snapshot(asset)

    if snap is None:
        print(f"No snapshot found for {asset}. Run: python -m src.cli.main fetch-data --asset {asset}")
        return

    print(f"Loading snapshot: {snap}")
    df_raw = load_snapshot(snap)
    print(f"Raw data: {len(df_raw)} instruments")
    print(f"Columns: {list(df_raw.columns)}")

    # Parse instrument names
    parsed = df_raw["instrument_name"].apply(parse_deribit_instrument)
    parsed_df = pd.DataFrame(parsed.tolist())
    df = pd.concat([df_raw, parsed_df], axis=1)
    df = df[df["instrument_type"] == "option"]

    print(f"\nOptions: {len(df)}")
    print(f"Expiries: {sorted(df['expiry_dt'].unique())}")
    print(f"Strike range: {df['strike'].min():.0f} - {df['strike'].max():.0f}")

    # Clean data
    df_clean = clean_options_data(df_raw)
    print(f"\nAfter cleaning: {len(df_clean)} options")
    print(f"Expiries retained: {df_clean['expiry_dt'].nunique()}")

    # Summary by expiry
    print("\n--- Summary by Expiry ---")
    for expiry, group in df_clean.groupby("expiry_dt"):
        T = group["T"].iloc[0]
        fwd = group["forward"].iloc[0]
        n_calls = (group["option_type"] == "C").sum()
        n_puts = (group["option_type"] == "P").sum()
        iv_range = group["mark_iv_decimal"]
        print(
            f"  {expiry} | T={T:.4f}y ({T*365.25:.0f}d) | F={fwd:.0f} | "
            f"{n_calls}C/{n_puts}P | IV: [{iv_range.min():.1%}, {iv_range.max():.1%}]"
        )

    # Plot: IV by strike for nearest expiry
    nearest_exp = df_clean["expiry_dt"].min()
    slice_df = df_clean[df_clean["expiry_dt"] == nearest_exp]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # IV smile
    ax = axes[0, 0]
    calls = slice_df[slice_df["option_type"] == "C"]
    puts = slice_df[slice_df["option_type"] == "P"]
    ax.scatter(calls["strike"], calls["mark_iv_decimal"] * 100, label="Calls", s=20)
    ax.scatter(puts["strike"], puts["mark_iv_decimal"] * 100, label="Puts", s=20)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Mark IV (%)")
    ax.set_title(f"IV Smile — {nearest_exp}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bid-ask spread
    ax = axes[0, 1]
    spread = (slice_df["ask_price_usd"] - slice_df["bid_price_usd"]) / slice_df["mid_usd"]
    ax.scatter(slice_df["strike"], spread * 100, s=20)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Spread (% of mid)")
    ax.set_title("Bid-Ask Spread by Strike")
    ax.grid(True, alpha=0.3)

    # Mid price by strike
    ax = axes[1, 0]
    ax.scatter(slice_df["strike"], slice_df["mid_usd"], s=20)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Mid Price (USD)")
    ax.set_title("Option Mid Price by Strike")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Number of options by expiry
    ax = axes[1, 1]
    counts = df_clean.groupby("expiry_dt").size()
    ax.bar(range(len(counts)), counts.values)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(d.date()) for d in counts.index], rotation=45, ha="right")
    ax.set_ylabel("Number of Options")
    ax.set_title("Options Count by Expiry")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "reports/exploration.png"
    Path("reports").mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved exploration plots: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
