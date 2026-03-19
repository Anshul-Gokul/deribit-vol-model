"""
CLI entrypoints for the options-implied probability engine.

Commands:
- fetch-data: Fetch Deribit options data
- build-surface: Fit IV surface from snapshot
- query-prob: Query implied probability for a contract
- backtest: Run backtest simulation
- make-report: Generate analysis report with plots
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="deribit-prob", help="Options-implied probability engine")
console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def fetch_data(
    asset: str = typer.Option("BTC", help="Asset: BTC or ETH"),
    data_dir: str = typer.Option("data", help="Data directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fetch current options chain data from Deribit."""
    _setup_logging(verbose)
    from src.data.fetch_deribit import run_fetch

    console.print(f"[bold]Fetching {asset} options data from Deribit...[/bold]")
    try:
        path = run_fetch(asset, data_dir)
        console.print(f"[green]Saved snapshot: {path}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def build_surface(
    asset: str = typer.Option("BTC", help="Asset: BTC or ETH"),
    asof: str = typer.Option("", help="As-of timestamp (ISO format). Default: now"),
    data_dir: str = typer.Option("data", help="Data directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build IV surface from latest snapshot."""
    _setup_logging(verbose)
    from src.data.fetch_deribit import get_latest_snapshot, load_snapshot
    from src.cleaning.options_cleaner import clean_options_data
    from src.surface.iv_surface import IVSurface

    snap = get_latest_snapshot(asset, "options", data_dir)
    if snap is None:
        console.print(f"[red]No snapshot found for {asset}. Run fetch-data first.[/red]")
        raise typer.Exit(1)

    console.print(f"Using snapshot: {snap}")

    as_of = datetime.fromisoformat(asof) if asof else datetime.now(timezone.utc)
    df_raw = load_snapshot(snap)
    df_clean = clean_options_data(df_raw, as_of=as_of)

    if df_clean.empty:
        console.print("[red]No valid options after cleaning.[/red]")
        raise typer.Exit(1)

    surface = IVSurface()
    surface.fit(df_clean)
    console.print(surface.summary())


@app.command()
def query_prob(
    asset: str = typer.Option("BTC", help="Asset: BTC or ETH"),
    contract_type: str = typer.Option("above", help="Contract type: above, below, between, outside"),
    strike: float = typer.Option(None, help="Strike for above/below"),
    lower: float = typer.Option(None, help="Lower bound for between/outside"),
    upper: float = typer.Option(None, help="Upper bound for between/outside"),
    settlement: str = typer.Option("", help="Settlement timestamp (ISO format)"),
    data_dir: str = typer.Option("data", help="Data directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Query options-implied probability for a prediction market contract."""
    _setup_logging(verbose)
    from src.data.fetch_deribit import get_latest_snapshot, load_snapshot
    from src.cleaning.options_cleaner import clean_options_data
    from src.surface.iv_surface import IVSurface
    from src.distribution.risk_neutral_density import extract_density
    from src.pricing.contracts import price_contract
    from src.utils.time_utils import find_nearest_expiry

    now = datetime.now(timezone.utc)
    snap = get_latest_snapshot(asset, "options", data_dir)
    if snap is None:
        console.print(f"[red]No snapshot found for {asset}. Run fetch-data first.[/red]")
        raise typer.Exit(1)

    df_raw = load_snapshot(snap)
    df_clean = clean_options_data(df_raw, as_of=now)

    if df_clean.empty:
        console.print("[red]No valid options after cleaning.[/red]")
        raise typer.Exit(1)

    surface = IVSurface()
    surface.fit(df_clean)

    # Determine target time
    if settlement:
        target = datetime.fromisoformat(settlement.replace("Z", "+00:00"))
    else:
        # Default: nearest available expiry
        from src.cleaning.options_cleaner import get_available_expiries
        expiries = get_available_expiries(df_clean)
        target = expiries[0] if expiries else now

    T_target = max((target - now).total_seconds() / (365.25 * 24 * 3600), 0.001)
    forward = surface._interpolate_forward(T_target)

    # Find nearest expiry for metadata
    expiries = [s.expiry for s in surface.slices.values()]
    nearest = find_nearest_expiry(target, expiries) if expiries else target

    # Extract density
    density = extract_density(surface, T_target, forward)

    # Price the contract
    pricing = price_contract(
        density,
        contract_type=contract_type,
        strike=strike,
        lower=lower,
        upper=upper,
        asset=asset,
        settlement=target.isoformat(),
        forecast_time=now.isoformat(),
        expiry_used=nearest.isoformat(),
    )

    # Display results
    console.print()
    console.print("[bold]Options-Implied Contract Pricing[/bold]")
    console.print("=" * 50)
    console.print(pricing.display())
    console.print()
    console.print("[dim]Note: These are risk-neutral (Q-measure) probabilities, not real-world forecasts.[/dim]")

    # Also output JSON
    console.print()
    console.print("[bold]JSON output:[/bold]")
    console.print(json.dumps(pricing.to_dict(), indent=2))


@app.command()
def backtest(
    asset: str = typer.Option("BTC", help="Asset: BTC or ETH"),
    start: str = typer.Option("", help="Start timestamp (ISO format)"),
    end: str = typer.Option("", help="End timestamp (ISO format)"),
    horizon: str = typer.Option("daily", help="Forecast horizon: 1h, 4h, daily, weekly"),
    data_dir: str = typer.Option("data", help="Data directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run backtest of options-implied probabilities."""
    _setup_logging(verbose)
    from src.backtest.backtest_runner import (
        simulate_backtest_from_single_snapshot,
    )

    console.print(f"[bold]Running backtest for {asset} ({horizon})...[/bold]")

    try:
        result = simulate_backtest_from_single_snapshot(asset, data_dir)
        console.print(result.summary())

        # Display strike ladder
        df = result.to_dataframe()
        if not df.empty:
            table = Table(title="Probability by Strike")
            table.add_column("Contract", style="cyan")
            table.add_column("Strike", justify="right")
            table.add_column("Prob(Above)", justify="right")
            table.add_column("Fair YES", justify="right")

            for _, row in df.iterrows():
                table.add_row(
                    row.get("contract", ""),
                    f"{row['strike']:.0f}",
                    f"{row['implied_prob']:.4f}",
                    f"{row['implied_prob']:.4f}",
                )
            console.print(table)

    except FileNotFoundError:
        console.print(f"[red]No data found for {asset}. Run fetch-data first.[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Backtest error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def make_report(
    asset: str = typer.Option("BTC", help="Asset: BTC or ETH"),
    data_dir: str = typer.Option("data", help="Data directory"),
    output_dir: str = typer.Option("reports", help="Report output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate analysis report with plots."""
    _setup_logging(verbose)
    from src.data.fetch_deribit import get_latest_snapshot, load_snapshot
    from src.cleaning.options_cleaner import clean_options_data
    from src.surface.iv_surface import IVSurface
    from src.distribution.risk_neutral_density import extract_density
    from src.utils.visualization import (
        plot_smile,
        plot_surface_3d,
        plot_density,
        plot_probability_by_strike,
    )

    now = datetime.now(timezone.utc)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    snap = get_latest_snapshot(asset, "options", data_dir)
    if snap is None:
        console.print(f"[red]No snapshot found for {asset}. Run fetch-data first.[/red]")
        raise typer.Exit(1)

    console.print(f"Using snapshot: {snap}")

    df_raw = load_snapshot(snap)
    df_clean = clean_options_data(df_raw, as_of=now)

    if df_clean.empty:
        console.print("[red]No valid options after cleaning.[/red]")
        raise typer.Exit(1)

    surface = IVSurface()
    surface.fit(df_clean)
    console.print(surface.summary())

    # Generate plots
    console.print("[bold]Generating plots...[/bold]")

    plot_smile(surface, save_path=out / f"{asset}_iv_smiles.png")
    console.print(f"  Saved: {out / f'{asset}_iv_smiles.png'}")

    plot_surface_3d(surface, save_path=out / f"{asset}_iv_surface_3d.png")
    console.print(f"  Saved: {out / f'{asset}_iv_surface_3d.png'}")

    # Density for each expiry
    for T in surface.expiry_times[:5]:  # Limit to first 5
        forward = surface._interpolate_forward(T)
        density = extract_density(surface, T, forward)

        dte = int(T * 365.25)
        plot_density(density, save_path=out / f"{asset}_density_{dte}d.png")
        plot_probability_by_strike(
            density, "above", save_path=out / f"{asset}_prob_above_{dte}d.png"
        )
        console.print(f"  Saved density/probability plots for T={T:.3f}y ({dte}d)")

    # Summary report
    report_path = out / f"{asset}_report.md"
    with open(report_path, "w") as f:
        f.write(f"# {asset} Options-Implied Probability Report\n\n")
        f.write(f"Generated: {now.isoformat()}\n\n")
        f.write("## IV Surface\n\n")
        f.write(f"```\n{surface.summary()}\n```\n\n")
        f.write("## Methodology\n\n")
        f.write("1. **Data**: Options chain fetched from Deribit public API\n")
        f.write("2. **Cleaning**: Filtered for valid bid/ask, OTM options, no-arb bounds\n")
        f.write("3. **Surface**: Cubic spline interpolation in log-moneyness space\n")
        f.write("4. **Density**: Breeden-Litzenberger (numerical 2nd derivative of call prices)\n")
        f.write("5. **Probabilities**: Integration of risk-neutral density over target ranges\n\n")
        f.write("## Important Caveat\n\n")
        f.write("All probabilities are **risk-neutral (Q-measure)**, not real-world forecasts.\n")
        f.write("They reflect option market pricing, which incorporates risk premia.\n\n")
        f.write("## Plots\n\n")
        f.write(f"- IV Smiles: `{asset}_iv_smiles.png`\n")
        f.write(f"- IV Surface 3D: `{asset}_iv_surface_3d.png`\n")
        f.write(f"- Density plots: `{asset}_density_*d.png`\n")
        f.write(f"- Probability plots: `{asset}_prob_above_*d.png`\n")

    console.print(f"\n[green]Report saved: {report_path}[/green]")


@app.command()
def spot_forecast(
    asset: str = typer.Option("BTC", help="Asset: BTC or ETH"),
    target: str = typer.Option("", help="Target timestamp (ISO format). Default: multi-horizon"),
    no_tilt: bool = typer.Option(False, help="Disable directional tilt (pure forward only)"),
    data_dir: str = typer.Option("data", help="Data directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Forecast future spot price using derivatives-implied model."""
    _setup_logging(verbose)
    from src.models.spot_forecast import build_forecast, forecast_at_timestamp

    if target:
        # Single timestamp forecast
        target_dt = datetime.fromisoformat(target.replace("Z", "+00:00"))
        console.print(f"[bold]Forecasting {asset} at {target_dt.isoformat()}...[/bold]")

        h = forecast_at_timestamp(asset, target_dt, data_dir, apply_tilt=not no_tilt)
        console.print()
        console.print(f"[bold]Target:[/bold] {target_dt.strftime('%Y-%m-%d %H:%M UTC')}")
        console.print(f"[bold]Forward:[/bold]  {h.forward:,.2f}")
        console.print(f"[bold]Forecast:[/bold] {h.point_forecast:,.2f}  (tilt: {h.tilt_pct:+.3f}%)")
        console.print(f"[bold]IV:[/bold]       {h.implied_vol:.1%}")
        console.print(f"[bold]±1 SD:[/bold]    {h.expected_move_pct:+.1f}%  (${h.expected_move_1sd:,.0f})")
        console.print()
        console.print("[bold]Confidence Intervals (risk-neutral):[/bold]")
        console.print(f"   5%: {h.q05:>10,.0f}")
        console.print(f"  25%: {h.q25:>10,.0f}")
        console.print(f"  50%: {h.q50:>10,.0f}  (median)")
        console.print(f"  75%: {h.q75:>10,.0f}")
        console.print(f"  95%: {h.q95:>10,.0f}")
    else:
        # Multi-horizon forecast
        console.print(f"[bold]Building {asset} spot forecast from derivatives...[/bold]")
        fc = build_forecast(asset, data_dir, apply_tilt=not no_tilt)
        console.print(fc.summary())
        if fc.signals:
            console.print()
            console.print(fc.signals.summary())


@app.command()
def forward_curve(
    asset: str = typer.Option("BTC", help="Asset: BTC or ETH"),
    data_dir: str = typer.Option("data", help="Data directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Display the futures-implied forward curve."""
    _setup_logging(verbose)
    from src.models.forward_curve import build_forward_curve_from_futures
    from src.data.fetch_deribit import get_latest_snapshot, load_snapshot

    now = datetime.now(timezone.utc)

    fut_snap = get_latest_snapshot(asset, "futures", data_dir)
    if fut_snap is None:
        console.print(f"[red]No futures snapshot for {asset}. Run fetch-data first.[/red]")
        raise typer.Exit(1)

    opt_snap = get_latest_snapshot(asset, "options", data_dir)
    df_fut = load_snapshot(fut_snap)

    # Get spot from options or futures
    if opt_snap:
        df_opt = load_snapshot(opt_snap)
        spot = float(df_opt["index_price"].iloc[0])
    else:
        spot = float(df_fut["index_price"].iloc[0]) if "index_price" in df_fut.columns else 0

    curve = build_forward_curve_from_futures(df_fut, spot, now)

    console.print(f"[bold]{asset} Forward Curve[/bold]")
    console.print(curve.summary())
    console.print()

    # Full table
    table = Table(title="Forward Curve Details")
    table.add_column("Days", justify="right")
    table.add_column("Forward", justify="right", style="cyan")
    table.add_column("Basis %", justify="right")
    table.add_column("Ann. Carry %", justify="right")
    table.add_column("Impl. Rate %", justify="right")

    df = curve.curve_table()
    for _, row in df.iterrows():
        table.add_row(
            str(row["days"]),
            f"{row['forward']:,.2f}",
            f"{row['basis_pct']:+.3f}",
            f"{row['annualized_carry_pct']:+.2f}",
            f"{row['implied_rate_pct']:+.2f}",
        )
    console.print(table)


@app.command()
def signals(
    asset: str = typer.Option("BTC", help="Asset: BTC or ETH"),
    data_dir: str = typer.Option("data", help="Data directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Extract derivatives-implied directional and volatility signals."""
    _setup_logging(verbose)
    from src.models.spot_forecast import build_forecast

    console.print(f"[bold]Extracting {asset} derivatives signals...[/bold]")
    fc = build_forecast(asset, data_dir, apply_tilt=False)

    if fc.signals:
        console.print(fc.signals.summary())
    else:
        console.print("[red]Could not extract signals.[/red]")


if __name__ == "__main__":
    app()
