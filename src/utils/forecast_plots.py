"""
Forecast visualization suite for the spot price model.

Produces publication-quality plots:

1. Fan Chart        — forward + confidence bands across horizons
2. Forward Curve    — term structure of forwards with basis overlay
3. Signals Dashboard — multi-panel view of all derivatives signals
4. Density Evolution — how the implied distribution widens over time
5. Scenario Cone    — Monte Carlo paths sampled from implied vol
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

from src.models.spot_forecast import SpotForecast, HorizonForecast
from src.models.forward_curve import ForwardCurve
from src.models.signals import DerivativesSignals

# ── Style constants ──────────────────────────────────────────────────
_BG       = "#0d1117"
_FG       = "#c9d1d9"
_GRID     = "#21262d"
_ACCENT   = "#58a6ff"
_GREEN    = "#3fb950"
_RED      = "#f85149"
_ORANGE   = "#d29922"
_PURPLE   = "#bc8cff"
_CYAN     = "#39d2c0"

_BAND_COLORS = [
    ("#58a6ff", 0.08),   # 5-95 band (widest)
    ("#58a6ff", 0.16),   # 25-75 band
]


def _apply_dark_theme(fig, axes):
    """Apply consistent dark theme to figure and axes."""
    fig.patch.set_facecolor(_BG)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_FG, labelsize=9)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.title.set_color(_FG)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.grid(True, color=_GRID, alpha=0.5, linewidth=0.5)


def _format_price_axis(ax, prefix: str = "$"):
    """Format y-axis with dollar signs and comma separators."""
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{prefix}{x:,.0f}")
    )


def _add_watermark(fig, text: str = "deribit-vol-model"):
    """Add subtle watermark."""
    fig.text(
        0.99, 0.01, text,
        ha="right", va="bottom",
        fontsize=7, color=_FG, alpha=0.3,
        fontfamily="monospace",
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. FAN CHART
# ═══════════════════════════════════════════════════════════════════════

def plot_fan_chart(
    forecast: SpotForecast,
    save_path: str | Path | None = None,
    show_tilt: bool = True,
    figsize: tuple = (14, 7),
) -> plt.Figure:
    """
    Fan chart: point forecast with widening confidence bands.

    The x-axis is time (absolute dates), y-axis is price.
    Bands show 5-95% and 25-75% quantile ranges from the
    risk-neutral density at each horizon.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_theme(fig, ax)

    horizons = forecast.horizons
    if not horizons:
        return fig

    # Build time axis
    times = [forecast.as_of] + [h.target_time for h in horizons]
    spot = forecast.spot

    # Build price arrays (prepend spot for T=0)
    forwards  = [spot] + [h.forward for h in horizons]
    points    = [spot] + [h.point_forecast for h in horizons]
    q05s      = [spot] + [h.q05 for h in horizons]
    q25s      = [spot] + [h.q25 for h in horizons]
    q50s      = [spot] + [h.q50 for h in horizons]
    q75s      = [spot] + [h.q75 for h in horizons]
    q95s      = [spot] + [h.q95 for h in horizons]

    # 5-95 band
    ax.fill_between(
        times, q05s, q95s,
        color=_ACCENT, alpha=0.08, label="5th–95th percentile",
    )
    # 25-75 band
    ax.fill_between(
        times, q25s, q75s,
        color=_ACCENT, alpha=0.18, label="25th–75th percentile",
    )

    # Median
    ax.plot(times, q50s, color=_FG, linewidth=1, linestyle=":", alpha=0.5, label="Median")

    # Forward curve
    ax.plot(times, forwards, color=_ORANGE, linewidth=2, label="Forward (risk-neutral)")

    # Tilted forecast
    if show_tilt:
        ax.plot(times, points, color=_CYAN, linewidth=2.5, label="Forecast (tilted)")

    # Spot marker
    ax.plot(
        forecast.as_of, spot,
        "o", color=_GREEN, markersize=10, zorder=5,
        markeredgecolor="white", markeredgewidth=1.5,
    )
    ax.annotate(
        f"  Spot: ${spot:,.0f}",
        xy=(forecast.as_of, spot),
        fontsize=10, fontweight="bold", color=_GREEN,
        va="center",
    )

    # Annotate final horizon
    last = horizons[-1]
    ax.annotate(
        f"${last.point_forecast:,.0f}\n±${last.expected_move_1sd:,.0f}",
        xy=(last.target_time, last.point_forecast),
        xytext=(15, 0), textcoords="offset points",
        fontsize=9, color=_CYAN, va="center",
        arrowprops=dict(arrowstyle="-", color=_CYAN, alpha=0.4),
    )

    # Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    _format_price_axis(ax)

    ax.set_title(
        f"{forecast.asset} Spot Price Forecast — "
        f"{forecast.as_of.strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_ylabel("Price (USD)")

    ax.legend(
        loc="upper left", fontsize=9,
        facecolor=_BG, edgecolor=_GRID, labelcolor=_FG,
    )

    # Direction badge
    if forecast.signals:
        s = forecast.signals
        direction = "BULLISH" if s.directional_score > 0.1 else \
                    "BEARISH" if s.directional_score < -0.1 else "NEUTRAL"
        badge_color = _GREEN if direction == "BULLISH" else _RED if direction == "BEARISH" else _ORANGE
        ax.text(
            0.98, 0.97,
            f" {direction}  score={s.directional_score:+.2f} ",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=11, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_color, alpha=0.85),
        )

    _add_watermark(fig)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 2. FORWARD CURVE
# ═══════════════════════════════════════════════════════════════════════

def plot_forward_curve(
    curve: ForwardCurve,
    asset: str = "BTC",
    save_path: str | Path | None = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Two-panel plot: forward prices and annualized basis/carry.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1], sharex=True)
    _apply_dark_theme(fig, [ax1, ax2])

    if not curve.points:
        return fig

    pts = sorted(curve.points, key=lambda p: p.T)
    days = [p.T * 365.25 for p in pts]
    fwds = [p.forward for p in pts]
    carries = [p.annualized_basis for p in pts]
    sources = [p.source for p in pts]

    # Interpolated curve
    T_fine = np.linspace(0, max(p.T for p in pts), 200)
    F_fine = [curve.forward(t) for t in T_fine]
    days_fine = T_fine * 365.25

    # Top: forward prices
    ax1.plot(days_fine, F_fine, color=_ACCENT, linewidth=2, label="Interpolated")

    # Scatter by source
    for src, marker, color in [("futures", "D", _CYAN), ("put_call_parity", "s", _PURPLE)]:
        src_pts = [(d, f) for d, f, s in zip(days, fwds, sources) if s == src]
        if src_pts:
            ds, fs = zip(*src_pts)
            ax1.scatter(ds, fs, marker=marker, s=60, color=color, zorder=5,
                        edgecolors="white", linewidth=0.8, label=src.replace("_", " ").title())

    # Spot line
    ax1.axhline(curve.spot, color=_GREEN, linestyle="--", alpha=0.6, linewidth=1)
    ax1.text(days_fine[-1] * 0.02, curve.spot, f"  Spot: ${curve.spot:,.0f}",
             color=_GREEN, fontsize=9, va="bottom")

    ax1.set_ylabel("Forward Price (USD)")
    _format_price_axis(ax1)
    ax1.set_title(f"{asset} Forward Curve", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(facecolor=_BG, edgecolor=_GRID, labelcolor=_FG, fontsize=9)

    # Bottom: annualized carry
    carry_fine = [curve.annualized_carry(t) * 100 for t in T_fine[1:]]
    ax2.plot(days_fine[1:], carry_fine, color=_ORANGE, linewidth=2)
    ax2.axhline(0, color=_FG, linewidth=0.5, alpha=0.3)

    # Color fill
    carry_arr = np.array(carry_fine)
    ax2.fill_between(
        days_fine[1:], carry_arr, 0,
        where=carry_arr >= 0, color=_GREEN, alpha=0.15,
    )
    ax2.fill_between(
        days_fine[1:], carry_arr, 0,
        where=carry_arr < 0, color=_RED, alpha=0.15,
    )

    ax2.set_xlabel("Days to Expiry")
    ax2.set_ylabel("Ann. Carry (%)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))

    _add_watermark(fig)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 3. SIGNALS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

def plot_signals_dashboard(
    signals: DerivativesSignals,
    asset: str = "BTC",
    save_path: str | Path | None = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Multi-panel dashboard showing all derivatives signals.
    """
    fig = plt.figure(figsize=figsize)
    _apply_dark_theme(fig, [])
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Directional gauge ────────────────────────────────
    ax_gauge = fig.add_subplot(gs[0, 0])
    _apply_dark_theme(fig, ax_gauge)
    _draw_gauge(ax_gauge, signals.directional_score, "Directional Score")

    # ── Panel 2: Forward basis bars ──────────────────────────────
    ax_basis = fig.add_subplot(gs[0, 1])
    _apply_dark_theme(fig, ax_basis)
    labels = ["1d", "7d", "30d"]
    values = [signals.basis_1d_pct, signals.basis_7d_pct, signals.basis_30d_pct]
    colors = [_GREEN if v >= 0 else _RED for v in values]
    ax_basis.barh(labels, values, color=colors, height=0.5, edgecolor="white", linewidth=0.5)
    ax_basis.axvline(0, color=_FG, linewidth=0.5)
    ax_basis.set_title("Basis (%)", fontsize=11, fontweight="bold")
    ax_basis.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.3f}%"))

    # ── Panel 3: ATM IV term structure ───────────────────────────
    ax_iv = fig.add_subplot(gs[0, 2])
    _apply_dark_theme(fig, ax_iv)
    iv_labels = ["1d", "7d", "30d"]
    iv_values = [signals.atm_iv_1d * 100, signals.atm_iv_7d * 100, signals.atm_iv_30d * 100]
    ax_iv.plot(iv_labels, iv_values, "o-", color=_ACCENT, linewidth=2, markersize=8)
    ax_iv.set_title("ATM IV Term Structure", fontsize=11, fontweight="bold")
    ax_iv.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # ── Panel 4: Risk reversal ───────────────────────────────────
    ax_rr = fig.add_subplot(gs[1, 0])
    _apply_dark_theme(fig, ax_rr)
    rr_labels = ["RR25 7d", "RR25 30d"]
    rr_values = [signals.rr25_7d * 100, signals.rr25_30d * 100]
    rr_colors = [_GREEN if v > 0 else _RED for v in rr_values]
    bars = ax_rr.barh(rr_labels, rr_values, color=rr_colors, height=0.4,
                       edgecolor="white", linewidth=0.5)
    ax_rr.axvline(0, color=_FG, linewidth=0.5)
    ax_rr.set_title("25Δ Risk Reversal", fontsize=11, fontweight="bold")
    ax_rr.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    # Annotation
    for bar, val in zip(bars, rr_values):
        label = "calls bid" if val > 0 else "puts bid"
        ax_rr.text(
            bar.get_width() + (0.2 if val >= 0 else -0.2), bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left" if val >= 0 else "right",
            fontsize=8, color=_FG, alpha=0.7,
        )

    # ── Panel 5: Butterfly ───────────────────────────────────────
    ax_bf = fig.add_subplot(gs[1, 1])
    _apply_dark_theme(fig, ax_bf)
    bf_val = signals.butterfly_25_7d * 100
    ax_bf.barh(["BF25 7d"], [bf_val], color=_PURPLE, height=0.4,
               edgecolor="white", linewidth=0.5)
    ax_bf.set_title("25Δ Butterfly (tail risk)", fontsize=11, fontweight="bold")
    ax_bf.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # ── Panel 6: Put/Call ratios ─────────────────────────────────
    ax_pc = fig.add_subplot(gs[1, 2])
    _apply_dark_theme(fig, ax_pc)
    pc_labels = ["OI Ratio", "Volume Ratio"]
    pc_values = [signals.put_call_oi_ratio, signals.put_call_volume_ratio]
    pc_colors = [_RED if v > 1.0 else _GREEN for v in pc_values]
    ax_pc.barh(pc_labels, pc_values, color=pc_colors, height=0.4,
               edgecolor="white", linewidth=0.5)
    ax_pc.axvline(1.0, color=_FG, linewidth=1, linestyle="--", alpha=0.5)
    ax_pc.set_title("Put/Call Ratios", fontsize=11, fontweight="bold")
    ax_pc.text(1.0, -0.35, "neutral", ha="center", fontsize=8, color=_FG, alpha=0.5,
               transform=ax_pc.get_xaxis_transform())

    # ── Panel 7-9: Summary text ──────────────────────────────────
    ax_txt = fig.add_subplot(gs[2, :])
    _apply_dark_theme(fig, ax_txt)
    ax_txt.axis("off")

    direction = "BULLISH" if signals.directional_score > 0.1 else \
                "BEARISH" if signals.directional_score < -0.1 else "NEUTRAL"
    badge_color = _GREEN if direction == "BULLISH" else _RED if direction == "BEARISH" else _ORANGE

    summary_text = (
        f"{asset} Derivatives Signal Summary\n\n"
        f"Spot: ${signals.spot:,.2f}    "
        f"Fwd 7d: ${signals.forward_7d:,.2f}    "
        f"Fwd 30d: ${signals.forward_30d:,.2f}\n"
        f"ATM IV 7d: {signals.atm_iv_7d:.1%}    "
        f"Term Slope: {signals.iv_term_slope:+.1%}    "
        f"Carry: {signals.annualized_carry_7d:+.2f}%/yr\n"
        f"Direction: {direction} (score={signals.directional_score:+.3f}, "
        f"confidence={signals.confidence:.0%})"
    )
    ax_txt.text(
        0.5, 0.5, summary_text,
        transform=ax_txt.transAxes,
        ha="center", va="center",
        fontsize=11, fontfamily="monospace", color=_FG,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#161b22", edgecolor=_GRID),
    )

    fig.suptitle(
        f"{asset} Derivatives Signals Dashboard",
        fontsize=15, fontweight="bold", color=_FG, y=0.98,
    )

    _add_watermark(fig)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)
    return fig


def _draw_gauge(ax, score: float, title: str):
    """Draw a simple directional gauge (-1 to +1)."""
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Draw arc
    theta = np.linspace(np.pi, 0, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, color=_GRID, linewidth=6, solid_capstyle="round")

    # Color segments
    for start, end, color in [(np.pi, np.pi * 2 / 3, _RED),
                               (np.pi * 2 / 3, np.pi / 3, _ORANGE),
                               (np.pi / 3, 0, _GREEN)]:
        th = np.linspace(start, end, 30)
        ax.plot(np.cos(th), np.sin(th), color=color, linewidth=6, solid_capstyle="round", alpha=0.4)

    # Needle
    needle_angle = np.pi * (1 - (score + 1) / 2)  # map [-1,1] to [pi, 0]
    nx = 0.85 * np.cos(needle_angle)
    ny = 0.85 * np.sin(needle_angle)
    ax.annotate(
        "", xy=(nx, ny), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2),
    )
    ax.plot(0, 0, "o", color="white", markersize=6, zorder=5)

    # Labels
    ax.text(-1.1, -0.15, "BEAR", ha="center", fontsize=8, color=_RED)
    ax.text(0, 1.15, "NEUTRAL", ha="center", fontsize=8, color=_ORANGE)
    ax.text(1.1, -0.15, "BULL", ha="center", fontsize=8, color=_GREEN)
    ax.text(0, -0.3, f"{score:+.2f}", ha="center", fontsize=14,
            fontweight="bold", color="white")


# ═══════════════════════════════════════════════════════════════════════
# 4. DENSITY EVOLUTION
# ═══════════════════════════════════════════════════════════════════════

def plot_density_evolution(
    forecast: SpotForecast,
    save_path: str | Path | None = None,
    figsize: tuple = (14, 7),
) -> plt.Figure:
    """
    Show how the risk-neutral density widens at longer horizons.

    Overlays the PDF for each forecast horizon on a single plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_theme(fig, ax)

    cmap = plt.cm.plasma
    n_horizons = len(forecast.horizons)

    for i, h in enumerate(forecast.horizons):
        if h.density is None:
            continue
        color = cmap(i / max(n_horizons - 1, 1))

        # Smooth the PDF for display with Gaussian filter
        from scipy.ndimage import gaussian_filter1d
        sigma_pts = max(3, int(len(h.density.pdf) * 0.02 * (1 + h.T * 5)))
        pdf_smooth = gaussian_filter1d(h.density.pdf, sigma=sigma_pts)
        pdf_smooth = np.maximum(pdf_smooth, 0)

        # Normalize for visual clarity (peak = 1)
        peak = pdf_smooth.max()
        pdf_norm = pdf_smooth / peak if peak > 0 else pdf_smooth

        ax.fill_between(
            h.density.strikes, pdf_norm,
            color=color, alpha=0.08,
        )
        ax.plot(
            h.density.strikes, pdf_norm,
            color=color, linewidth=1.5, label=h.label,
        )

    # Spot line
    ax.axvline(forecast.spot, color=_GREEN, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(forecast.spot, ax.get_ylim()[1] * 0.95, f"  Spot ${forecast.spot:,.0f}",
            color=_GREEN, fontsize=9, va="top")

    ax.set_xlabel("Price at Horizon")
    ax.set_ylabel("Normalized Density")
    ax.set_title(
        f"{forecast.asset} Risk-Neutral Density by Horizon",
        fontsize=13, fontweight="bold", pad=12,
    )

    # Focus on the interesting range
    ax.set_xlim(forecast.spot * 0.7, forecast.spot * 1.3)

    ax.legend(
        title="Horizon", loc="upper right",
        facecolor=_BG, edgecolor=_GRID, labelcolor=_FG, fontsize=9,
    )

    _add_watermark(fig)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 5. SCENARIO CONE (Monte Carlo from IV)
# ═══════════════════════════════════════════════════════════════════════

def plot_scenario_cone(
    forecast: SpotForecast,
    n_paths: int = 200,
    n_steps: int = 100,
    save_path: str | Path | None = None,
    figsize: tuple = (14, 7),
) -> plt.Figure:
    """
    Monte Carlo scenario cone sampled from the implied vol surface.

    Simulates GBM paths using the ATM IV at each horizon as the local vol,
    producing a spaghetti plot with percentile envelopes.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_theme(fig, ax)

    if not forecast.horizons:
        return fig

    # Build time grid
    T_max = forecast.horizons[-1].T
    dt = T_max / n_steps
    times_years = np.linspace(0, T_max, n_steps + 1)

    # Map times to absolute dates
    times_dates = [forecast.as_of + timedelta(days=t * 365.25) for t in times_years]

    # Get IV at each time step (use nearest horizon's IV)
    ivs = np.array([_iv_at_time(forecast, t) for t in times_years])

    # Simulate paths
    rng = np.random.default_rng(42)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = forecast.spot

    for i in range(n_steps):
        sigma = ivs[i]
        z = rng.standard_normal(n_paths)
        # GBM step: S_{t+1} = S_t * exp((r - 0.5*σ²)*dt + σ*sqrt(dt)*Z)
        paths[:, i + 1] = paths[:, i] * np.exp(-0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z)

    # Percentiles
    p05 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    # Spaghetti paths (subsample)
    n_show = min(n_paths, 50)
    for j in range(n_show):
        ax.plot(times_dates, paths[j], color=_ACCENT, alpha=0.03, linewidth=0.5)

    # Bands
    ax.fill_between(times_dates, p05, p95, color=_ACCENT, alpha=0.08)
    ax.fill_between(times_dates, p25, p75, color=_ACCENT, alpha=0.18)

    # Median
    ax.plot(times_dates, p50, color=_ORANGE, linewidth=2, label="Median path")

    # Forward overlay
    fwd_prices = [forecast.spot] + [h.forward for h in forecast.horizons]
    fwd_times = [forecast.as_of] + [h.target_time for h in forecast.horizons]
    ax.plot(fwd_times, fwd_prices, color=_CYAN, linewidth=2, linestyle="--", label="Forward")

    # Spot
    ax.plot(forecast.as_of, forecast.spot, "o", color=_GREEN, markersize=10,
            markeredgecolor="white", markeredgewidth=1.5, zorder=5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    _format_price_axis(ax)

    ax.set_title(
        f"{forecast.asset} Scenario Cone — {n_paths} MC Paths from Implied Vol",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_ylabel("Price (USD)")
    ax.legend(facecolor=_BG, edgecolor=_GRID, labelcolor=_FG, fontsize=9)

    _add_watermark(fig)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)
    return fig


def _iv_at_time(forecast: SpotForecast, t: float) -> float:
    """Get ATM IV at time t by interpolating between horizons."""
    if not forecast.horizons:
        return 0.5
    # Find bracketing horizons
    for i, h in enumerate(forecast.horizons):
        if h.T >= t:
            if i == 0:
                return h.implied_vol
            prev = forecast.horizons[i - 1]
            alpha = (t - prev.T) / (h.T - prev.T) if h.T != prev.T else 0
            return prev.implied_vol + alpha * (h.implied_vol - prev.implied_vol)
    return forecast.horizons[-1].implied_vol


# ═══════════════════════════════════════════════════════════════════════
# 6. COMBINED REPORT
# ═══════════════════════════════════════════════════════════════════════

def generate_all_forecast_plots(
    forecast: SpotForecast,
    output_dir: str | Path = "reports",
) -> list[Path]:
    """Generate all forecast visualizations and save to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    asset = forecast.asset

    paths = []

    p = out / f"{asset}_fan_chart.png"
    plot_fan_chart(forecast, save_path=p)
    paths.append(p)

    if forecast.forward_curve:
        p = out / f"{asset}_forward_curve.png"
        plot_forward_curve(forecast.forward_curve, asset, save_path=p)
        paths.append(p)

    if forecast.signals:
        p = out / f"{asset}_signals_dashboard.png"
        plot_signals_dashboard(forecast.signals, asset, save_path=p)
        paths.append(p)

    p = out / f"{asset}_density_evolution.png"
    plot_density_evolution(forecast, save_path=p)
    paths.append(p)

    p = out / f"{asset}_scenario_cone.png"
    plot_scenario_cone(forecast, save_path=p)
    paths.append(p)

    return paths
