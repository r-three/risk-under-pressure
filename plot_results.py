#!/usr/bin/env python3
"""
plot_results.py — Phase 3: Plot risk-pressure curves from metrics CSV.

Generates:
  1. One figure per attack showing risk curves for each model with CI shading.
  2. One combined figure with all (model, attack) pairs overlaid.

Usage:
    python plot_results.py \\
        --metrics-csv outputs/pressure_sensitivity/metrics.csv \\
        --output-dir outputs/pressure_sensitivity/plots

    python plot_results.py \\
        --metrics-csv /path/to/metrics.csv \\
        --output-dir /path/to/plots \\
        --format pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Aesthetics
# --------------------------------------------------------------------------- #

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

ATTACK_MARKERS = {"gcg": "o", "pair": "s", "jailbroken": "^"}
ATTACK_LINESTYLES = {"gcg": "-", "pair": "--", "jailbroken": ":"}

ATTACK_DISPLAY = {"gcg": "GCG", "pair": "PAIR", "jailbroken": "Jailbroken"}


def _attack_label(attack_id: str) -> str:
    return ATTACK_DISPLAY.get(attack_id, attack_id)


def _model_color(model_ids: list[str]) -> dict[str, str]:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(model_ids)}


# --------------------------------------------------------------------------- #
# Core plotting helpers
# --------------------------------------------------------------------------- #

def _plot_risk_curve(
    ax: plt.Axes,
    df_pair: pd.DataFrame,
    color: str,
    label: str,
    marker: str = "o",
    linestyle: str = "-",
) -> None:
    """Draw a single risk-pressure curve with vertical CI error bars onto ax."""
    df_pair = df_pair.sort_values("lambda")
    lam = df_pair["lambda"].values
    risk = df_pair["risk"].values
    lo = df_pair["risk_lower"].values
    hi = df_pair["risk_upper"].values

    yerr_lo = risk - lo   # downward extent
    yerr_hi = hi - risk   # upward extent

    ax.errorbar(
        lam, risk,
        yerr=[yerr_lo, yerr_hi],
        color=color,
        label=label,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.8,
        markersize=5,
        capsize=4,
        capthick=1.4,
        elinewidth=1.2,
        zorder=3,
    )


def _style_axes(ax: plt.Axes, title: str, lambda_max: float) -> None:
    ax.set_xlabel("Pressure level λ", fontsize=11)
    ax.set_ylabel("Risk R̂(M, λ)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(left=0, right=lambda_max + 0.5)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


# --------------------------------------------------------------------------- #
# Per-attack figures
# --------------------------------------------------------------------------- #

def plot_per_attack(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """One figure per attack: all models overlaid."""
    attacks = sorted(df["attack_id"].unique())
    models = sorted(df["model_id"].unique())
    color_map = _model_color(models)
    lambda_max = df["lambda"].max()

    for attack in attacks:
        df_attack = df[df["attack_id"] == attack]
        fig, ax = plt.subplots(figsize=(6, 4.5))

        for model in models:
            df_pair = df_attack[df_attack["model_id"] == model]
            if df_pair.empty:
                continue
            _plot_risk_curve(
                ax, df_pair,
                color=color_map[model],
                label=model,
                marker=ATTACK_MARKERS.get(attack, "o"),
                linestyle="-",
            )

        _style_axes(ax, f"Risk-Pressure Curves — {_attack_label(attack)}", lambda_max)
        ax.legend(title="Model", fontsize=9, title_fontsize=9,
                  loc="lower right", framealpha=0.9,
                  borderpad=0.8, labelspacing=0.4)

        fig.tight_layout()
        out_path = output_dir / f"risk_curves_{attack}.{fmt}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# --------------------------------------------------------------------------- #
# All-in-one figure
# --------------------------------------------------------------------------- #

def plot_combined(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """Single figure: all (model, attack) pairs with colour=model, style=attack."""
    models = sorted(df["model_id"].unique())
    attacks = sorted(df["attack_id"].unique())
    color_map = _model_color(models)
    lambda_max = df["lambda"].max()

    fig, ax = plt.subplots(figsize=(10, 5))

    for model in models:
        for attack in attacks:
            df_pair = df[(df["model_id"] == model) & (df["attack_id"] == attack)]
            if df_pair.empty:
                continue
            _plot_risk_curve(
                ax, df_pair,
                color=color_map[model],
                label=f"{model} / {_attack_label(attack)}",
                marker=ATTACK_MARKERS.get(attack, "o"),
                linestyle=ATTACK_LINESTYLES.get(attack, "-"),
            )

    _style_axes(ax, "Risk-Pressure Curves — All Models & Attacks", lambda_max)

    # Split legend: models (colour patches) + attacks (line styles)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    model_handles = [Patch(color=color_map[m], label=m) for m in models]
    attack_handles = [
        Line2D([0], [0], color="gray",
               linestyle=ATTACK_LINESTYLES.get(a, "-"),
               marker=ATTACK_MARKERS.get(a, "o"),
               label=_attack_label(a))
        for a in attacks
    ]

    leg1 = ax.legend(handles=model_handles, title="Model",
                     fontsize=8, title_fontsize=9,
                     loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9,
                     borderpad=0.8, labelspacing=0.4)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=attack_handles, title="Attack",
                     fontsize=8, title_fontsize=9,
                     loc="upper left", bbox_to_anchor=(1.01, 0.45), framealpha=0.9,
                     borderpad=0.8, labelspacing=0.4)

    fig.subplots_adjust(right=0.68)
    out_path = output_dir / f"risk_curves_combined.{fmt}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    plt.close(fig)
    print(f"  Saved: {out_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot pRisk-Pressure curves from metrics CSV")
    p.add_argument("--metrics-csv", required=True,
                   help="Path to metrics.csv produced by run_evaluation.py")
    p.add_argument("--output-dir", default=None,
                   help="Directory to save plot files (default: same directory as metrics CSV)")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                   help="Output image format (default: png)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    metrics_path = Path(args.metrics_csv)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")

    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path)
    required = {"model_id", "attack_id", "lambda", "risk", "risk_lower", "risk_upper"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    print(f"Loaded {len(df)} rows — "
          f"{df['model_id'].nunique()} model(s), "
          f"{df['attack_id'].nunique()} attack(s)")

    print("\nGenerating per-attack plots...")
    plot_per_attack(df, output_dir, args.format)

    print("\nGenerating combined plot...")
    plot_combined(df, output_dir, args.format)

    print(f"\nDone. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
