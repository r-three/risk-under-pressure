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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #

def _setup_style() -> None:
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_context("notebook", font_scale=1.3)
    font_path = Path("AtkinsonHyperlegible-Regular.ttf")
    if font_path.exists():
        matplotlib.font_manager.fontManager.addfont(str(font_path))
        plt.rcParams.update({"font.family": "Atkinson Hyperlegible"})


# --------------------------------------------------------------------------- #
# Aesthetics
# --------------------------------------------------------------------------- #

PALETTE = ["#f6511d", "#ffb400", "#00a6ed", "#7fb800", "#8338ec", "#ff4365", "#06d6a0", "#0d2c54"]

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
    show_ci: bool = True,
) -> None:
    """Draw a single risk-pressure curve with optional CI error bars onto ax."""
    df_pair = df_pair.sort_values("lambda")
    lam = df_pair["lambda"].values
    risk = df_pair["risk"].values

    if show_ci:
        lo = df_pair["risk_lower"].values
        hi = df_pair["risk_upper"].values
        ax.errorbar(
            lam, risk,
            yerr=[risk - lo, hi - risk],
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
    else:
        ax.plot(
            lam, risk,
            color=color,
            label=label,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=5,
            zorder=3,
        )


def _style_axes(ax: plt.Axes, title: str, lambda_max: float) -> None:
    ax.set_xlabel("Pressure level λ")
    ax.set_ylabel("Risk R̂(M, λ)")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(left=0, right=lambda_max + 0.5)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    sns.despine(ax=ax)
    ax.tick_params(direction="out", which="both")


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

        fig.tight_layout(pad=0.5)
        out_path = output_dir / f"risk_curves_{attack}.{fmt}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
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
                show_ci=False,
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
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0, bbox_extra_artists=(leg1, leg2))
    plt.close(fig)
    print(f"  Saved: {out_path}")


# --------------------------------------------------------------------------- #
# Category-level plots
# --------------------------------------------------------------------------- #

def _category_color(categories: list[str]) -> dict[str, str]:
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(categories)}


def plot_category_curves(df_cat: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """One figure per attack: per-category risk curves, one subplot per model."""
    attacks = sorted(df_cat["attack_id"].unique())
    models = sorted(df_cat["model_id"].unique())
    categories = sorted(df_cat["category"].unique())
    color_map = _category_color(categories)
    lambda_max = df_cat["lambda"].max()

    for attack in attacks:
        df_a = df_cat[df_cat["attack_id"] == attack]
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5), sharey=True)
        if n_models == 1:
            axes = [axes]

        for ax, model in zip(axes, models):
            df_m = df_a[df_a["model_id"] == model]
            for cat in categories:
                df_pair = df_m[df_m["category"] == cat].sort_values("lambda")
                if df_pair.empty:
                    continue
                ax.plot(df_pair["lambda"], df_pair["risk"],
                        color=color_map[cat], label=cat,
                        marker="o", linestyle="-", linewidth=1.6, markersize=4)
            _style_axes(ax, model, lambda_max)
            ax.set_title(model, fontweight="bold")

        axes[0].set_ylabel("Risk R̂(M, λ)", fontsize=11)
        for ax in axes[1:]:
            ax.set_ylabel("")

        # Shared legend on the right
        handles = [plt.Line2D([0], [0], color=color_map[c], linewidth=2, label=c)
                   for c in categories]
        fig.legend(handles=handles, title="Category", fontsize=7, title_fontsize=8,
                   loc="center right", bbox_to_anchor=(1.0, 0.5),
                   borderpad=0.8, labelspacing=0.4)
        fig.suptitle(f"Risk by Category — {_attack_label(attack)}", fontsize=12,
                     fontweight="bold", y=1.01)
        fig.tight_layout(pad=0.5)
        out_path = output_dir / f"risk_curves_by_category_{attack}.{fmt}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0, bbox_extra_artists=fig.legends)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_category_heatmap(df_cat: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """One heatmap per attack: rows=categories, columns=models, value=risk at lambda_max."""
    import seaborn as sns

    attacks = sorted(df_cat["attack_id"].unique())
    lambda_max = df_cat["lambda"].max()
    df_max = df_cat[df_cat["lambda"] == lambda_max]

    for attack in attacks:
        df_a = df_max[df_max["attack_id"] == attack]
        pivot = df_a.pivot_table(index="category", columns="model_id", values="risk")
        pivot = pivot.sort_index()

        fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns) * 2.2), max(4, len(pivot) * 0.65)))
        sns.heatmap(
            pivot, ax=ax, vmin=0, vmax=1, cmap="YlOrRd",
            annot=pivot.applymap(lambda v: f"{v:.0%}"),
            fmt="", linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Risk", "shrink": 1.0,
                      "format": mticker.PercentFormatter(xmax=1, decimals=0)},
        )
        ax.set_title(f"Risk at λ={lambda_max:.0f} — {_attack_label(attack)}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel("Category", fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout(pad=0.5)
        out_path = output_dir / f"heatmap_category_{attack}.{fmt}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_break_pressure(df_cat: pd.DataFrame, output_dir: Path, fmt: str,
                        tau: float = 0.5) -> None:
    """Horizontal bar chart: min lambda to reach tau risk per category, grouped by model."""
    models = sorted(df_cat["model_id"].unique())
    categories = sorted(df_cat["category"].unique())
    color_map = _model_color(models)

    # For each (model, attack, category) find min lambda where risk >= tau
    rows = []
    for (model, attack, cat), grp in df_cat.groupby(["model_id", "attack_id", "category"]):
        grp = grp.sort_values("lambda")
        hit = grp[grp["risk"] >= tau]
        bp = hit["lambda"].iloc[0] if not hit.empty else None
        rows.append({"model_id": model, "attack_id": attack, "category": cat, "break_pressure": bp})

    df_bp = pd.DataFrame(rows)
    # Average across attacks per (model, category); None -> lambda_max + 1 (never broken)
    lambda_max = df_cat["lambda"].max()
    df_bp["break_pressure"] = df_bp["break_pressure"].fillna(lambda_max + 1)
    df_mean = df_bp.groupby(["category", "model_id"])["break_pressure"].mean().reset_index()

    # Sort categories by mean break pressure across all models
    cat_order = (df_mean.groupby("category")["break_pressure"].mean()
                 .sort_values().index.tolist())

    n_cats = len(cat_order)
    n_models = len(models)
    bar_height = 0.7 / n_models
    fig, ax = plt.subplots(figsize=(8, max(4, n_cats * 0.55)))

    for i, model in enumerate(models):
        df_m = df_mean[df_mean["model_id"] == model].set_index("category")
        values = [df_m.loc[c, "break_pressure"] if c in df_m.index else lambda_max + 1
                  for c in cat_order]
        y_pos = np.arange(n_cats) + (i - (n_models - 1) / 2) * bar_height
        ax.barh(y_pos, values, height=bar_height * 0.9,
                color=color_map[model], label=model, alpha=0.85)

    ax.set_yticks(np.arange(n_cats))
    ax.set_yticklabels(cat_order, fontsize=9)
    ax.set_xlabel("Break pressure λ (min λ to reach ≥50% risk)", fontsize=10)
    ax.set_title("Category Exploitability — Break Pressure", fontsize=12, fontweight="bold")
    ax.axvline(lambda_max + 1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(left=0, right=lambda_max + 1.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(title="Model", fontsize=8, title_fontsize=9,
              borderpad=0.8, labelspacing=0.4)
    sns.despine(ax=ax)
    ax.tick_params(direction="out", which="both")
    fig.tight_layout(pad=0.5)
    out_path = output_dir / f"break_pressure_by_category.{fmt}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot pRisk-Pressure curves from metrics CSV")
    p.add_argument("--metrics-csv", required=True,
                   help="Path to metrics.csv produced by run_evaluation.py")
    p.add_argument("--category-metrics-csv", default=None,
                   help="Path to metrics_by_category.csv (optional; enables category plots)")
    p.add_argument("--output-dir", default=None,
                   help="Directory to save plot files (default: same directory as metrics CSV)")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                   help="Output image format (default: png)")
    return p.parse_args()


def main() -> None:
    _setup_style()
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

    if args.category_metrics_csv:
        cat_path = Path(args.category_metrics_csv)
        if not cat_path.exists():
            raise FileNotFoundError(f"Category metrics CSV not found: {cat_path}")
        df_cat = pd.read_csv(cat_path)
        print(f"\nLoaded {len(df_cat)} category rows — "
              f"{df_cat['category'].nunique()} categories")

        print("\nGenerating per-category risk curves...")
        plot_category_curves(df_cat, output_dir, args.format)

        print("\nGenerating category heatmaps...")
        plot_category_heatmap(df_cat, output_dir, args.format)

        print("\nGenerating break-pressure chart...")
        plot_break_pressure(df_cat, output_dir, args.format)

    print(f"\nDone. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
