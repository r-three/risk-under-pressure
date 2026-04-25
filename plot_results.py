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
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

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

def _x_col(x_axis: str) -> str:
    return _X_AXIS_META.get(x_axis, _X_AXIS_META["lambda"])["col"]


def _plot_risk_curve(
    ax: plt.Axes,
    df_pair: pd.DataFrame,
    color: str,
    label: str,
    marker: str = "o",
    linestyle: str = "-",
    show_ci: bool = True,
    x_axis: str = "lambda",
) -> None:
    """Draw a single risk curve with optional CI error bars onto ax."""
    xcol = _x_col(x_axis)
    if xcol not in df_pair.columns:
        xcol = "lambda"
    df_pair = df_pair.sort_values(xcol)
    x    = df_pair[xcol].values
    risk = df_pair["risk"].values

    if show_ci:
        lo = df_pair["risk_lower"].values
        hi = df_pair["risk_upper"].values
        ax.errorbar(
            x, risk,
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
            x, risk,
            color=color,
            label=label,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=5,
            zorder=3,
        )


_X_AXIS_META = {
    "lambda":  {"col": "lambda",             "label": r"Pressure level $\lambda$",        "integer_ticks": True,  "k_scale": False},
    "tokens":  {"col": "mean_total_tokens",  "label": "Attack token budget (total tokens)", "integer_ticks": False, "k_scale": True},
    "flops":   {"col": "mean_total_tflops",  "label": "Attack compute budget (TFLOPs)",     "integer_ticks": False, "k_scale": False},
}


def _style_axes(ax: plt.Axes, title: str, x_max: float, x_axis: str = "lambda") -> None:
    meta = _X_AXIS_META.get(x_axis, _X_AXIS_META["lambda"])
    xlabel = meta["label"]
    if meta["k_scale"] and x_max > 2000:
        xlabel = xlabel.replace("total tokens", "total tokens (k)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Risk $\hat{R}(M, \lambda)$")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(left=0, right=x_max * 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    if meta["integer_ticks"]:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if meta["k_scale"] and x_max > 2000:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.1f}k"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    sns.despine(ax=ax)
    ax.tick_params(direction="out", which="both")


# --------------------------------------------------------------------------- #
# Per-attack figures
# --------------------------------------------------------------------------- #

def _compute_x_max(df: pd.DataFrame, xcol: str, x_axis: str) -> float:
    """For token/flop axes, use the max value at the highest pressure level only."""
    if x_axis == "lambda" or xcol not in df.columns:
        return df.get(xcol, df["lambda"]).max()
    lambda_max = df["lambda"].max()
    df_top = df[df["lambda"] == lambda_max]
    return df_top[xcol].max() if not df_top.empty else df[xcol].max()


def plot_per_attack(df: pd.DataFrame, output_dir: Path, fmt: str, x_axis: str = "lambda",
                    suptitle: str = "") -> None:
    """One figure per attack: all models overlaid."""
    attacks = sorted(df["attack_id"].unique())
    models = sorted(df["model_id"].unique())
    color_map = _model_color(models)
    xcol = _x_col(x_axis)
    x_max = _compute_x_max(df, xcol, x_axis)

    for attack in attacks:
        df_attack = df[df["attack_id"] == attack]
        attack_x_max = _compute_x_max(df_attack, xcol, x_axis)
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
                x_axis=x_axis,
            )

        _style_axes(ax, f"Risk-Pressure Curves — {_attack_label(attack)}", attack_x_max, x_axis)
        ax.legend(title="Model", fontsize=9, title_fontsize=9,
                  loc="lower right", framealpha=0.9,
                  borderpad=0.8, labelspacing=0.4)

        if suptitle:
            fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.01)
        fig.tight_layout(pad=0.5)
        out_path = output_dir / f"risk_curves_{attack}.{fmt}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# --------------------------------------------------------------------------- #
# All-in-one figure
# --------------------------------------------------------------------------- #

def plot_combined(df: pd.DataFrame, output_dir: Path, fmt: str, x_axis: str = "lambda",
                  suptitle: str = "") -> None:
    """Single figure: all (model, attack) pairs with colour=model, style=attack."""
    models = sorted(df["model_id"].unique())
    attacks = sorted(df["attack_id"].unique())
    color_map = _model_color(models)
    xcol = _x_col(x_axis)
    x_max = _compute_x_max(df, xcol, x_axis)

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
                x_axis=x_axis,
            )

    _style_axes(ax, "Risk-Pressure Curves — All Models & Attacks", x_max, x_axis)

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

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.01)
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
    """One figure per (attack, base_model): per-category risk curves with CI bands.

    Expects df_cat to already be seed-aggregated (model_id = base model name).
    """
    attacks = sorted(df_cat["attack_id"].unique())
    models = sorted(df_cat["model_id"].unique())
    categories = sorted(df_cat["category"].unique())
    color_map = _category_color(categories)
    lambda_max = df_cat["lambda"].max()

    handles = [plt.Line2D([0], [0], color=color_map[c], linewidth=2, label=c)
               for c in categories]

    for attack in attacks:
        df_a = df_cat[df_cat["attack_id"] == attack]
        for model in models:
            df_m = df_a[df_a["model_id"] == model]
            if df_m.empty:
                continue

            fig, ax = plt.subplots(figsize=(6, 4.5))
            for cat in categories:
                df_pair = df_m[df_m["category"] == cat].sort_values("lambda")
                if df_pair.empty:
                    continue
                ax.plot(df_pair["lambda"], df_pair["risk"],
                        color=color_map[cat], label=cat,
                        marker="o", linestyle="-", linewidth=1.6, markersize=4)
                if "risk_lower" in df_pair.columns and "risk_upper" in df_pair.columns:
                    ax.fill_between(df_pair["lambda"],
                                    df_pair["risk_lower"], df_pair["risk_upper"],
                                    color=color_map[cat], alpha=0.15)
            _style_axes(ax, f"{model} — {_attack_label(attack)}", lambda_max)
            ax.legend(handles=handles, title="Category",
                      fontsize=7, title_fontsize=8,
                      loc="lower right", framealpha=0.9,
                      borderpad=0.8, labelspacing=0.4)
            fig.tight_layout(pad=0.5)
            out_path = output_dir / f"risk_curves_by_category_{attack}_{model}.{fmt}"
            fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
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
# Cross-model category comparison (one figure per attack × category)
# --------------------------------------------------------------------------- #

def plot_category_comparison(df_cat: pd.DataFrame, output_dir: Path, fmt: str,
                              suptitle: str = "") -> None:
    """One figure per (attack, category): all models overlaid with seed-CI error bars.

    Expects df_cat to already be seed-aggregated (model_id = base model name).
    """
    attacks    = sorted(df_cat["attack_id"].unique())
    categories = sorted(df_cat["category"].unique())
    models     = sorted(df_cat["model_id"].unique())
    color_map  = _model_color(models)
    lambda_max = df_cat["lambda"].max()

    for attack in attacks:
        for category in categories:
            df_ac = df_cat[(df_cat["attack_id"] == attack) & (df_cat["category"] == category)]
            if df_ac.empty:
                continue

            fig, ax = plt.subplots(figsize=(6, 4.5))
            for model in models:
                df_m = df_ac[df_ac["model_id"] == model].sort_values("lambda")
                if df_m.empty:
                    continue
                _plot_risk_curve(
                    ax, df_m,
                    color=color_map[model],
                    label=model,
                    marker=ATTACK_MARKERS.get(attack, "o"),
                    linestyle="-",
                )

            _style_axes(ax, f"{_attack_label(attack)} — {category}", lambda_max)
            ax.legend(title="Model", fontsize=9, title_fontsize=9,
                      loc="lower right", framealpha=0.9,
                      borderpad=0.8, labelspacing=0.4)
            if suptitle:
                fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.01)
            fig.tight_layout(pad=0.5)

            safe_cat = category.replace("/", "_").replace(" ", "_")
            out_path = output_dir / f"risk_curves_category_{attack}_{safe_cat}.{fmt}"
            fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print(f"  Saved: {out_path}")


# --------------------------------------------------------------------------- #
# Seed-aggregated plot (one line per attack, CI from seed variance)
# --------------------------------------------------------------------------- #

def plot_seed_aggregated(df: pd.DataFrame, output_dir: Path, fmt: str, x_axis: str = "lambda") -> None:
    """One figure: one line per attack, risk averaged across seeds with CI shading."""
    df_agg = aggregate_seeds(df)
    attacks = sorted(df_agg["attack_id"].unique())
    xcol = _x_col(x_axis)
    xcol_use = xcol if xcol in df_agg.columns else "lambda"

    attack_colors = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(attacks)}
    x_max = _compute_x_max(df_agg, xcol_use, x_axis)

    fig, ax = plt.subplots(figsize=(7, 5))

    for attack in attacks:
        df_a = df_agg[df_agg["attack_id"] == attack].sort_values(xcol_use)
        if df_a.empty:
            continue
        x    = df_a[xcol_use].values
        risk = df_a["risk"].values
        lo   = df_a["risk_lower"].values
        hi   = df_a["risk_upper"].values
        color = attack_colors[attack]

        ax.plot(x, risk, color=color, label=_attack_label(attack),
                marker=ATTACK_MARKERS.get(attack, "o"),
                linestyle=ATTACK_LINESTYLES.get(attack, "-"),
                linewidth=1.8, markersize=5, zorder=3)
        ax.fill_between(x, lo, hi, color=color, alpha=0.2, zorder=2)

    _style_axes(ax, "Risk-Pressure Curves — Seed-Averaged", x_max, x_axis)
    ax.legend(title="Attack", fontsize=9, title_fontsize=9,
              loc="lower right", framealpha=0.9,
              borderpad=0.8, labelspacing=0.4)

    fig.tight_layout(pad=0.5)
    out_path = output_dir / f"risk_curves_seed_aggregated.{fmt}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# --------------------------------------------------------------------------- #
# Efficiency summary plot
# --------------------------------------------------------------------------- #

def plot_efficiency_summary(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """
    Grouped bar chart: expected token cost (mean_total_tokens at max lambda)
    per (model, attack), grouped by attack.

    Requires cost columns from compute_attack_costs.py.
    Falls back gracefully if the columns are missing.
    """
    if "mean_total_tokens" not in df.columns:
        print("  Skipping efficiency summary — run compute_attack_costs.py first to add cost columns")
        return

    lambda_max = df["lambda"].max()
    df_top = df[df["lambda"] == lambda_max].copy()
    df_top["mean_total_tokens"] = pd.to_numeric(df_top["mean_total_tokens"], errors="coerce")
    df_top = df_top.dropna(subset=["mean_total_tokens"])

    if df_top.empty:
        print("  Skipping efficiency summary — no cost data at max lambda")
        return

    attacks = sorted(df_top["attack_id"].unique())
    models  = sorted(df_top["model_id"].unique())
    color_map = _model_color(models)

    n_attacks = len(attacks)
    n_models  = len(models)
    bar_width = 0.7 / n_models
    x = np.arange(n_attacks)

    fig, ax = plt.subplots(figsize=(max(5, n_attacks * 2), 4.5))

    for i, model in enumerate(models):
        df_m = df_top[df_top["model_id"] == model].set_index("attack_id")
        values = [
            df_m.loc[a, "mean_total_tokens"] if a in df_m.index else 0.0
            for a in attacks
        ]
        offset = (i - (n_models - 1) / 2) * bar_width
        ax.bar(x + offset, values, width=bar_width * 0.9,
               color=color_map[model], label=model, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([_attack_label(a) for a in attacks], fontsize=10)
    ax.set_xlabel("Attack", fontsize=10)
    ax.set_ylabel("Expected tokens consumed", fontsize=10)
    ax.set_title(
        f"Attack Cost at λ={int(lambda_max)} — Expected Total Tokens",
        fontsize=12, fontweight="bold",
    )
    ax.legend(title="Model", fontsize=8, title_fontsize=9,
              borderpad=0.8, labelspacing=0.4)

    # K-scale if values are large
    ymax = ax.get_ylim()[1]
    if ymax > 2000:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.1f}k"))
        ax.set_ylabel("Expected tokens consumed (k)", fontsize=10)

    sns.despine(ax=ax)
    ax.tick_params(direction="out", which="both")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout(pad=0.5)

    out_path = output_dir / f"efficiency_summary.{fmt}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# --------------------------------------------------------------------------- #
# Seed aggregation
# --------------------------------------------------------------------------- #

_SEED_RE = re.compile(r"_seed\d+$")

COST_COLS = [
    "mean_target_tokens", "mean_total_tokens",
    "mean_target_tflops", "mean_total_tflops",
]


def aggregate_seeds(df: pd.DataFrame, alpha: float = 0.05,
                    extra_group_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Collapse multi-seed rows into one row per (base_model, attack, [extra_group_cols], lambda).

    extra_group_cols: additional columns to include in the groupby key, e.g. ["category"].
    Confidence intervals are computed from the empirical variance across seeds
    using a t-distribution with (n_seeds - 1) degrees of freedom.
    Rows whose model_id has no _seedN suffix are returned unchanged.
    """
    if extra_group_cols is None:
        extra_group_cols = []

    df = df.copy()
    df["_base_model"] = df["model_id"].str.replace(_SEED_RE, "", regex=True)

    if (df["_base_model"] == df["model_id"]).all():
        df.drop(columns=["_base_model"], inplace=True)
        return df

    numeric_extra = ["aurc", "delta_r", "lambda_star", "n_prompts"] + [
        c for c in COST_COLS if c in df.columns
    ]

    group_keys = ["_base_model", "attack_id"] + extra_group_cols + ["lambda"]

    rows = []
    for group_vals, grp in df.groupby(group_keys, sort=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        base_model = group_vals[0]
        attack_id  = group_vals[1]
        lam        = group_vals[-1]
        extra_vals = group_vals[2:-1] if extra_group_cols else ()

        n = len(grp)
        risks = grp["risk"].values
        mean_r = risks.mean()

        if n > 1:
            sem    = risks.std(ddof=1) / n ** 0.5
            t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=n - 1)
            lo = float(np.clip(mean_r - t_crit * sem, 0.0, 1.0))
            hi = float(np.clip(mean_r + t_crit * sem, 0.0, 1.0))
        else:
            lo = hi = float(mean_r)

        row: dict = {
            "model_id":   base_model,
            "attack_id":  attack_id,
            **{k: v for k, v in zip(extra_group_cols, extra_vals)},
            "lambda":     lam,
            "risk":       float(mean_r),
            "risk_lower": lo,
            "risk_upper": hi,
            "n_seeds":    n,
        }
        for col in numeric_extra:
            if col in grp.columns:
                row[col] = pd.to_numeric(grp[col], errors="coerce").mean()
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot pRisk-Pressure curves from metrics CSV")
    p.add_argument("--metrics-csv", nargs="+", required=True,
                   help="Path(s) to metrics.csv or metrics_summary.csv. Pass multiple paths to compare models across folders.")
    p.add_argument("--category-metrics-csv", nargs="+", default=None,
                   help="Path(s) to metrics_by_category.csv. Pass multiple paths to enable cross-model category comparison plots.")
    p.add_argument("--title", default="",
                   help="Optional suptitle added to all comparison figures (per-attack and category plots).")
    p.add_argument("--output-dir", default=None,
                   help="Directory to save plot files (default: same directory as metrics CSV)")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                   help="Output image format (default: png)")
    p.add_argument(
        "--x-axis", default="lambda", choices=["lambda", "tokens", "flops"],
        help=(
            "X-axis for risk curves. "
            "'lambda' = pressure steps (default); "
            "'tokens' = mean total tokens (requires cost_metrics.csv); "
            "'flops'  = mean total TFLOPs (requires cost_metrics.csv)"
        ),
    )
    p.add_argument(
        "--ci-method", default="bootstrap", choices=["bootstrap", "seeds"],
        help=(
            "Confidence interval method. "
            "'bootstrap' = per-seed bootstrap CIs stored in risk_lower/risk_upper (default); "
            "'seeds' = t-distribution CI computed from empirical variance across seeds "
            "(requires multi-seed metrics.csv with model_id like model_seed<N>)"
        ),
    )
    return p.parse_args()


def main() -> None:
    _setup_style()
    args = parse_args()

    metrics_paths = [Path(p) for p in args.metrics_csv]
    for metrics_path in metrics_paths:
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif len(metrics_paths) == 1:
        output_dir = metrics_paths[0].parent / "plots"
    else:
        output_dir = metrics_paths[0].parent.parent / "comparison_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.concat([pd.read_csv(p) for p in metrics_paths], ignore_index=True)
    required = {"model_id", "attack_id", "lambda", "risk", "risk_lower", "risk_upper"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    if args.x_axis != "lambda":
        needed_col = _x_col(args.x_axis)
        if needed_col not in df.columns:
            raise ValueError(
                f"Column '{needed_col}' not found in CSV. "
                "Run scripts/compute_attack_costs.py first to generate cost_metrics.csv, "
                "then pass that as --metrics-csv."
            )

    if args.ci_method == "seeds":
        df = aggregate_seeds(df)
        n_seeds_info = (
            f", {int(df['n_seeds'].max())} seeds" if "n_seeds" in df.columns else ""
        )
    else:
        n_seeds_info = ""

    print(f"Loaded {len(df)} rows — "
          f"{df['model_id'].nunique()} model(s), "
          f"{df['attack_id'].nunique()} attack(s)  "
          f"[x-axis: {args.x_axis}, CI: {args.ci_method}{n_seeds_info}]")

    title = args.title

    print("\nGenerating per-attack plots...")
    plot_per_attack(df, output_dir, args.format, x_axis=args.x_axis, suptitle=title)

    print("\nGenerating combined plot...")
    plot_combined(df, output_dir, args.format, x_axis=args.x_axis, suptitle=title)

    print("\nGenerating seed-aggregated plot...")
    plot_seed_aggregated(df, output_dir, args.format, x_axis=args.x_axis)

    print("\nGenerating efficiency summary...")
    plot_efficiency_summary(df, output_dir, args.format)

    if args.category_metrics_csv:
        cat_paths = [Path(p) for p in args.category_metrics_csv]
        for cp in cat_paths:
            if not cp.exists():
                raise FileNotFoundError(f"Category metrics CSV not found: {cp}")
        df_cat = pd.concat([pd.read_csv(cp) for cp in cat_paths], ignore_index=True)
        df_cat = aggregate_seeds(df_cat, extra_group_cols=["category"])
        print(f"\nLoaded {len(df_cat)} category rows — "
              f"{df_cat['category'].nunique()} categories, "
              f"{df_cat['model_id'].nunique()} model(s)")

        print("\nGenerating per-category risk curves (per model)...")
        plot_category_curves(df_cat, output_dir, args.format)

        print("\nGenerating cross-model category comparison plots...")
        plot_category_comparison(df_cat, output_dir, args.format, suptitle=title)

        print("\nGenerating category heatmaps...")
        plot_category_heatmap(df_cat, output_dir, args.format)

        print("\nGenerating break-pressure chart...")
        plot_break_pressure(df_cat, output_dir, args.format)

    print(f"\nDone. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
