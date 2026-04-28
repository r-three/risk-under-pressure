#!/usr/bin/env python3
"""
plot_cost_curves.py — Cost-axis risk curves (tokens / TFLOPs) with seed decomposition.

Reads cost_metrics.csv (from compute_attack_costs.py), which contains one row
per (model_seedN, attack, lambda), and produces three plot families:

  1. Per-seed trace figures  — one figure per (base model × attack): each seed
     as a light translucent line with the seed-aggregated mean ± 95% CI overlaid.

  2. Seed-aggregated figures — one figure per base model: one line per attack,
     mean ± CI fill, seeds collapsed to a single curve.

  3. Comparison figures      — one figure per attack: all (base) models overlaid
     with seed-aggregated CIs.  Used for ablation sets.

Usage — per-model:
    python plot_cost_curves.py \\
        --cost-csv results/.../harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \\
        --output-dir results/plots/cost/harmbench/qwen2.5-0.5b-instruct/tokens \\
        --x-axis tokens

Usage — ablation comparison (pass multiple CSVs):
    python plot_cost_curves.py \\
        --cost-csv model1/cost/cost_metrics.csv model2/cost/cost_metrics.csv \\
        --output-dir results/plots/cost/harmbench/ablations/qwen_size/tokens \\
        --x-axis tokens \\
        --title "HarmBench — Qwen2.5 Size Ablation" \\
        --skip-missing
"""

from __future__ import annotations

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

# ─────────────────────────────────────────────────────────────────────────────
# Style constants
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = [
    "#f6511d", "#ffb400", "#00a6ed", "#7fb800",
    "#8338ec", "#ff4365", "#06d6a0", "#0d2c54",
]

ATTACK_MARKERS = {
    "gcg": "o", "pair": "s", "jailbroken": "^", "jailbroken-v1": "D",
}
ATTACK_LINESTYLES = {
    "gcg": "-", "pair": "--", "jailbroken": ":", "jailbroken-v1": "-.",
}
ATTACK_DISPLAY = {
    "gcg": "GCG", "pair": "PAIR",
    "jailbroken": "Jailbroken", "jailbroken-v1": "Jailbroken-v1",
}

SEED_RE = re.compile(r"_seed\d+$")

COST_COLS_ALL = [
    "mean_target_tokens", "mean_judge_tokens", "mean_total_tokens",
    "mean_target_tflops", "mean_judge_tflops", "mean_total_tflops",
]

X_AXIS_META: dict[str, dict] = {
    "tokens": {
        "col":     "mean_total_tokens",
        "label":   "Mean token cost per prompt",
        "k_scale": True,
    },
    "flops": {
        "col":     "mean_total_tflops",
        "label":   "Attack compute budget (TFLOPs)",
        "k_scale": False,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _setup_style() -> None:
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_context("notebook", font_scale=1.3)
    font_path = Path("AtkinsonHyperlegible-Regular.ttf")
    if font_path.exists():
        matplotlib.font_manager.fontManager.addfont(str(font_path))
        plt.rcParams.update({"font.family": "Atkinson Hyperlegible"})


def _attack_label(a: str) -> str:
    return ATTACK_DISPLAY.get(a, a)


def _model_color(models: list[str]) -> dict[str, str]:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(sorted(set(models)))}


def _base(model_id: str) -> str:
    return SEED_RE.sub("", model_id)


def _xcol(x_axis: str) -> str:
    return X_AXIS_META[x_axis]["col"]


def _style_axes(ax: plt.Axes, title: str, x_max: float, x_axis: str) -> None:
    meta   = X_AXIS_META[x_axis]
    xlabel = meta["label"]
    if meta["k_scale"] and x_max > 2000:
        xlabel = xlabel + " (k)"
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v / 1000:.1f}k")
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Risk $\hat{R}(M, \lambda)$")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(left=0, right=x_max * 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    sns.despine(ax=ax)
    ax.tick_params(direction="out", which="both")


def load_cost_csv(paths: list[Path], skip_missing: bool = False) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p.exists():
            if skip_missing:
                print(f"  WARNING: {p} not found — skipping")
                continue
            raise FileNotFoundError(f"Cost CSV not found: {p}")
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("No cost CSV files could be loaded (all paths missing)")
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Seed aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_seeds(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Collapse *_seedN model_ids into base model with t-distribution CIs.

    Rows without a _seedN suffix (e.g. qwen3-8b) are returned as-is, keeping
    any bootstrap risk_lower/risk_upper already stored in the CSV.
    """
    df = df.copy()
    df["_base"] = df["model_id"].apply(_base)

    # No seed suffixes → return unchanged (preserve existing CI columns)
    if (df["_base"] == df["model_id"]).all():
        df.drop(columns=["_base"], inplace=True)
        if "n_seeds" not in df.columns:
            df["n_seeds"] = 1
        return df

    extra_num = ["aurc", "delta_r", "lambda_star", "n_prompts"] + [
        c for c in COST_COLS_ALL if c in df.columns
    ]

    rows: list[dict] = []
    for (bm, atk, lam), grp in df.groupby(
        ["_base", "attack_id", "lambda"], sort=False
    ):
        n      = len(grp)
        risks  = grp["risk"].values
        mean_r = float(risks.mean())

        if n > 1:
            sem    = risks.std(ddof=1) / n ** 0.5
            t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=n - 1)
            lo = float(np.clip(mean_r - t_crit * sem, 0.0, 1.0))
            hi = float(np.clip(mean_r + t_crit * sem, 0.0, 1.0))
        else:
            # Single row: fall back to stored bootstrap CI if available
            if "risk_lower" in grp.columns and "risk_upper" in grp.columns:
                lo = float(grp["risk_lower"].values[0])
                hi = float(grp["risk_upper"].values[0])
            else:
                lo = hi = mean_r

        row: dict = {
            "model_id":   bm,
            "attack_id":  atk,
            "lambda":     lam,
            "risk":       mean_r,
            "risk_lower": lo,
            "risk_upper": hi,
            "n_seeds":    n,
        }
        for col in extra_num:
            if col in grp.columns:
                row[col] = pd.to_numeric(grp[col], errors="coerce").mean()
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Per-seed trace figures
#   One figure per (base_model × attack).
#   Each seed as a light translucent line; mean ± CI overlaid in dark.
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_seed_traces(
    df_raw: pd.DataFrame,
    df_agg: pd.DataFrame,
    output_dir: Path,
    fmt: str,
    x_axis: str = "tokens",
) -> None:
    xcol = _xcol(x_axis)
    if xcol not in df_raw.columns:
        print(f"  Skipping per-seed traces: column '{xcol}' not in data")
        return

    df_raw = df_raw.copy()
    df_raw["_base"] = df_raw["model_id"].apply(_base)

    base_models = sorted(df_raw["_base"].unique())
    attacks     = sorted(df_raw["attack_id"].unique())
    seed_color  = PALETTE[0]   # single hue; alpha distinguishes seeds from mean

    for bm in base_models:
        df_bm = df_raw[df_raw["_base"] == bm]
        seeds  = sorted(df_bm["model_id"].unique())
        n_seeds = len(seeds)

        for attack in attacks:
            df_a = df_bm[df_bm["attack_id"] == attack]
            if df_a.empty:
                continue

            fig, ax = plt.subplots(figsize=(7, 5))

            # Individual seed traces
            for s_idx, seed in enumerate(seeds):
                ds = df_a[df_a["model_id"] == seed].sort_values(xcol)
                if ds.empty:
                    continue
                ax.plot(
                    ds[xcol], ds["risk"],
                    color=seed_color, alpha=0.20, linewidth=0.9,
                    marker=ATTACK_MARKERS.get(attack, "o"), markersize=3,
                    zorder=2,
                )

            # Aggregated mean ± CI
            df_mean = (
                df_agg[
                    (df_agg["model_id"] == bm) & (df_agg["attack_id"] == attack)
                ].sort_values(xcol)
            )
            if not df_mean.empty:
                x_vals = df_mean[xcol].values
                r_vals = df_mean["risk"].values
                lo_vals = df_mean["risk_lower"].values
                hi_vals = df_mean["risk_upper"].values
                ax.plot(
                    x_vals, r_vals,
                    color="#1a1a2e", linewidth=2.2,
                    marker=ATTACK_MARKERS.get(attack, "o"), markersize=5,
                    zorder=5,
                )
                ax.fill_between(
                    x_vals, lo_vals, hi_vals,
                    color="#1a1a2e", alpha=0.15, zorder=4,
                )

            x_max = df_a[xcol].max() if not df_a.empty else 1.0
            _style_axes(
                ax,
                f"{bm} — {_attack_label(attack)}",
                x_max, x_axis,
            )

            from matplotlib.lines import Line2D
            legend_handles = [
                Line2D(
                    [0], [0], color=seed_color, alpha=0.5, linewidth=1.2,
                    marker=ATTACK_MARKERS.get(attack, "o"), markersize=3,
                    label=f"Individual seeds (n={n_seeds})",
                ),
                Line2D(
                    [0], [0], color="#1a1a2e", linewidth=2.2,
                    marker=ATTACK_MARKERS.get(attack, "o"), markersize=5,
                    label="Mean ± 95% CI",
                ),
            ]
            ax.legend(
                handles=legend_handles, fontsize=9,
                loc="lower right", framealpha=0.9,
                borderpad=0.8, labelspacing=0.4,
            )
            fig.tight_layout(pad=0.5)

            safe_bm = bm.replace("/", "_")
            out_path = output_dir / f"cost_seeds_{safe_bm}_{attack}.{fmt}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Seed-aggregated per-model figures
#   One figure per base model; one line per attack with CI fill.
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_aggregated(
    df_agg: pd.DataFrame,
    output_dir: Path,
    fmt: str,
    x_axis: str = "tokens",
) -> None:
    xcol = _xcol(x_axis)
    if xcol not in df_agg.columns:
        print(f"  Skipping model-aggregated: column '{xcol}' not in data")
        return

    models  = sorted(df_agg["model_id"].unique())
    attacks = sorted(df_agg["attack_id"].unique())
    atk_colors = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(attacks)}

    for model in models:
        df_m  = df_agg[df_agg["model_id"] == model]
        x_max = float(df_m[xcol].max()) if not df_m.empty else 1.0

        fig, ax = plt.subplots(figsize=(7, 5))
        for attack in attacks:
            df_a = df_m[df_m["attack_id"] == attack].sort_values(xcol)
            if df_a.empty:
                continue
            color = atk_colors[attack]
            n_s   = int(df_a["n_seeds"].max()) if "n_seeds" in df_a.columns else 1
            label = f"{_attack_label(attack)} (n={n_s})"
            ax.plot(
                df_a[xcol], df_a["risk"],
                color=color, label=label,
                marker=ATTACK_MARKERS.get(attack, "o"),
                linestyle=ATTACK_LINESTYLES.get(attack, "-"),
                linewidth=1.8, markersize=5, zorder=3,
            )
            has_ci = "n_seeds" in df_a.columns and (df_a["n_seeds"] > 1).any()
            if has_ci:
                ax.fill_between(
                    df_a[xcol], df_a["risk_lower"], df_a["risk_upper"],
                    color=color, alpha=0.18, zorder=2,
                )

        _style_axes(ax, f"Cost-Risk Curves — {model}", x_max, x_axis)
        ax.legend(
            title="Attack", fontsize=9, title_fontsize=9,
            loc="lower right", framealpha=0.9,
            borderpad=0.8, labelspacing=0.4,
        )
        fig.tight_layout(pad=0.5)

        safe_m   = model.replace("/", "_")
        out_path = output_dir / f"cost_aggregated_{safe_m}.{fmt}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Per-attack comparison figures
#   One figure per attack; all (seed-aggregated) models overlaid.
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(
    df_agg: pd.DataFrame,
    output_dir: Path,
    fmt: str,
    x_axis: str = "tokens",
    suptitle: str = "",
) -> None:
    xcol = _xcol(x_axis)
    if xcol not in df_agg.columns:
        print(f"  Skipping comparison: column '{xcol}' not in data")
        return

    models    = sorted(df_agg["model_id"].unique())
    attacks   = sorted(df_agg["attack_id"].unique())
    color_map = _model_color(models)

    for attack in attacks:
        df_a  = df_agg[df_agg["attack_id"] == attack]
        x_max = float(df_a[xcol].max()) if not df_a.empty else 1.0

        fig, ax = plt.subplots(figsize=(7, 5))
        for model in models:
            df_m = df_a[df_a["model_id"] == model].sort_values(xcol)
            if df_m.empty:
                continue
            color = color_map[model]
            ax.plot(
                df_m[xcol], df_m["risk"],
                color=color, label=model,
                marker=ATTACK_MARKERS.get(attack, "o"),
                linewidth=1.8, markersize=5, zorder=3,
            )
            has_ci = "n_seeds" in df_m.columns and (df_m["n_seeds"] > 1).any()
            if has_ci:
                ax.fill_between(
                    df_m[xcol], df_m["risk_lower"], df_m["risk_upper"],
                    color=color, alpha=0.15, zorder=2,
                )

        _style_axes(
            ax, f"Cost-Risk Curves — {_attack_label(attack)}", x_max, x_axis
        )
        ax.legend(
            title="Model", fontsize=9, title_fontsize=9,
            loc="lower right", framealpha=0.9,
            borderpad=0.8, labelspacing=0.4,
        )
        if suptitle:
            fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.01)
        fig.tight_layout(pad=0.5)

        out_path = output_dir / f"cost_comparison_{attack}.{fmt}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Combined figure (all models × attacks on one canvas)
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined(
    df_agg: pd.DataFrame,
    output_dir: Path,
    fmt: str,
    x_axis: str = "tokens",
    suptitle: str = "",
) -> None:
    xcol = _xcol(x_axis)
    if xcol not in df_agg.columns:
        return

    models    = sorted(df_agg["model_id"].unique())
    attacks   = sorted(df_agg["attack_id"].unique())
    color_map = _model_color(models)
    x_max     = float(df_agg[xcol].max())

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10, 5))

    for model in models:
        for attack in attacks:
            df_p = df_agg[
                (df_agg["model_id"] == model) & (df_agg["attack_id"] == attack)
            ].sort_values(xcol)
            if df_p.empty:
                continue
            ax.plot(
                df_p[xcol], df_p["risk"],
                color=color_map[model],
                marker=ATTACK_MARKERS.get(attack, "o"),
                linestyle=ATTACK_LINESTYLES.get(attack, "-"),
                linewidth=1.4, markersize=4, alpha=0.85,
            )

    model_handles  = [Patch(color=color_map[m], label=m) for m in models]
    attack_handles = [
        Line2D(
            [0], [0], color="gray",
            linestyle=ATTACK_LINESTYLES.get(a, "-"),
            marker=ATTACK_MARKERS.get(a, "o"),
            label=_attack_label(a),
        )
        for a in attacks
    ]
    leg1 = ax.legend(
        handles=model_handles, title="Model",
        fontsize=8, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9,
        borderpad=0.8, labelspacing=0.4,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=attack_handles, title="Attack",
        fontsize=8, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(1.01, 0.45), framealpha=0.9,
        borderpad=0.8, labelspacing=0.4,
    )

    _style_axes(ax, "Cost-Risk Curves — All Models & Attacks", x_max, x_axis)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.01)
    fig.subplots_adjust(right=0.68)
    out_path = output_dir / f"cost_combined.{fmt}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path, dpi=150, bbox_inches="tight", pad_inches=0,
        bbox_extra_artists=(leg1,),
    )
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot cost-axis (tokens/TFLOPs) risk curves with seed decomposition"
    )
    p.add_argument(
        "--cost-csv", nargs="+", required=True,
        help="Path(s) to cost_metrics.csv (per-seed rows expected). "
             "Pass multiple paths to compare models.",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory where plots are written",
    )
    p.add_argument(
        "--x-axis", default="tokens", choices=["tokens", "flops"],
        help="X-axis: 'tokens' = mean total token count (default), "
             "'flops' = mean total TFLOPs",
    )
    p.add_argument(
        "--format", default="png", choices=["png", "pdf", "svg"],
        help="Output image format (default: png)",
    )
    p.add_argument(
        "--title", default="",
        help="Optional suptitle for comparison / combined figures",
    )
    p.add_argument(
        "--attacks", nargs="+", default=None,
        help="Restrict to specific attack IDs, e.g. --attacks gcg pair",
    )
    p.add_argument(
        "--mode", default="all",
        choices=["all", "per-seed", "aggregated", "comparison"],
        help=(
            "Which plot families to generate.\n"
            "  all         — all four families (default)\n"
            "  per-seed    — only seed trace figures\n"
            "  aggregated  — only per-model aggregated figures\n"
            "  comparison  — only per-attack comparison + combined figure"
        ),
    )
    p.add_argument(
        "--skip-missing", action="store_true",
        help="Silently skip --cost-csv paths that do not exist",
    )
    return p.parse_args()


def main() -> None:
    _setup_style()
    args = parse_args()

    paths    = [Path(p) for p in args.cost_csv]
    df_raw   = load_cost_csv(paths, skip_missing=args.skip_missing)

    required = {"model_id", "attack_id", "lambda", "risk"}
    missing  = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    xcol = _xcol(args.x_axis)
    if xcol not in df_raw.columns:
        raise ValueError(
            f"Column '{xcol}' not found in the data. "
            "Run scripts/compute_attack_costs.py first to generate cost_metrics.csv."
        )

    if args.attacks:
        df_raw = df_raw[df_raw["attack_id"].isin(args.attacks)]
        if df_raw.empty:
            raise ValueError(f"No data after filtering to attacks: {args.attacks}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_agg = aggregate_seeds(df_raw)

    n_base    = df_raw["model_id"].apply(_base).nunique()
    n_seeds_m = (
        df_raw.groupby(df_raw["model_id"].apply(_base))["model_id"]
        .nunique()
        .max()
    )
    print(
        f"Loaded {len(df_raw)} rows — {n_base} base model(s), "
        f"up to {n_seeds_m} seed(s), {df_raw['attack_id'].nunique()} attack(s) "
        f"[x-axis: {args.x_axis}]"
    )

    mode = args.mode

    if mode in ("all", "per-seed"):
        print("\nGenerating per-seed trace plots…")
        plot_per_seed_traces(df_raw, df_agg, output_dir, args.format, x_axis=args.x_axis)

    if mode in ("all", "aggregated"):
        print("\nGenerating seed-aggregated per-model plots…")
        plot_model_aggregated(df_agg, output_dir, args.format, x_axis=args.x_axis)

    if mode in ("all", "comparison"):
        if n_base > 1:
            print("\nGenerating per-attack comparison plots…")
            plot_comparison(
                df_agg, output_dir, args.format,
                x_axis=args.x_axis, suptitle=args.title,
            )
            print("\nGenerating combined plot…")
            plot_combined(
                df_agg, output_dir, args.format,
                x_axis=args.x_axis, suptitle=args.title,
            )
        else:
            print("\nSingle model — skipping comparison / combined plots")

    print(f"\nDone. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
