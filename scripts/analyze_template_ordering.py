#!/usr/bin/env python3
"""
analyze_template_ordering.py — Compute per-template ASR mean and std across models.

Reads raw results.jsonl files produced by compute_template_ordering.py and prints
a comparison table of mean ± std ASR per template, plus a per-model breakdown.
Also saves bar chart visualizations.

Usage:
    python scripts/analyze_template_ordering.py \
        --results-dir /path/to/outputs/template_ordering

    # Save CSV and plots to a specific directory:
    python scripts/analyze_template_ordering.py \
        --results-dir /path/to/outputs/template_ordering \
        --output results/template_ordering_stats.csv \
        --plots-dir results/plots
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Analyze template ordering experiment results")
    p.add_argument(
        "--results-dir",
        default="/home/ehghaghi/scratch/ehghaghi/outputs/template_ordering",
        help="Directory containing per-model subdirs with results.jsonl files",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Optional path to save summary CSV (e.g. results/template_stats.csv)",
    )
    p.add_argument(
        "--plots-dir",
        default=None,
        help="Directory to save bar chart plots (e.g. results/plots). Skipped if not set.",
    )
    return p.parse_args()


def compute_asr(results_path: Path) -> float:
    """Return ASR (fraction of successful attacks) from a results.jsonl file."""
    successes, total = 0, 0
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] Skipping malformed JSON line in {results_path}: {e}")
                continue
            total += 1
            if record.get("success", False):
                successes += 1
    if total == 0:
        return float("nan")
    return successes / total


def load_all_results(results_dir: Path) -> dict[str, dict[str, float]]:
    """
    Returns asr[template_name][model_id] = asr_value.
    Template names have the 'jailbroken_' prefix stripped.
    """
    asr: dict[str, dict[str, float]] = defaultdict(dict)

    model_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir())
    if not model_dirs:
        raise ValueError(f"No model subdirectories found in {results_dir}")

    for model_dir in model_dirs:
        model_id = model_dir.name
        for attack_dir in sorted(model_dir.iterdir()):
            if not attack_dir.is_dir():
                continue
            results_path = attack_dir / "results.jsonl"
            if not results_path.exists():
                print(f"  [warn] Missing: {results_path}")
                continue
            # Strip the 'jailbroken_' prefix from the attack dir name
            template_name = attack_dir.name.removeprefix("jailbroken_")
            asr[template_name][model_id] = compute_asr(results_path)

    return dict(asr)


def summarize(asr: dict[str, dict[str, float]]) -> list[dict]:
    """Compute mean, std, min, max ASR per template across models."""
    import math

    rows = []
    for template, per_model in sorted(asr.items()):
        values = [v for v in per_model.values() if not math.isnan(v)]
        if not values:
            mean, std, mn, mx = float("nan"), float("nan"), float("nan"), float("nan")
        else:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance)
            mn, mx = min(values), max(values)
        rows.append(
            {
                "template": template,
                "mean": mean,
                "std": std,
                "min": mn,
                "max": mx,
                "n_models": len(values),
                "per_model": per_model,
            }
        )
    # Sort by mean ASR ascending (weakest → strongest template)
    rows.sort(key=lambda r: r["mean"])
    return rows


def print_table(rows: list[dict], all_models: list[str]):
    col_w = 12

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("  Template ASR — Mean ± Std across models (sorted by mean ASR)")
    print("=" * 70)
    header = f"{'Template':<25}  {'Mean':>6}  {'±Std':>6}  {'Min':>6}  {'Max':>6}  {'N':>3}"
    print(header)
    print("-" * 70)
    for r in rows:
        print(
            f"{r['template']:<25}  "
            f"{r['mean']:>6.3f}  "
            f"{r['std']:>6.3f}  "
            f"{r['min']:>6.3f}  "
            f"{r['max']:>6.3f}  "
            f"{r['n_models']:>3}"
        )
    print("=" * 70)

    # --- Per-model breakdown ---
    print("\n" + "=" * (25 + col_w * len(all_models) + 4))
    print("  Per-model ASR breakdown")
    print("=" * (25 + col_w * len(all_models) + 4))
    header = f"{'Template':<25}" + "".join(f"  {m[:10]:>10}" for m in all_models)
    print(header)
    print("-" * (25 + col_w * len(all_models) + 4))
    for r in rows:
        row_str = f"{r['template']:<25}"
        for m in all_models:
            val = r["per_model"].get(m, float("nan"))
            row_str += f"  {val:>10.3f}"
        print(row_str)
    print("=" * (25 + col_w * len(all_models) + 4))


def plot_charts(rows: list[dict], all_models: list[str], plots_dir: Path):
    """
    Save two bar charts:
      1. mean_asr_by_template.png  — mean ± std per template (aggregated across models)
      2. per_model_asr.png         — grouped bars: one group per template, one bar per model
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plots_dir.mkdir(parents=True, exist_ok=True)
    templates = [r["template"] for r in rows]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]

    # ── Plot 1: Mean ± Std per template ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(templates))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8, ecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(templates, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("ASR", fontsize=12)
    ax.set_title("Mean ± Std ASR per Template (across all models)", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    # Annotate bars with mean value
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05,
            f"{mean:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    out1 = plots_dir / "mean_asr_by_template.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out1}")

    # ── Plot 2: Grouped bars — template × model ───────────────────────────────
    n_templates = len(templates)
    n_models = len(all_models)
    bar_width = 0.8 / n_models
    colors = plt.cm.tab10.colors  # up to 10 distinct colors

    fig, ax = plt.subplots(figsize=(max(12, n_templates * 1.5), 6))
    x = np.arange(n_templates)

    for i, model in enumerate(all_models):
        model_asrs = [r["per_model"].get(model, float("nan")) for r in rows]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            model_asrs,
            width=bar_width,
            label=model,
            color=colors[i % len(colors)],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(templates, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("ASR", fontsize=12)
    ax.set_title("ASR per Template per Model", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    out2 = plots_dir / "per_model_asr.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out2}")


def save_csv(rows: list[dict], all_models: list[str], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["template", "mean", "std", "min", "max", "n_models"] + all_models)
        for r in rows:
            per_model_vals = [r["per_model"].get(m, "") for m in all_models]
            writer.writerow(
                [
                    r["template"],
                    f"{r['mean']:.4f}",
                    f"{r['std']:.4f}",
                    f"{r['min']:.4f}",
                    f"{r['max']:.4f}",
                    r["n_models"],
                ]
                + [f"{v:.4f}" if isinstance(v, float) else v for v in per_model_vals]
            )
    print(f"\nCSV saved to: {output_path}")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    print(f"Loading results from: {results_dir}")
    asr = load_all_results(results_dir)

    all_models = sorted({m for per_model in asr.values() for m in per_model})
    print(f"Models found: {all_models}")
    print(f"Templates found: {sorted(asr.keys())}")

    rows = summarize(asr)
    print_table(rows, all_models)

    if args.output:
        save_csv(rows, all_models, Path(args.output))

    if args.plots_dir:
        plot_charts(rows, all_models, Path(args.plots_dir))


if __name__ == "__main__":
    main()
