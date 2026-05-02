#!/usr/bin/env python3
"""
compute_attack_costs.py — Augment metrics.csv with token and FLOP cost columns.

Reads the same results.jsonl tree used by run_evaluation.py, computes mean
cumulative token and FLOP costs at each pressure level, and writes
cost_metrics.csv in the same directory as the input metrics.csv.

cost_metrics.csv has all columns from metrics.csv plus:
  mean_target_tokens   — mean target-model tokens consumed up to this lambda
  mean_total_tokens    — mean total tokens (+ PAIR attacker) up to this lambda
  mean_target_tflops   — mean target-model TFLOPs up to this lambda
  mean_total_tflops    — mean total TFLOPs up to this lambda

Usage:
    python scripts/compute_attack_costs.py \\
        --results-dir results/multitrial_experiments/pressure_sensitivity/harmbench/qwen2.5-0.5b-instruct \\
        --metrics-csv  results/multitrial_experiments/pressure_sensitivity/harmbench/qwen2.5-0.5b-instruct/metrics.csv

    # Custom output path:
    python scripts/compute_attack_costs.py \\
        --results-dir path/to/results \\
        --metrics-csv path/to/metrics.csv \\
        --output path/to/cost_metrics.csv

    # Adjust GCG gradient-step multiplier (default 3.0 = forward+backward;
    # 128 candidate passes are always included separately):
    python scripts/compute_attack_costs.py \\
        --results-dir path/to/results \\
        --metrics-csv path/to/metrics.csv \\
        --gcg-backward-mult 3.0
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prisk.metrics.cost_mapper import aggregate_costs
from prisk.metrics.cost_summary import (
    compute_cost_summary_metrics,
    compute_cost_summary_by_category,
)
from prisk.utils.io import TrialRecord, read_jsonl
from prisk.utils.logging import get_logger

logger = get_logger("compute_attack_costs")

COST_COLS = [
    "mean_target_tokens",
    "mean_judge_tokens",
    "mean_total_tokens",
    "mean_target_tflops",
    "mean_judge_tflops",
    "mean_total_tflops",
]


def discover_results(results_dir: Path) -> dict[tuple[str, str], Path]:
    found: dict[tuple[str, str], Path] = {}
    for f in results_dir.rglob("results.jsonl"):
        attack_id = f.parent.name
        model_id  = f.parent.parent.name
        found[(model_id, attack_id)] = f
    return found


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute token/FLOP costs from JSONL results")
    p.add_argument("--results-dir", required=True,
                   help="Root of the results tree (same as run_evaluation.py)")
    p.add_argument("--metrics-csv", required=True,
                   help="Existing metrics.csv to read lambda levels and join results into")
    p.add_argument("--output", default=None,
                   help="Output CSV path (default: cost_metrics.csv next to metrics.csv)")
    p.add_argument("--gcg-backward-mult", type=float, default=3.0,
                   help="FLOPs multiplier for the GCG gradient step only (default 3.0 = "
                        "1 forward + 1 backward). The 128 candidate forward passes are "
                        "always counted separately and are not affected by this flag.")
    p.add_argument("--judge-model", default="llama3.1_8b_instruct",
                   help="Model ID of the safety judge used during inference "
                        "(default: llama3.1_8b_instruct). Used to compute judge token/FLOP costs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    metrics_path = Path(args.metrics_csv)

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)
    if not metrics_path.exists():
        logger.error(f"Metrics CSV not found: {metrics_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else metrics_path.parent / "cost_metrics.csv"

    # ------------------------------------------------------------------ #
    # Load existing metrics.csv to get lambda levels and join target
    # ------------------------------------------------------------------ #
    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        metrics_rows = list(reader)
        metrics_fieldnames = reader.fieldnames or []

    if not metrics_rows:
        logger.error("metrics.csv is empty")
        sys.exit(1)

    # Build lookup: (model_id, attack_id) → {lambda: row}
    metrics_by_key: dict[tuple[str, str], dict[int, dict]] = {}
    for row in metrics_rows:
        key = (row["model_id"], row["attack_id"])
        metrics_by_key.setdefault(key, {})[int(row["lambda"])] = row

    # ------------------------------------------------------------------ #
    # Discover JSONL files and compute costs
    # ------------------------------------------------------------------ #
    result_files = discover_results(results_dir)
    if not result_files:
        logger.error(f"No results.jsonl files found under {results_dir}")
        sys.exit(1)

    logger.info(f"Found {len(result_files)} (model, attack) result sets")

    cost_lookup: dict[tuple[str, str], dict[int, dict]] = {}
    cat_cost_lookup: dict[tuple[str, str, str], dict[int, dict]] = {}

    for (model_id, attack_id), path in sorted(result_files.items()):
        key = (model_id, attack_id)
        if key not in metrics_by_key:
            logger.warning(f"  Skipping {model_id}/{attack_id}: not in metrics.csv")
            continue

        records: list[TrialRecord] = list(read_jsonl(path))
        if not records:
            logger.warning(f"  Empty file: {path}")
            continue

        pressure_levels = sorted(metrics_by_key[key].keys())
        costs = aggregate_costs(
            records, pressure_levels,
            judge_model_id=args.judge_model,
            gcg_backward_mult=args.gcg_backward_mult,
        )

        cost_lookup[key] = costs
        max_lam = max(pressure_levels)
        top = costs.get(max_lam, {})
        logger.info(
            f"  {model_id}/{attack_id}: "
            f"target={top.get('mean_target_tokens', 0):.0f}tok  "
            f"judge={top.get('mean_judge_tokens', 0):.0f}tok  "
            f"total={top.get('mean_total_tokens', 0):.0f}tok  "
            f"TFLOPs={top.get('mean_total_tflops', 0):.4f}  (N={len(records)})"
        )

        # Per-category costs — reuse the already-loaded records
        by_cat: dict[str, list[TrialRecord]] = defaultdict(list)
        for rec in records:
            by_cat[rec.category].append(rec)
        for cat, cat_records in by_cat.items():
            cat_costs = aggregate_costs(
                cat_records, pressure_levels,
                judge_model_id=args.judge_model,
                gcg_backward_mult=args.gcg_backward_mult,
            )
            cat_cost_lookup[(model_id, attack_id, cat)] = cat_costs

    # ------------------------------------------------------------------ #
    # Join and write output CSV
    # ------------------------------------------------------------------ #
    out_fieldnames = list(metrics_fieldnames) + COST_COLS
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()

        for row in metrics_rows:
            key = (row["model_id"], row["attack_id"])
            lam = int(row["lambda"])
            costs = cost_lookup.get(key, {}).get(lam, {})
            out_row = dict(row)
            for col in COST_COLS:
                out_row[col] = costs.get(col, "")
            writer.writerow(out_row)

    print(f"\nCost metrics written to: {output_path}")
    print(f"Columns added: {COST_COLS}")

    cost_df = pd.read_csv(output_path)
    summary_df = compute_cost_summary_metrics(cost_df)
    summary_path = output_path.parent / "cost_summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False, na_rep="")
    print(f"Cost summary metrics written to: {summary_path}")

    # ------------------------------------------------------------------ #
    # Per-category: join with metrics_by_category.csv and write outputs
    # ------------------------------------------------------------------ #
    cat_metrics_path = metrics_path.parent / "metrics_by_category.csv"
    if not cat_metrics_path.exists():
        logger.warning(f"metrics_by_category.csv not found at {cat_metrics_path}; skipping per-category cost output")
    else:
        with open(cat_metrics_path, newline="") as f:
            reader = csv.DictReader(f)
            cat_rows = list(reader)
            cat_fieldnames = reader.fieldnames or []

        cat_out_fieldnames = list(cat_fieldnames) + COST_COLS
        cat_output_path = output_path.parent / "cost_metrics_by_category.csv"

        with open(cat_output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cat_out_fieldnames)
            writer.writeheader()
            for row in cat_rows:
                key = (row["model_id"], row["attack_id"], row["category"])
                lam = int(row["lambda"])
                costs = cat_cost_lookup.get(key, {}).get(lam, {})
                out_row = dict(row)
                for col in COST_COLS:
                    out_row[col] = costs.get(col, "")
                writer.writerow(out_row)

        print(f"Per-category cost metrics written to: {cat_output_path}")

        cat_cost_df = pd.read_csv(cat_output_path)
        cat_summary_df = compute_cost_summary_by_category(cat_cost_df)
        cat_summary_path = output_path.parent / "cost_summary_by_category.csv"
        cat_summary_df.to_csv(cat_summary_path, index=False, na_rep="")
        print(f"Per-category cost summary written to: {cat_summary_path}")


if __name__ == "__main__":
    main()
