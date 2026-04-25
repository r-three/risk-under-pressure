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

    # Adjust GCG backward-pass multiplier (default 3.0 = forward+backward):
    python scripts/compute_attack_costs.py \\
        --results-dir path/to/results \\
        --metrics-csv path/to/metrics.csv \\
        --gcg-backward-mult 3.0
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prisk.metrics.cost_mapper import aggregate_costs
from prisk.utils.io import TrialRecord, read_jsonl
from prisk.utils.logging import get_logger

logger = get_logger("compute_attack_costs")

COST_COLS = [
    "mean_target_tokens",
    "mean_total_tokens",
    "mean_target_tflops",
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
                   help="FLOPs multiplier for GCG gradient step vs single forward pass "
                        "(default 3.0 = forward+backward; use ~43 to include 128 candidates)")
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
        costs = aggregate_costs(records, pressure_levels, gcg_backward_mult=args.gcg_backward_mult)

        cost_lookup[key] = costs
        max_lam = max(pressure_levels)
        etfs_tok = costs.get(max_lam, {}).get("mean_total_tokens", 0)
        etfs_tfl = costs.get(max_lam, {}).get("mean_total_tflops", 0)
        logger.info(
            f"  {model_id}/{attack_id}: "
            f"ETFS tokens={etfs_tok:.0f}  TFLOPs={etfs_tfl:.4f}  (N={len(records)})"
        )

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


if __name__ == "__main__":
    main()
