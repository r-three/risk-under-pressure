#!/usr/bin/env python3
"""
run_evaluation.py — Phase 2: Compute pRisk-Pressure metrics from inference results.

Reads TrialRecord JSONL files written by run_inference.py, computes the
risk-pressure curve R̂(M, λ) and summary metrics (AURC, ΔR, λ*) with
bootstrap confidence intervals.

No model calls are made — this is purely a post-processing step and runs in seconds.

Usage:
    python scripts/run_evaluation.py \\
        --results-dir outputs/pressure_sensitivity \\
        --pressure-levels 0 1 3 5 10 \\
        --tau 0.5 \\
        --output outputs/pressure_sensitivity/metrics.json

    # Print results table to stdout:
    python scripts/run_evaluation.py \\
        --results-dir outputs/pressure_sensitivity \\
        --print-table

    # Export to CSV (for plotting):
    python scripts/run_evaluation.py \\
        --results-dir outputs/pressure_sensitivity \\
        --format csv \\
        --output outputs/pressure_sensitivity/metrics.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prisk.metrics import compute_all_metrics, format_metrics_table
from prisk.utils.io import TrialRecord, read_jsonl
from prisk.utils.logging import get_logger

logger = get_logger("run_evaluation")


def parse_args():
    p = argparse.ArgumentParser(description="Compute pRisk-Pressure metrics (Phase 2)")
    p.add_argument("--results-dir", required=True,
                   help="Directory containing inference results (model/attack/results.jsonl)")
    p.add_argument("--pressure-levels", nargs="+", type=int, default=None,
                   help="Pressure levels λ to evaluate (default: infer from data)")
    p.add_argument("--tau", type=float, default=0.5,
                   help="Risk tolerance τ for λ* computation (default: 0.5)")
    p.add_argument("--n-bootstrap", type=int, default=1000,
                   help="Number of bootstrap resamples (default: 1000)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    p.add_argument("--output", default=None,
                   help="Output file path (default: <results-dir>/metrics.json)")
    p.add_argument("--format", choices=["json", "csv"], default="json",
                   help="Output format (default: json)")
    p.add_argument("--print-table", action="store_true",
                   help="Print formatted metrics table to stdout")
    return p.parse_args()


def discover_results(results_dir: Path) -> dict[tuple[str, str], Path]:
    """
    Discover all results.jsonl files under results_dir.
    Returns dict mapping (model_id, attack_id) → file path.
    """
    found = {}
    for f in results_dir.rglob("results.jsonl"):
        # Expect structure: results_dir / model_id / attack_id / results.jsonl
        attack_id = f.parent.name
        model_id = f.parent.parent.name
        found[(model_id, attack_id)] = f
    return found


def infer_pressure_levels(records: list[TrialRecord]) -> list[int]:
    """Infer pressure levels from max budget in records: 0, 1, ..., lambda_max."""
    if not records:
        return [0]
    lambda_max = max(r.budget for r in records)
    # Default levels from the paper
    defaults = [lam for lam in [0, 1, 3, 5, 10, 15, 20, 25, 50] if lam <= lambda_max]
    if lambda_max not in defaults:
        defaults.append(lambda_max)
    return sorted(defaults)


def write_csv(results: dict, output_path: Path) -> None:
    """Write risk curve data as CSV for plotting."""
    rows = []
    for (model_id, attack_id), m in results.items():
        curve = m["risk_curve"]
        curve_ci = m.get("risk_curve_ci", {})
        for lam_str, risk in curve.items():
            lam = int(lam_str)
            ci = curve_ci.get(str(lam), [risk, risk, risk])
            rows.append({
                "model_id": model_id,
                "attack_id": attack_id,
                "lambda": lam,
                "risk": risk,
                "risk_lower": ci[1],
                "risk_upper": ci[2],
                "aurc": m["aurc"],
                "delta_r": m["delta_r"],
                "lambda_star": m.get("lambda_star"),
                "n_prompts": m["n_prompts"],
            })

    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    result_files = discover_results(results_dir)
    if not result_files:
        logger.error(f"No results.jsonl files found under {results_dir}")
        sys.exit(1)

    logger.info(f"Found {len(result_files)} (model, attack) result sets")

    all_metrics: dict[str, dict[str, dict]] = {}
    nested_results: dict[str, dict[str, dict]] = {}  # for format_metrics_table

    for (model_id, attack_id), path in sorted(result_files.items()):
        records = list(read_jsonl(path))
        if not records:
            logger.warning(f"Empty results file: {path}")
            continue

        pressure_levels = args.pressure_levels or infer_pressure_levels(records)

        # Clamp pressure levels to max budget in records
        max_budget = max(r.budget for r in records)
        pressure_levels = [lam for lam in pressure_levels if lam <= max_budget]

        logger.info(
            f"[{model_id}/{attack_id}] {len(records)} records | "
            f"pressure_levels={pressure_levels}"
        )

        metrics = compute_all_metrics(
            records=records,
            pressure_levels=pressure_levels,
            tau=args.tau,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )

        key = f"{model_id}/{attack_id}"
        all_metrics[key] = metrics
        nested_results.setdefault(model_id, {})[attack_id] = metrics

        logger.info(
            f"  AURC={metrics['aurc']:.4f} "
            f"ΔR={metrics['delta_r']:.4f} "
            f"λ*={metrics['lambda_star']} "
            f"(N={metrics['n_prompts']})"
        )

    if not all_metrics:
        logger.error("No metrics computed. Check that results files are non-empty.")
        sys.exit(1)

    # Print table
    if args.print_table:
        print("\n" + format_metrics_table(nested_results))

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    elif args.format == "csv":
        output_path = results_dir / "metrics.csv"
    else:
        output_path = results_dir / "metrics.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "csv":
        write_csv(nested_results, output_path)
        logger.info(f"CSV metrics written to {output_path}")
    else:
        with open(output_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"JSON metrics written to {output_path}")

    # Always print a brief summary to stdout
    print(f"\n{'='*60}")
    print(f"pRisk-Pressure Evaluation Summary")
    print(f"{'='*60}")
    for key, m in sorted(all_metrics.items()):
        ls = str(m["lambda_star"]) if m["lambda_star"] is not None else "N/A"
        aurc_lo, aurc_hi = m["aurc_ci"]
        print(f"  {key}")
        print(f"    AURC:    {m['aurc']:.4f}  [{aurc_lo:.4f}, {aurc_hi:.4f}]")
        print(f"    ΔR:      {m['delta_r']:.4f}")
        print(f"    λ*:      {ls}  (τ={args.tau})")
        print(f"    N:       {m['n_prompts']}")
        print()


if __name__ == "__main__":
    main()
