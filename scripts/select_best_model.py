"""Select the model with the lowest mean AURC from a set of candidates.

Usage:
    python scripts/select_best_model.py \
        --metrics-dir /path/to/harmbench \
        --models qwen2.5-0.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct

Prints the name of the best model (lowest mean AURC) to stdout.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def mean_aurc(metrics_dir: Path, model: str) -> float | None:
    csv_path = metrics_dir / model / "metrics_summary.csv"
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found, skipping {model}", file=sys.stderr)
        return None
    df = pd.read_csv(csv_path)
    if "aurc" not in df.columns:
        print(f"WARNING: 'aurc' column missing in {csv_path}, skipping {model}", file=sys.stderr)
        return None
    # One aurc value per (model_id, attack_id) pair — deduplicate before averaging
    per_attack = df.drop_duplicates(subset=["model_id", "attack_id"])["aurc"]
    return float(per_attack.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--models", required=True, nargs="+")
    args = parser.parse_args()

    scores = {}
    for model in args.models:
        score = mean_aurc(args.metrics_dir, model)
        if score is not None:
            scores[model] = score

    if not scores:
        print("ERROR: no valid models found", file=sys.stderr)
        sys.exit(1)

    best = min(scores, key=scores.__getitem__)
    print(best)


if __name__ == "__main__":
    main()
