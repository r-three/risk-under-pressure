#!/usr/bin/env python3
"""
compute_template_ordering.py — Empirically derive the JailBroken template ordering.

Runs each of the 8 JailBroken templates independently (budget=1, no cycling) across
a set of models and prompts, computes per-template ASR, and sorts by ascending ASR.
The resulting order can replace the hand-crafted _DEFAULT_TEMPLATES in jailbroken_attack.py.

Usage:
    python scripts/compute_template_ordering.py \\
        --experiment configs/experiments/template_ordering.yaml

    # Resume an interrupted run:
    python scripts/compute_template_ordering.py \\
        --experiment configs/experiments/template_ordering.yaml \\
        --resume

    # Re-use existing results (skip inference, just re-aggregate):
    python scripts/compute_template_ordering.py \\
        --experiment configs/experiments/template_ordering.yaml \\
        --aggregate-only
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from prisk.attacks.jailbroken_attack import TEMPLATE_REGISTRY, SingleTemplateAttack
from prisk.benchmarks import get_benchmark
from prisk.judges import get_judge
from prisk.metrics.risk_curve import compute_risk_at_pressure
from prisk.models.factory import load_model
from prisk.pipeline import run_trial
from prisk.utils.config import ExperimentConfig, load_model_config
from prisk.utils.io import append_jsonl, load_completed_ids, read_jsonl
from prisk.utils.logging import get_logger

load_dotenv()
logger = get_logger("compute_template_ordering")


def parse_args():
    p = argparse.ArgumentParser(description="Empirically order JailBroken templates by ASR")
    p.add_argument("--experiment", required=True, help="Path to experiment YAML config")
    p.add_argument("--configs-dir", default="configs", help="Root directory for configs")
    p.add_argument("--resume", action="store_true",
                   help="Skip (model, template, prompt) triples already in results files")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip inference; re-read existing results and recompute ordering")
    return p.parse_args()


def run_inference(config: ExperimentConfig, output_dir: Path, configs_dir: Path, resume: bool):
    """Run each seed × template × model × prompt and save TrialRecords to JSONL."""

    seeds = config.seeds if config.seeds else [config.seed]

    judge_model_config = load_model_config(config.judge_model, configs_dir)
    judge_model = load_model(judge_model_config)
    judge = get_judge("llm", model=judge_model)
    logger.info(f"Judge: {config.judge_model}")

    n_templates = len(TEMPLATE_REGISTRY)
    n_models = len(config.models)
    logger.info(f"Templates: {n_templates}  |  Models: {n_models}  |  Seeds: {seeds}  |  Budget: 1 (single-shot)")

    for seed in seeds:
        logger.info(f"=== Seed {seed} ===")

        benchmark = get_benchmark(config.benchmark)
        prompts = benchmark.load(n=config.n_prompts, seed=seed, categories=config.categories)
        logger.info(f"Loaded {len(prompts)} prompts from {config.benchmark} (seed={seed})")

        for model_name in config.models:
            model_config = load_model_config(model_name, configs_dir)
            target_model = load_model(model_config)
            logger.info(f"Loaded model: {model_config.model_id}")

            for template_name, template_fn in TEMPLATE_REGISTRY:
                attack = SingleTemplateAttack(template_fn, template_name)
                # Output path: outputs/{experiment}/{benchmark}/{model_id}_seed{seed}/{attack_id}/results.jsonl
                model_seed_dir = f"{model_config.model_id}_seed{seed}"
                out_path = output_dir / config.benchmark / model_seed_dir / attack.attack_id / "results.jsonl"

                if resume:
                    done_ids = load_completed_ids(out_path)
                else:
                    done_ids = set()
                    if out_path.exists():
                        out_path.unlink()

                remaining = [p for p in prompts if p.prompt_id not in done_ids]
                desc = f"{model_seed_dir}/{attack.attack_id}"

                if done_ids:
                    logger.info(f"[{desc}] Resuming: {len(done_ids)} done, {len(remaining)} remaining")

                if not remaining:
                    logger.info(f"[{desc}] Already complete.")
                    continue

                for prompt in tqdm(remaining, desc=desc, unit="prompt"):
                    record = run_trial(
                        base_prompt=prompt.text,
                        prompt_id=prompt.prompt_id,
                        behavior=prompt.text,
                        category=prompt.category,
                        source=prompt.source,
                        model=target_model,
                        judge=judge,
                        attack=attack,
                        budget=1,  # single shot — templates are stateless
                    )
                    append_jsonl(record, out_path)


def aggregate(config: ExperimentConfig, output_dir: Path) -> dict:
    """
    Read all results, compute per-(template, model, seed) ASR, average across
    models and seeds, and return a sorted ordering dict.
    """
    seeds = config.seeds if config.seeds else [config.seed]

    # asr_runs[template_name][(model_id, seed)] = ASR
    asr_runs: dict = defaultdict(dict)

    for seed in seeds:
        for model_name in config.models:
            for template_name, _ in TEMPLATE_REGISTRY:
                attack_id = f"jailbroken_{template_name}"
                results_path = output_dir / config.benchmark / f"{model_name}_seed{seed}" / attack_id / "results.jsonl"

                if not results_path.exists():
                    # Fall back: scan for any matching directory (handles model_id vs config name mismatch)
                    found = list(output_dir.glob(f"{config.benchmark}/*_seed{seed}/{attack_id}/results.jsonl"))
                    if found:
                        results_path = found[0]
                    else:
                        logger.warning(f"Missing results: {results_path}")
                        continue

                records = list(read_jsonl(results_path))
                if not records:
                    logger.warning(f"Empty results file: {results_path}")
                    continue

                model_id = records[0].model_id
                asr = compute_risk_at_pressure(records, pressure=1)
                asr_runs[template_name][(model_id, seed)] = asr
                logger.info(
                    f"  {template_name:25s} | {model_id:30s} | seed={seed} | ASR={asr:.3f}  (n={len(records)})"
                )

    # Average ASR across all (model, seed) runs for each template
    mean_asr = {}
    for template_name, _ in TEMPLATE_REGISTRY:
        runs = asr_runs.get(template_name, {})
        mean_asr[template_name] = sum(runs.values()) / len(runs) if runs else float("nan")

    ordered = sorted(mean_asr.items(), key=lambda x: x[1])

    # Build per-model summary averaged over seeds
    per_model_asr: dict = defaultdict(dict)
    for template_name, runs in asr_runs.items():
        by_model: dict = defaultdict(list)
        for (model_id, _seed), asr in runs.items():
            by_model[model_id].append(asr)
        for model_id, asrs in by_model.items():
            per_model_asr[template_name][model_id] = round(sum(asrs) / len(asrs), 4)

    return {
        "ordered_templates": [
            {"name": name, "mean_asr": round(asr, 4)} for name, asr in ordered
        ],
        "per_model_asr": dict(per_model_asr),
        "n_prompts": config.n_prompts,
        "models": config.models,
        "benchmark": config.benchmark,
        "seeds": seeds,
    }


def print_table(result: dict):
    print("\n" + "=" * 55)
    print(f"{'Template':<25}  {'Mean ASR':>8}  {'Rank':>4}")
    print("-" * 55)
    for rank, entry in enumerate(result["ordered_templates"], 1):
        print(f"{entry['name']:<25}  {entry['mean_asr']:>8.3f}  {rank:>4}")
    print("=" * 55)
    print("\nRecommended _DEFAULT_TEMPLATES order (paste into jailbroken_attack.py):")
    names = [e["name"] for e in result["ordered_templates"]]
    for name in names:
        print(f"  _template_{name},")


def main():
    args = parse_args()

    with open(args.experiment) as f:
        exp_data = yaml.safe_load(f)
    config = ExperimentConfig(**exp_data)

    output_dir = Path(config.output_dir)
    configs_dir = Path(args.configs_dir)

    logger.info(f"Experiment: {config.name}")
    logger.info(f"Output dir: {output_dir}")

    if not args.aggregate_only:
        run_inference(config, output_dir, configs_dir, resume=args.resume)

    logger.info("Aggregating results...")
    result = aggregate(config, output_dir)

    ordering_path = output_dir / "template_ordering.json"
    ordering_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ordering_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Ordering saved to: {ordering_path}")

    print_table(result)


if __name__ == "__main__":
    main()
