#!/usr/bin/env python3
"""
run_inference.py — Phase 1: Run adversarial attacks and collect model responses.

Implements Algorithm 1 (Budgeted Iterative Refinement) across all
(model, attack, prompt) combinations specified in an experiment config.

Results are written to JSONL, one record per prompt, immediately after each
trial completes (crash-resilient). Supports resuming interrupted runs.

Usage:
    python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity.yaml

    # Override specific dimensions:
    python scripts/run_inference.py \\
        --experiment configs/experiments/pressure_sensitivity.yaml \\
        --model qwen2.5_7b \\
        --attack pair \\
        --lambda-max 5 \\
        --n-prompts 20 \\
        --resume
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from rup.attacks.factory import load_attack
from rup.benchmarks import get_benchmark
from rup.judges import get_judge
from rup.models.factory import load_model
from rup.pipeline import run_trial
from rup.utils.config import ExperimentConfig, load_attack_config, load_model_config
from rup.utils.io import append_jsonl, load_completed_ids
from rup.utils.logging import get_logger

load_dotenv()
logger = get_logger("run_inference")


def parse_args():
    p = argparse.ArgumentParser(description="Run adversarial inference (Phase 1)")
    p.add_argument("--experiment", required=True, help="Path to experiment YAML config")
    p.add_argument("--model", help="Override: run only this model (config name, e.g. qwen2.5_7b)")
    p.add_argument("--attack", help="Override: run only this attack (e.g. pair)")
    p.add_argument("--attacks", nargs="+", help="Override: list of attacks (e.g. --attacks pair jailbroken gcg)")
    p.add_argument("--lambda-max", type=int, help="Override: maximum pressure budget")
    p.add_argument("--n-prompts", type=int, help="Override: number of prompts to use")
    p.add_argument("--seeds", type=int, nargs="+", help="Override: list of seeds (e.g. --seeds 42 123 456)")
    p.add_argument("--benchmark", help="Override: benchmark (harmbench or jailbreakbench)")
    p.add_argument("--output-dir", help="Override: root directory for results (e.g. $SCRATCH)")
    p.add_argument("--configs-dir", default="configs", help="Root directory for configs")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts already in results files")
    return p.parse_args()


def main():
    args = parse_args()

    # Load experiment config
    with open(args.experiment) as f:
        exp_data = yaml.safe_load(f)
    config = ExperimentConfig(**exp_data)

    # Apply CLI overrides
    if args.model:
        config.models = [args.model]
    if args.attacks:
        config.attacks = args.attacks
    elif args.attack:
        config.attacks = [args.attack]
    if args.lambda_max:
        config.lambda_max = args.lambda_max
    if args.n_prompts:
        config.n_prompts = args.n_prompts
    if args.seeds:
        config.seeds = args.seeds
    if args.benchmark:
        config.benchmark = args.benchmark
    if args.output_dir:
        config.output_dir = args.output_dir

    output_dir = Path(config.output_dir)
    configs_dir = Path(args.configs_dir)

    seeds = config.seeds if config.seeds else [config.seed]

    logger.info(f"Experiment: {config.name}")
    logger.info(f"Benchmark: {config.benchmark} | n_prompts={config.n_prompts}")
    logger.info(f"Models: {config.models}")
    logger.info(f"Attacks: {config.attacks}")
    logger.info(f"λ_max: {config.lambda_max}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Output: {output_dir}")

    # Load judge model (shared across all seeds / model / attack combos)
    judge_model_config = load_model_config(config.judge_model, configs_dir)
    judge_model = load_model(judge_model_config)
    judge = get_judge("llm", model=judge_model)
    logger.info(f"Judge: {config.judge_model}")

    # Outer loop: seed × model × attack
    for seed in seeds:
        logger.info(f"=== Seed {seed} ===")

        benchmark = get_benchmark(config.benchmark)
        prompts = benchmark.load(
            n=config.n_prompts,
            seed=seed,
            categories=config.categories,
        )
        logger.info(f"Loaded {len(prompts)} prompts from {config.benchmark} (seed={seed})")

        for model_name in config.models:
            model_config = load_model_config(model_name, configs_dir)
            target_model = load_model(model_config)

            for attack_name in config.attacks:
                attack_config = load_attack_config(attack_name, configs_dir)

                # Resolve list of attacker models to iterate over
                if config.attacker_models:
                    attacker_names = config.attacker_models
                else:
                    single = config.attacker_model or attack_config.attacker_model
                    attacker_names = [single]  # may be None (no attacker needed)

                for attacker_name in attacker_names:
                    # Load attacker model if specified
                    attacker_model = None
                    if attacker_name:
                        att_model_config = load_model_config(attacker_name, configs_dir)
                        attacker_model = load_model(att_model_config)

                    attack = load_attack(
                        attack_config,
                        attacker_model=attacker_model,
                        target_model=target_model,
                        seed=seed,
                    )

                    # Encode attacker name in folder when comparing multiple attackers
                    if config.attacker_models:
                        folder_id = f"{attack_config.attack_id}__{attacker_name}"
                    else:
                        folder_id = attack_config.attack_id

                    # Output path: outputs/{benchmark}/{model_id}/{seed}/{folder_id}/results.jsonl
                    out_path = output_dir / config.benchmark / model_config.model_id / str(seed) / folder_id / "results.jsonl"

                    # Determine completed IDs (resume) or clear the file (fresh run)
                    if args.resume:
                        done_ids = load_completed_ids(out_path)
                    else:
                        done_ids = set()
                        if out_path.exists():
                            out_path.unlink()
                            logger.info(f"Cleared existing results: {out_path}")
                    remaining = [p for p in prompts if p.prompt_id not in done_ids]

                    desc = f"{model_config.model_id}/{seed}/{folder_id}"

                    if done_ids:
                        logger.info(f"[{desc}] Resuming: {len(done_ids)} done, {len(remaining)} remaining")

                    if not remaining:
                        logger.info(f"[{desc}] All prompts complete.")
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
                            budget=config.lambda_max,
                        )
                        append_jsonl(record, out_path)

                    logger.info(f"[{desc}] Done. Results: {out_path}")

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
