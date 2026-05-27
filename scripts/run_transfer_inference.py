#!/usr/bin/env python3
"""
run_transfer_inference.py — Transfer attack evaluation.

Loads pre-computed attack trajectories from a source model and replays them against
one or more target models. Useful for measuring how well white-box attacks (e.g. GCG)
transfer in a black-box setting.

Output format is identical to run_inference.py so existing run_evaluation.py and
plot_results.py pipelines work unchanged. The attack folder in the output path encodes
both the source attack and source model:
    {output_dir}/{benchmark}/{target_model}_seed{seed}/
        transfer_{source_attack}_from_{source_model}/results.jsonl

Usage:
    python scripts/run_transfer_inference.py \\
        --experiment configs/experiments/HB_tulu3_8b_sft.yaml \\
        --source-results-dir $SOURCE/harmbench \\
        --source-model tulu3-8b-sft \\
        --source-attack gcg \\
        --target-models tulu3_8b_base qwen2.5_7b \\
        --output-dir $OUTPUT \\
        --resume

    # Quick smoke test (5 prompts, 1 seed):
    python scripts/run_transfer_inference.py \\
        --experiment configs/experiments/HB_tulu3_8b_sft.yaml \\
        --source-results-dir $SOURCE/harmbench \\
        --source-model tulu3-8b-sft \\
        --target-models tulu3_8b_base \\
        --seeds 1997 --n-prompts 5 \\
        --output-dir /tmp/transfer_test
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from rup.attacks.transfer_attack import TransferAttack
from rup.benchmarks import get_benchmark
from rup.judges import get_judge
from rup.models.factory import load_model
from rup.pipeline import run_trial
from rup.utils.config import ExperimentConfig, load_model_config
from rup.utils.io import append_jsonl, load_completed_ids, read_jsonl
from rup.utils.logging import get_logger

load_dotenv()
logger = get_logger("run_transfer_inference")


def parse_args():
    p = argparse.ArgumentParser(description="Transfer attack inference (replay source trajectories on target model)")
    p.add_argument("--experiment", required=True,
                   help="Experiment YAML (provides benchmark, n_prompts, seeds, judge_model, lambda_max)")
    p.add_argument("--source-results-dir", required=True, type=Path,
                   help="Benchmark-level directory containing {source_model}_seed{seed}/ subdirs")
    p.add_argument("--source-model", required=True,
                   help="model_id of the source model (e.g. tulu3-8b-sft)")
    p.add_argument("--source-attack", default="gcg",
                   help="Attack whose trajectories to transfer (default: gcg)")
    p.add_argument("--target-models", nargs="+", required=True,
                   help="Config names of target models (e.g. tulu3_8b_base qwen2.5_7b)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Root output directory (benchmark/ subdir is appended automatically)")
    p.add_argument("--seeds", type=int, nargs="+",
                   help="Override seeds from experiment config")
    p.add_argument("--n-prompts", type=int,
                   help="Override n_prompts from experiment config")
    p.add_argument("--lambda-max", type=int,
                   help="Override lambda_max from experiment config")
    p.add_argument("--configs-dir", default="configs",
                   help="Root directory for model/attack configs")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompt_ids already present in output files")
    return p.parse_args()


def load_trajectories(source_jsonl: Path) -> dict[str, list[dict]]:
    """Read source results.jsonl and return prompt_id → list-of-step-dicts."""
    trajectories = {}
    for record in read_jsonl(source_jsonl):
        trajectories[record.prompt_id] = [asdict(s) for s in record.steps]
    return trajectories


def main():
    args = parse_args()

    with open(args.experiment) as f:
        config = ExperimentConfig(**yaml.safe_load(f))

    if args.seeds:
        config.seeds = args.seeds
    if args.n_prompts:
        config.n_prompts = args.n_prompts
    if args.lambda_max:
        config.lambda_max = args.lambda_max

    seeds = config.seeds if config.seeds else [config.seed]
    configs_dir = Path(args.configs_dir)
    transfer_attack_id = f"transfer_{args.source_attack}_from_{args.source_model}"

    logger.info(f"Experiment:      {config.name}")
    logger.info(f"Benchmark:       {config.benchmark} | n_prompts={config.n_prompts}")
    logger.info(f"Source model:    {args.source_model}")
    logger.info(f"Source attack:   {args.source_attack}")
    logger.info(f"Transfer id:     {transfer_attack_id}")
    logger.info(f"Target models:   {args.target_models}")
    logger.info(f"λ_max:           {config.lambda_max}")
    logger.info(f"Seeds:           {seeds}")
    logger.info(f"Output:          {args.output_dir}")

    judge_model_config = load_model_config(config.judge_model, configs_dir)
    judge_model = load_model(judge_model_config)
    judge = get_judge("llm", model=judge_model)
    logger.info(f"Judge:           {config.judge_model}")

    for seed in seeds:
        logger.info(f"=== Seed {seed} ===")

        source_jsonl = (
            args.source_results_dir
            / f"{args.source_model}_seed{seed}"
            / args.source_attack
            / "results.jsonl"
        )

        if not source_jsonl.exists():
            logger.warning(f"Source JSONL not found, skipping seed {seed}: {source_jsonl}")
            continue

        trajectories = load_trajectories(source_jsonl)
        logger.info(f"Loaded {len(trajectories)} trajectories from {source_jsonl}")

        benchmark = get_benchmark(config.benchmark)
        prompts = benchmark.load(n=config.n_prompts, seed=seed, categories=config.categories)
        logger.info(f"Loaded {len(prompts)} prompts (seed={seed})")

        missing = [p for p in prompts if p.prompt_id not in trajectories]
        if missing:
            logger.warning(
                f"{len(missing)} prompt(s) have no source trajectory and will be skipped "
                f"(source had {len(trajectories)} records, benchmark has {len(prompts)})"
            )

        for model_name in args.target_models:
            model_config = load_model_config(model_name, configs_dir)
            target_model = load_model(model_config)

            attack = TransferAttack(transfer_attack_id, trajectories)

            out_path = (
                args.output_dir
                / config.benchmark
                / f"{model_config.model_id}_seed{seed}"
                / transfer_attack_id
                / "results.jsonl"
            )

            if args.resume:
                done_ids = load_completed_ids(out_path)
            else:
                done_ids = set()
                if out_path.exists():
                    out_path.unlink()
                    logger.info(f"Cleared existing results: {out_path}")

            remaining = [
                p for p in prompts
                if p.prompt_id not in done_ids and p.prompt_id in trajectories
            ]

            desc = f"{model_config.model_id}_seed{seed}/{transfer_attack_id}"
            if done_ids:
                logger.info(f"[{desc}] Resuming: {len(done_ids)} done, {len(remaining)} remaining")

            if not remaining:
                logger.info(f"[{desc}] All prompts complete.")
                continue

            for prompt in tqdm(remaining, desc=desc, unit="prompt"):
                traj_len = attack.prepare(prompt.prompt_id)
                if not traj_len:
                    continue
                # Cap budget at source trajectory length: the target model is only
                # evaluated on prompts the source actually produced. Steps beyond
                # the trajectory length are not available, so success stays at 0.
                record = run_trial(
                    base_prompt=prompt.text,
                    prompt_id=prompt.prompt_id,
                    behavior=prompt.text,
                    category=prompt.category,
                    source=prompt.source,
                    model=target_model,
                    judge=judge,
                    attack=attack,
                    budget=min(config.lambda_max, traj_len),
                )
                append_jsonl(record, out_path)

            logger.info(f"[{desc}] Done. Results: {out_path}")

    logger.info("Transfer inference complete.")


if __name__ == "__main__":
    main()
