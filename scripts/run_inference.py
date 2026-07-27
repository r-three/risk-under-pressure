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
import gc
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
from rup.pipeline import build_rl_attacker, run_prompt_rl, run_trial
from rup.pipeline.rl_refinement import GRPOAttackConfig
from rup.training.rl_env import RewardConfig
from rup.utils.config import ExperimentConfig, load_attack_config, load_model_config
from rup.utils.io import append_jsonl, load_completed_ids
from rup.utils.logging import get_logger

load_dotenv()
logger = get_logger("run_inference")


def _reward_from_extra(extra):
    fields = RewardConfig.__dataclass_fields__
    return RewardConfig(**{k: v for k, v in (extra or {}).items() if k in fields})


def _release_rl_attacker(rl_cache):
    """Drop the cached trainable attacker and give its GPU memory back."""
    if not rl_cache.pop("attacker", None):
        return
    rl_cache.pop("cfg", None)
    rl_cache.pop("reward", None)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _run_rl_for_model(config, target_model, judge, prompts, seed, output_dir, resume, rl_cache,
                      attack_config, configs_dir, attacker_name, folder_id):
    """RL attack: per-prompt GRPO (rup.pipeline.run_prompt_rl) instead of run_trial.

    The trainable attacker is expensive to build, so it is cached across models/seeds — but
    keyed on `attacker_name`, and only ONE is held at a time: RL attackers load in bf16 + LoRA
    (unquantized), so keeping two arms of an attacker ablation resident would double a ~15 GB
    footprint to save a reload the arm loop only benefits from once per seed anyway.
    Each prompt resets its LoRA adapter internally so prompts stay independent.
    """
    if rl_cache.get("attacker_name") != attacker_name:
        _release_rl_attacker(rl_cache)
        attacker_hf = load_model_config(attacker_name, configs_dir).hf_name
        cfg = GRPOAttackConfig.from_extra(attack_config.extra, attacker_hf=attacker_hf)
        rl_cache["attacker_name"] = attacker_name
        rl_cache["cfg"] = cfg
        rl_cache["reward"] = _reward_from_extra(attack_config.extra)
        rl_cache["attacker"] = build_rl_attacker(cfg)
    attacker, cfg, reward_cfg = rl_cache["attacker"], rl_cache["cfg"], rl_cache["reward"]

    out_path = output_dir / config.benchmark / target_model.model_id / str(seed) / folder_id / "results.jsonl"
    trace_path = out_path.parent / "training_trace.jsonl"
    if resume:
        done_ids = load_completed_ids(out_path)
    else:
        done_ids = set()
        if out_path.exists():
            out_path.unlink()
            logger.info(f"Cleared existing results: {out_path}")
        if trace_path.exists():
            trace_path.unlink()
            logger.info(f"Cleared existing training trace: {trace_path}")
    remaining = [p for p in prompts if p.prompt_id not in done_ids]
    desc = f"{target_model.model_id}/{seed}/{folder_id}"
    if done_ids:
        logger.info(f"[{desc}] Resuming: {len(done_ids)} done, {len(remaining)} remaining")
    if not remaining:
        logger.info(f"[{desc}] All prompts complete.")
        return

    for prompt in tqdm(remaining, desc=desc, unit="prompt"):
        record = run_prompt_rl(
            base_prompt=prompt.text,
            prompt_id=prompt.prompt_id,
            behavior=prompt.text,
            category=prompt.category,
            source=prompt.source,
            target=target_model,
            judge=judge,
            attacker=attacker,
            budget=config.lambda_max,
            cfg=cfg,
            reward_config=reward_cfg,
            trace_path=trace_path,
        )
        append_jsonl(record, out_path)
    logger.info(f"[{desc}] Done. Results: {out_path}")


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
    p.add_argument("--judge-model",
                   help="Override: safety judge model config name (e.g. gemma3_4b_it_judge). "
                        "Pair with a judge-specific --output-dir so runs under different "
                        "judges do not overwrite each other.")
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
    if args.judge_model:
        config.judge_model = args.judge_model
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

    # Trainable RL attacker is built lazily once and reused across models/seeds.
    rl_cache: dict = {}

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

                # RL attack: per-prompt GRPO path (own trainable attacker, no AttackPolicy).
                # It honours `attacker_models` like PAIR does, but takes its default from the
                # attack config's extra.base_attacker rather than attack_config.attacker_model.
                if attack_config.attack_id.lower() == "rl":
                    if config.attacker_models:
                        rl_attackers = config.attacker_models
                    else:
                        rl_attackers = [
                            config.attacker_model
                            or (attack_config.extra or {}).get("base_attacker", "qwen2.5_7b")
                        ]
                    for rl_attacker_name in rl_attackers:
                        # Folder carries the attacker only when arms are being compared, so
                        # single-attacker runs keep writing to the plain `rl/` tree they always had.
                        rl_folder = (
                            f"{attack_config.attack_id}__{rl_attacker_name}"
                            if config.attacker_models else attack_config.attack_id
                        )
                        _run_rl_for_model(
                            config, target_model, judge, prompts, seed,
                            output_dir, args.resume, rl_cache, attack_config, configs_dir,
                            attacker_name=rl_attacker_name, folder_id=rl_folder,
                        )
                    continue

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
