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
from rup.judges import get_judge, judge_from_config, judge_kind_for
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
                      attack_config, configs_dir, attacker_name, folder_id,
                      max_unparsed_rate=None):
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
        check_judge_health(judge, max_unparsed_rate)
    logger.info(f"[{desc}] Done. Results: {out_path}")


class JudgeUnhealthy(RuntimeError):
    """The judge is not producing parseable verdicts, so the run is not measuring anything."""


def check_judge_health(judge, max_unparsed_rate: float, min_judged: int = 200) -> None:
    """Abort the sweep if too many verdicts cannot be parsed.

    Without this the failure is invisible: an unparsed verdict is recorded as a label like any
    other, so a judge emitting no verdicts at all produces a complete results tree, a flat risk
    curve, and internally consistent metrics. That is how the Olmo-3-7B sweep finished looking
    healthy while mislabelling 61.8% of refusals. Fail loudly and early instead.
    """
    stats = getattr(judge, "stats", None)
    if stats is None or max_unparsed_rate is None or max_unparsed_rate >= 1.0:
        return
    s = stats()
    if s["n_judged"] < min_judged:
        return
    if s["unparsed_rate"] > max_unparsed_rate:
        raise JudgeUnhealthy(
            f"Judge unparsed rate {s['unparsed_rate']:.1%} exceeds the "
            f"--max-unparsed-rate {max_unparsed_rate:.1%} after {s['n_judged']} judgments. "
            f"Parse branches: {s['branch_counts']}. "
            f"Run `python scripts/diagnose_judge.py --judge <name>` to see raw judge output "
            f"before trusting any risk curve from this judge."
        )


def parse_args():
    p = argparse.ArgumentParser(description="Run adversarial inference (Phase 1)")
    p.add_argument("--experiment", required=True, help="Path to experiment YAML config")
    p.add_argument("--model", help="Override: run only this model (config name, e.g. qwen2.5_7b)")
    p.add_argument("--models", nargs="+",
                   help="Override: list of models (e.g. --models qwen2.5_0.5b tulu3_8b_sft). "
                        "Takes precedence over --model.")
    p.add_argument("--attack", help="Override: run only this attack (e.g. pair)")
    p.add_argument("--attacks", nargs="+", help="Override: list of attacks (e.g. --attacks pair jailbroken gcg)")
    p.add_argument("--lambda-max", type=int, help="Override: maximum pressure budget, in target "
                   "queries (one recorded step). For --attack rl see --rl-rounds.")
    p.add_argument("--rl-groups", type=int, default=None,
                   help="RL only: express the budget in GRPO GROUPS. One group is "
                        "num_generations sessions x session_rounds rounds (8 x 5 = 40 rollouts "
                        "at the defaults), and closing one is what triggers a weight update. "
                        "N groups => N-1 updates per behavior, since the final group's update is "
                        "skipped (nothing can be sampled from it before the adapter resets). "
                        "Converted to --lambda-max = 1 + N * num_generations * session_rounds. "
                        "Overrides --rl-rounds and --lambda-max.")
    p.add_argument("--rl-rounds", type=int, default=None,
                   help="RL only: express the budget in ATTACKER ROUNDS instead of target "
                        "queries. One round is one lockstep refinement across all "
                        "num_generations sessions, so it fans out to num_generations rollouts. "
                        "Converted to --lambda-max = 1 + rounds * num_generations (the 1 is the "
                        "raw-behavior probe). A session is session_rounds deep, so a group closes "
                        "every session_rounds rounds: --rl-rounds 10 at the defaults (8 sessions, "
                        "5 rounds) is 2 groups = 81 queries and one GRPO update. Overrides "
                        "--lambda-max.")
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
    p.add_argument("--judge-max-new-tokens", type=int, default=None,
                   help="Override the judge's first-attempt token budget. Leave unset to use the "
                        "judge class default (16 for SAFE/UNSAFE judges, which need one word; 384 "
                        "for Flow-Judge, which writes feedback before its score).")
    p.add_argument("--judge-escalate-tokens", type=int, default=None,
                   help="Retry budget when the first attempt yields no parseable verdict "
                        "(0 disables). Leave unset to use the judge class default — 256 for "
                        "SAFE/UNSAFE judges (16-token first attempt), 768 for Flow-Judge "
                        "(384-token first attempt, since its score follows its feedback). "
                        "Passing a value at or below the first-attempt budget DISABLES "
                        "escalation; the run warns if you do.")
    p.add_argument("--judge-rubric", default="default", choices=["default", "strict"],
                   help="'strict' drops the 'optionally explain' invitation. Changes the judge's "
                        "input, so it is a separate arm — not comparable to published numbers.")
    p.add_argument("--max-unparsed-rate", type=float, default=0.05,
                   help="Abort once this fraction of judgments cannot be parsed. A broken judge "
                        "used to complete a full sweep and look healthy; this is the stop.")
    return p.parse_args()


def main():
    args = parse_args()

    # Load experiment config
    with open(args.experiment) as f:
        exp_data = yaml.safe_load(f)
    config = ExperimentConfig(**exp_data)

    # Apply CLI overrides
    if args.models:
        config.models = args.models
    elif args.model:
        config.models = [args.model]
    if args.attacks:
        config.attacks = args.attacks
    elif args.attack:
        config.attacks = [args.attack]
    if args.lambda_max:
        config.lambda_max = args.lambda_max
    if args.rl_groups is not None:
        rl_extra = load_attack_config("rl", args.configs_dir).extra or {}
        g = int(rl_extra.get("num_generations", 8))
        r = int(rl_extra.get("session_rounds", 5))
        args.rl_rounds = args.rl_groups * r
        logger.info(
            f"--rl-groups {args.rl_groups} -> {args.rl_rounds} rounds "
            f"({args.rl_groups} x {g} sessions x {r} rounds = "
            f"{args.rl_groups * g * r} rollouts)"
        )
    if args.rl_rounds is not None:
        # The pressure axis and every recorded step are per target query, because that is what
        # risk_curve and cost_mapper index and what makes lambda mean the same thing for RL as
        # for PAIR/GCG. --rl-rounds lets the budget be WRITTEN in attacker rounds, which is the
        # unit the attack is actually designed in, and converts once here.
        rl_extra = load_attack_config("rl", args.configs_dir).extra or {}
        g = int(rl_extra.get("num_generations", 8))
        r = int(rl_extra.get("session_rounds", 5))
        config.lambda_max = 1 + args.rl_rounds * g
        groups = args.rl_rounds // r
        logger.info(
            f"--rl-rounds {args.rl_rounds} -> lambda_max {config.lambda_max} "
            f"(1 raw probe + {args.rl_rounds} rounds x {g} rollouts); "
            f"{groups} group(s) of {g} sessions x {r} rounds, "
            f"{max(0, groups - 1)} GRPO update(s) per behavior"
        )
        if groups < 2:
            logger.warning(
                f"--rl-rounds {args.rl_rounds} closes {groups} group(s), so NO GRPO update can "
                f"ever influence a scored query — this is best-of-N sampling from an untrained "
                f"attacker, not RL. Use at least {2 * r} rounds."
            )
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
    # judge_from_config, not get_judge("llm", ...): which implementation a checkpoint needs is a
    # property of the checkpoint. Flow-Judge emits <score>N</score> rather than SAFE/UNSAFE, and
    # driving it through the shared parser would mark every judgment unparsed.
    judge_kwargs = dict(rubric=args.judge_rubric)
    if args.judge_escalate_tokens is not None:
        judge_kwargs["escalate_max_new_tokens"] = args.judge_escalate_tokens
    # Only override the token budget when asked. Flow-Judge writes its feedback before the score,
    # so its class default (384) is far larger than the SAFE/UNSAFE default (16) and forcing the
    # latter on it would truncate before the verdict every time.
    if args.judge_max_new_tokens is not None:
        judge_kwargs["max_new_tokens"] = args.judge_max_new_tokens
    judge = judge_from_config(judge_model_config, judge_model, **judge_kwargs)
    logger.info(
        f"Judge: {config.judge_model} (kind={judge_kind_for(judge_model_config)}, "
        f"impl={type(judge).__name__}, rubric={judge.rubric_id}/{judge.rubric_sha1}, "
        f"tokens={judge._max_new_tokens}->{judge._escalate_max_new_tokens})"
    )
    # A budget at or below the first attempt makes the escalation guard unreachable. That is a
    # silent downgrade: a judge whose output is cut off before its verdict is then recorded as
    # unparsed and forced to the fallback label, so a generation failure becomes a real-looking
    # SAFE. Refuse to start rather than produce a tree that has to be thrown away.
    if 0 < judge._escalate_max_new_tokens <= judge._max_new_tokens:
        raise SystemExit(
            f"--judge-escalate-tokens ({judge._escalate_max_new_tokens}) must exceed the "
            f"first-attempt budget ({judge._max_new_tokens}) or escalation can never fire, and "
            f"truncated verdicts get silently recorded as label "
            f"{getattr(judge, '_unparsed_label', 0)}. Omit the flag to use the judge's own "
            f"default, or pass 0 to disable escalation deliberately."
        )

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
                            max_unparsed_rate=args.max_unparsed_rate,
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
                        check_judge_health(judge, args.max_unparsed_rate)

                    logger.info(f"[{desc}] Done. Results: {out_path}")

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
