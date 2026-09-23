"""Reward oracle for the per-prompt GRPO attack.

See `rup.training.rl_env.AttackEnv` (judge + perplexity-shaping reward). The per-prompt GRPO
optimization loop itself lives in `rup.pipeline.rl_refinement`.
"""

from .rl_env import AttackEnv, RewardConfig

__all__ = ["AttackEnv", "RewardConfig"]
