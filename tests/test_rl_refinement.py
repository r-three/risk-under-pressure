"""Unit tests for the per-prompt GRPO runner's pure/config pieces (no torch/model loading)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rup.pipeline.rl_refinement import GRPOAttackConfig


def test_from_extra_picks_known_fields_and_ignores_others():
    extra = {
        "base_attacker": "qwen2.5_7b",       # not a GRPOAttackConfig field → ignored
        "perplexity_weight": 0.5,             # RewardConfig field → ignored here
        "num_generations": 16,
        "session_rounds": 5,
        "beta": 0.1,
        "max_completion_length": 128,
        "unknown_key": 123,                   # ignored
    }
    cfg = GRPOAttackConfig.from_extra(extra, attacker_hf="Qwen/Qwen2.5-7B-Instruct")
    assert cfg.base_attacker_hf == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.num_generations == 16
    assert cfg.session_rounds == 5
    assert cfg.beta == 0.1
    assert cfg.max_completion_length == 128


def test_from_extra_defaults_when_empty():
    cfg = GRPOAttackConfig.from_extra({}, attacker_hf="X")
    assert cfg.base_attacker_hf == "X"
    assert cfg.num_generations == 8       # default
    assert cfg.session_rounds == 5    # paper: 5 rounds per session


def test_explicit_base_attacker_hf_in_extra_wins_over_none():
    cfg = GRPOAttackConfig.from_extra({"base_attacker_hf": "Y"})
    assert cfg.base_attacker_hf == "Y"
