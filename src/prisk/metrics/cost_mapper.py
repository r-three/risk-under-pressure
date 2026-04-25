"""
Derive token-count and FLOP-count cost axes from TrialRecord data.

Token counts use chars/4 (standard English approximation; ~10-15% error for
LLM-tokenized text). FLOPs use the transformer approximation:
  forward_TFLOPs = 2 × N_B × L / 1000
where N_B = model size in billions of parameters, L = sequence length in tokens.

Per-attack overhead:
  jailbroken/jailbroken-*: one target forward pass per step
  pair:  one target forward + one attacker forward (estimated from logged text)
  gcg:   one target forward+backward (gradient); gcg_backward_mult ≈ 3.0
         (GCG candidate evaluation — num_turb_sample=128 — is excluded for
         clarity; multiply gcg_backward_mult by ~43 to include it)
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..utils.io import TrialRecord

# ---------------------------------------------------------------------------
# Model sizes
# ---------------------------------------------------------------------------

MODEL_PARAMS_B: Dict[str, float] = {
    "qwen2.5-0.5b-instruct": 0.494,
    "qwen2.5-3b-instruct":   3.09,
    "qwen2.5-7b-instruct":   7.62,
    "qwen3-4b-saferl":       4.0,
    "tulu2-7b-base":         6.74,
    "tulu2-7b-sft":          6.74,
    "tulu2-7b-dpo":          6.74,
    "tulu3-8b-base":         8.03,
    "tulu3-8b-sft":          8.03,
    "tulu3-8b-dpo":          8.03,
    "tulu3-8b-rlvr":         8.03,
}

# PAIR attacker (configs/attacks/pair.yaml → attacker_model: qwen2.5_7b)
_PAIR_ATTACKER_PARAMS_B: float = 7.62

# Fixed character overhead in the PAIR attacker prompt:
# _ATTACKER_SYSTEM_PROMPT (~330) + user template labels (~100) + goal (~100)
_PAIR_BOILERPLATE_CHARS: int = 530

CHARS_PER_TOKEN: int = 4


def _params_b(model_id: str) -> float:
    """Return model size in billions, stripping any _seed suffix."""
    key = model_id.split("_seed")[0]
    if key in MODEL_PARAMS_B:
        return MODEL_PARAMS_B[key]
    m = re.search(r"(\d+(?:\.\d+)?)b", key.lower())
    return float(m.group(1)) if m else 7.0


def _is_jailbroken(attack_id: str) -> bool:
    return attack_id == "jailbroken" or attack_id.startswith("jailbroken")


# ---------------------------------------------------------------------------
# Per-step cost
# ---------------------------------------------------------------------------

def step_cost(
    prompt: str,
    response: str,
    model_id: str,
    attack_id: str,
    gcg_backward_mult: float = 3.0,
) -> Tuple[float, float, float, float]:
    """
    Return (target_tokens, attacker_tokens, target_tflops, total_tflops) for one step.

    target_tokens:   tokens consumed by the target model (prompt + response)
    attacker_tokens: tokens consumed by the PAIR attacker (0 for gcg/jailbroken)
    target_tflops:   TFLOPs spent on the target model
    total_tflops:    target_tflops + attacker_tflops
    """
    n_b = _params_b(model_id)
    prompt_tok = len(prompt) / CHARS_PER_TOKEN
    resp_tok   = len(response) / CHARS_PER_TOKEN
    target_tok = prompt_tok + resp_tok

    # forward TFLOPs: 2 × N_B × L / 1000
    fwd_tfl = 2 * n_b * target_tok / 1000

    if attack_id == "gcg":
        target_tfl = fwd_tfl * gcg_backward_mult
        att_tok    = 0.0
        att_tfl    = 0.0
    elif attack_id == "pair":
        target_tfl = fwd_tfl
        # attacker input: boilerplate + current prompt (goal) + prev prompt + response snippet
        att_in_chars  = _PAIR_BOILERPLATE_CHARS + len(prompt) + min(500, len(response))
        att_out_chars = len(prompt)  # refined prompt ≈ same length as input prompt
        att_tok = (att_in_chars + att_out_chars) / CHARS_PER_TOKEN
        att_tfl = 2 * _PAIR_ATTACKER_PARAMS_B * att_tok / 1000
    else:  # jailbroken variants and anything else
        target_tfl = fwd_tfl
        att_tok    = 0.0
        att_tfl    = 0.0

    return target_tok, att_tok, target_tfl, target_tfl + att_tfl


# ---------------------------------------------------------------------------
# Per-trial cumulative cost at a given lambda
# ---------------------------------------------------------------------------

def cumulative_costs(
    record: TrialRecord,
    lambda_val: int,
    gcg_backward_mult: float = 3.0,
) -> Tuple[float, float, float, float]:
    """
    Cumulative (target_tokens, total_tokens, target_tflops, total_tflops)
    consumed up to min(first_success_step, lambda_val) steps.

    Trials that succeeded early have fewer stored steps, so slicing to
    lambda_val naturally caps at the actual stopping point.
    """
    tt = at = ttfl = totfl = 0.0
    for s in record.steps[:lambda_val]:
        tgt_tok, att_tok, tgt_tfl, tot_tfl = step_cost(
            s.prompt, s.response, record.model_id, record.attack_id, gcg_backward_mult
        )
        tt    += tgt_tok
        at    += att_tok
        ttfl  += tgt_tfl
        totfl += tot_tfl
    return tt, tt + at, ttfl, totfl


# ---------------------------------------------------------------------------
# Aggregation across trials
# ---------------------------------------------------------------------------

def aggregate_costs(
    records: List[TrialRecord],
    pressure_levels: List[int],
    gcg_backward_mult: float = 3.0,
) -> Dict[int, Dict[str, float]]:
    """
    For each pressure level, return mean costs across all trials.

    Keys in inner dict:
      mean_target_tokens, mean_total_tokens,
      mean_target_tflops, mean_total_tflops
    """
    result: Dict[int, Dict[str, float]] = {}
    n = len(records)

    for lam in sorted(pressure_levels):
        if lam == 0 or n == 0:
            result[lam] = dict(
                mean_target_tokens=0.0, mean_total_tokens=0.0,
                mean_target_tflops=0.0, mean_total_tflops=0.0,
            )
            continue
        costs = [cumulative_costs(r, lam, gcg_backward_mult) for r in records]
        result[lam] = {
            "mean_target_tokens": sum(c[0] for c in costs) / n,
            "mean_total_tokens":  sum(c[1] for c in costs) / n,
            "mean_target_tflops": sum(c[2] for c in costs) / n,
            "mean_total_tflops":  sum(c[3] for c in costs) / n,
        }
    return result
