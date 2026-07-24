"""Per-prompt GRPO adaptive attack (arXiv:2510.09023, App. A.2).

For EACH behavior, run a short GRPO optimization against the (frozen) target: sample a group
of candidate adversarial prompts, score them with the reward oracle, and update the attacker's
weights with a group-relative policy gradient — repeated over rounds until the FIRST judged-unsafe
candidate is found (early stop) or the per-prompt query budget (= lambda_max) is exhausted.
Success = whether any step ever reached `judgment == 1` (recorded as `first_success_step`).

DEVIATION FROM THE PAPER: arXiv:2510.09023 runs the RL attack for a FIXED budget (multiple
sessions × ~5 rounds) with best-of scoring and NO first-success early stop. Here we early-stop at
the first success to mirror `run_trial`'s semantics (PAIR/GCG break at first `judgment == 1`), so
RL's pressure / cost accounting is comparable across attacks. This does NOT change risk-vs-lambda
or the success labels — `first_success_step` is identical either way; early stopping only trims
the queries/FLOPs spent AFTER the jailbreak is found.

There is no shared training set: each prompt starts from a freshly-reset LoRA adapter, so the
attack is optimized on the prompt itself (worst-case adaptive, like GCG/PAIR — no train/test
leakage). One candidate = one target query = one recorded `StepResult`, so the resulting
`TrialRecord` drops straight into the existing risk / cost / plotting pipeline, and the attacker
backward passes are what make RL's per-query FLOPs higher than PAIR's.

The GRPO update here is the group-relative core (group-normalized advantage + a KL penalty to
the frozen base model via adapter-disable), single inner epoch, no PPO clipping.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..attacks.rl_prompts import RL_ATTACKER_SYSTEM_PROMPT, build_attacker_user_message
from ..training.rl_env import AttackEnv, RewardConfig
from ..utils.io import StepResult, TrialRecord
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _append_trace(trace_path: Optional[Path], row: Dict[str, Any]) -> None:
    """Append one JSONL row to the GRPO training trace (rollouts + scores per round).

    Written incrementally so training can be watched live; a failure to log must never
    kill the attack loop.
    """
    if trace_path is None:
        return
    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:  # noqa: BLE001 - tracing is best-effort
        logger.warning(f"failed to write RL trace row: {e}")


@dataclass
class GRPOAttackConfig:
    """Per-prompt GRPO hyperparameters (populated from configs/attacks/rl.yaml `extra`)."""

    base_attacker_hf: str = "Qwen/Qwen2.5-7B-Instruct"
    num_generations: int = 8            # GRPO group size (candidates per round)
    session_rounds: int = 1             # in-context refinement rounds per candidate (paper: 5)
    learning_rate: float = 1e-5
    beta: float = 0.04                  # KL penalty coefficient (to frozen base)
    temperature: float = 1.0
    top_p: float = 0.95
    max_completion_length: int = 256
    max_grad_norm: float = 1.0
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    @classmethod
    def from_extra(cls, extra: Dict[str, Any], attacker_hf: Optional[str] = None) -> "GRPOAttackConfig":
        extra = dict(extra or {})
        if attacker_hf:
            extra.setdefault("base_attacker_hf", attacker_hf)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in extra.items() if k in known})


@dataclass
class RLAttacker:
    """Holds the trainable attacker (loaded once, LoRA reset per prompt)."""

    model: Any
    tokenizer: Any
    init_adapter_state: Dict[str, Any]
    device: Any


# --------------------------------------------------------------------------- #
# Attacker construction / reset
# --------------------------------------------------------------------------- #
def build_rl_attacker(cfg: GRPOAttackConfig) -> RLAttacker:
    """Load Qwen2.5-7B-Instruct (bf16) + a fresh LoRA adapter for GRPO. Loaded once, reused."""
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading trainable RL attacker: {cfg.base_attacker_hf} (bf16 + LoRA)")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_attacker_hf, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_attacker_hf,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, lora)
    # Snapshot the freshly-initialized adapter so every prompt starts from the same state.
    init_state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    device = next(model.parameters()).device
    return RLAttacker(model=model, tokenizer=tokenizer, init_adapter_state=init_state, device=device)


def _reset_adapter(attacker: RLAttacker) -> None:
    """Restore the adapter to its initial state — per-prompt independence (no cross-prompt learning)."""
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(attacker.model, attacker.init_adapter_state)


# --------------------------------------------------------------------------- #
# Generation + log-probs
# --------------------------------------------------------------------------- #
def _prompt_ids(attacker: RLAttacker, behavior: str, prompt: str, response: str, judgment: int):
    """Chat-formatted attacker prompt ids for one refinement context."""
    messages = [
        {"role": "system", "content": RL_ATTACKER_SYSTEM_PROMPT},
        {"role": "user", "content": build_attacker_user_message(behavior, prompt, response, judgment)},
    ]
    ids = attacker.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
    )
    return ids.to(attacker.device)


def _sample_group(attacker: RLAttacker, prompt_ids, cfg: GRPOAttackConfig):
    """Sample `num_generations` candidates. Returns (list[str] texts, list[Tensor] completion_ids)."""
    import torch

    with torch.no_grad():
        out = attacker.model.generate(
            input_ids=prompt_ids,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_completion_length,
            num_return_sequences=cfg.num_generations,
            pad_token_id=attacker.tokenizer.pad_token_id,
        )
    plen = prompt_ids.shape[1]
    texts, comp_ids = [], []
    for seq in out:
        comp = seq[plen:]
        comp_ids.append(comp)
        texts.append(attacker.tokenizer.decode(comp, skip_special_tokens=True).strip())
    return texts, comp_ids


def _completion_logprob(attacker: RLAttacker, prompt_ids, completion_ids):
    """Sum log-prob of `completion_ids` given `prompt_ids` under the current (adapter-on) policy.

    Differentiable — used for the policy-gradient loss.
    """
    import torch

    full = torch.cat([prompt_ids[0], completion_ids], dim=0).unsqueeze(0)  # [1, L]
    plen = prompt_ids.shape[1]
    out = attacker.model(input_ids=full)
    logits = out.logits[:, :-1, :]                      # predict next token
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    targets = full[:, 1:]
    tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
    return tok_lp[:, plen - 1:].sum()                   # completion tokens only


def _ref_logprob(attacker: RLAttacker, prompt_ids, completion_ids):
    """Same as _completion_logprob but under the frozen base model (adapter disabled), no grad."""
    import torch

    with torch.no_grad(), attacker.model.disable_adapter():
        return _completion_logprob(attacker, prompt_ids, completion_ids).detach()


# --------------------------------------------------------------------------- #
# Per-prompt GRPO loop
# --------------------------------------------------------------------------- #
def run_prompt_rl(
    base_prompt: str,
    prompt_id: str,
    behavior: str,
    category: str,
    source: str,
    target,
    judge,
    attacker: RLAttacker,
    budget: int,
    cfg: GRPOAttackConfig,
    reward_config: Optional[RewardConfig] = None,
    trace_path: Optional[Path] = None,
) -> TrialRecord:
    """Run per-prompt GRPO for one behavior and return a TrialRecord (schema-identical to run_trial).

    If `trace_path` is given, a per-round JSONL training trace is appended there: every candidate
    rollout (prompt/response/reward/judgment), the group-relative advantages, and the GRPO loss —
    so you can inspect how the trajectory drives the attack toward a jailbreak.
    """
    import torch

    env = AttackEnv(target_model=target, judge=judge, reward_config=reward_config)
    _reset_adapter(attacker)
    optimizer = torch.optim.AdamW(
        [p for p in attacker.model.parameters() if p.requires_grad], lr=cfg.learning_rate
    )

    t0 = time.time()
    steps: List[StepResult] = []
    t_star: Optional[int] = None
    q = 0  # cumulative target queries (= pressure)

    # Step 1: the raw behavior itself (p^0), mirroring PAIR — gives the round-1 feedback context.
    r0, resp0, judg0 = env.score(base_prompt, behavior)
    q += 1
    steps.append(StepResult(step=q, prompt=base_prompt, response=resp0, judgment=judg0))
    if judg0 == 1 and t_star is None:
        t_star = q
    ctx_prompt, ctx_resp, ctx_judg = base_prompt, resp0, judg0
    final_prompt = base_prompt

    # Round 0 = the raw behavior, the context the first GRPO round refines from.
    _append_trace(trace_path, {
        "prompt_id": prompt_id, "model_id": target.model_id, "round": 0,
        "query": q, "reward": r0, "judgment": judg0,
        "prompt": base_prompt, "response": resp0,
    })

    # GRPO rounds until the per-prompt query budget is exhausted OR the first
    # jailbreak is found, mirroring run_trial's early-stop semantics so query
    # count / pressure stay comparable across attacks. NOTE: this diverges from
    # the paper's fixed-budget/best-of RL procedure (see module docstring).
    round_idx = 0
    while q < budget and t_star is None:
        round_idx += 1
        q_start = q
        prompt_ids = _prompt_ids(attacker, behavior, ctx_prompt, ctx_resp, ctx_judg)
        texts, comp_ids = _sample_group(attacker, prompt_ids, cfg)

        rewards: List[float] = []
        rollouts: List[Dict[str, Any]] = []  # per-candidate log for this round
        best = None  # (reward, text, response, judgment)
        best_idx = -1
        for text, comp in zip(texts, comp_ids):
            if q >= budget:
                break
            reward, response, judgment = env.score(text, behavior)
            q += 1
            steps.append(StepResult(step=q, prompt=text, response=response, judgment=judgment))
            rewards.append(reward)
            rollouts.append({
                "query": q, "candidate": text, "reward": reward,
                "judgment": judgment, "response": response,
            })
            if best is None or reward > best[0]:
                best = (reward, text, response, judgment)
                best_idx = len(rewards) - 1
            # First-success early stop — no further refinement/queries needed.
            # Deliberate deviation from the paper's full-budget/best-of RL procedure,
            # for parity with run_trial's pressure/cost semantics (see module docstring).
            if judgment == 1 and t_star is None:
                t_star = q
                break

        # GRPO update on the candidates we actually scored this round.
        # Backward per candidate (grad accumulation) so only one 7B autograd graph is alive at a
        # time — bounds peak VRAM regardless of num_generations.
        # Skip the update if we just succeeded — this round is the last one and
        # the per-prompt adapter is reset before the next behavior anyway.
        scored = len(rewards)
        advantages: Optional[List[float]] = None
        round_loss: Optional[float] = None
        if scored >= 2 and t_star is None:
            adv = torch.tensor(rewards, dtype=torch.float32, device=attacker.device)
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            advantages = [float(a) for a in adv]
            optimizer.zero_grad()
            total_loss = 0.0
            for i in range(scored):
                ref = _ref_logprob(attacker, prompt_ids, comp_ids[i])       # no grad
                lp = _completion_logprob(attacker, prompt_ids, comp_ids[i])  # grad
                loss_i = (-adv[i] * lp + cfg.beta * (lp - ref)) / scored     # KL(policy‖base) est.
                loss_i.backward()
                total_loss += float(loss_i.detach())
            torch.nn.utils.clip_grad_norm_(
                [p for p in attacker.model.parameters() if p.requires_grad], cfg.max_grad_norm
            )
            optimizer.step()
            round_loss = total_loss
            logger.debug(f"[{prompt_id}] q={q} loss={total_loss:.3f} meanR={sum(rewards)/scored:.3f}")

        # Persist the round's rollouts + GRPO stats so the trajectory is inspectable.
        if scored:
            for i, r in enumerate(rollouts):
                r["advantage"] = advantages[i] if advantages is not None else None
                r["is_best"] = (i == best_idx)
            _append_trace(trace_path, {
                "prompt_id": prompt_id, "model_id": target.model_id, "round": round_idx,
                "query_start": q_start + 1, "query_end": q,
                "mean_reward": sum(rewards) / scored, "max_reward": max(rewards),
                "best_idx": best_idx, "any_success": t_star is not None and t_star > q_start,
                "loss": round_loss, "rollouts": rollouts,
            })

        # Carry the best candidate's feedback into the next round (in-context refinement).
        if best is not None:
            _, ctx_prompt, ctx_resp, ctx_judg = best
            final_prompt = ctx_prompt

    elapsed = time.time() - t0
    success = t_star is not None
    logger.info(
        f"[{target.model_id}/rl] {prompt_id}: success={success} t*={t_star} "
        f"budget={budget} ({elapsed:.1f}s)"
    )
    return TrialRecord(
        prompt_id=prompt_id,
        base_prompt=base_prompt,
        behavior=behavior,
        category=category,
        source=source,
        model_id=target.model_id,
        attack_id="rl",
        budget=budget,
        steps=steps,
        success=success,
        first_success_step=t_star,
        final_prompt=final_prompt,
        metadata={"elapsed_seconds": elapsed},
    )
