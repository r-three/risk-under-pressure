"""Attack policy factory."""

from __future__ import annotations

from ..utils.config import AttackConfig
from .base import AttackPolicy


def load_attack(config: AttackConfig, attacker_model=None, target_model=None) -> AttackPolicy:
    """
    Instantiate an attack policy from config.

    Args:
        config:          AttackConfig (attack_id determines which class to use)
        attacker_model:  BaseModel used by PAIR to generate refined prompts
        target_model:    BaseModel used by GCG for gradient access
    """
    attack_id = config.attack_id.lower()

    if attack_id == "pair":
        from .pair_attack import PAIRAttack
        return PAIRAttack(config, attacker_model=attacker_model)
    elif attack_id == "gcg":
        from .gcg_attack import GCGAttack
        return GCGAttack(config, target_model=target_model)
    elif attack_id in ("jailbroken", "jailbreak"):
        from .jailbroken_attack import JailBrokenAttack
        return JailBrokenAttack(config)
    else:
        raise ValueError(
            f"Unknown attack '{attack_id}'. Available: 'pair', 'gcg', 'jailbroken'"
        )
