"""Pydantic config models for models, attacks, and experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    max_new_tokens: int = 128
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    repetition_penalty: float = 1.0


class ModelConfig(BaseModel):
    model_id: str
    backend: Literal["huggingface"]
    hf_name: Optional[str] = None           # HuggingFace model name/path
    hf_tokenizer_id: Optional[str] = None   # override tokenizer if it differs from hf_name
    params_b: Optional[float] = None        # parameter count in billions (for FLOP calculation)
    model_type: Literal["base", "instruct"] = "instruct"
    quantization: Literal["4bit", "8bit", "none"] = "none"
    device: str = "cuda"
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    model_class: Literal["causal_lm", "image_text_to_text"] = "causal_lm"
    enable_thinking: bool = True            # set false for Qwen3-style thinking models
    extra: Dict[str, Any] = Field(default_factory=dict)


class AttackConfig(BaseModel):
    attack_id: str
    attacker_model: Optional[str] = None    # model_id for PAIR's attacker LLM
    max_query_per_step: int = 1             # EasyJailbreak queries per refine() call
    templates: Optional[List[str]] = None  # JailBroken template list override
    extra: Dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    name: str
    benchmark: Literal["jailbreakbench", "harmbench"]
    n_prompts: int = 100
    categories: Optional[List[str]] = None  # filter to specific harm categories
    pressure_levels: List[int] = Field(default_factory=lambda: [0, 1, 3, 5, 10])
    lambda_max: int = 10
    models: List[str]                        # model config filenames (without .yaml)
    attacks: List[str]                       # attack config filenames (without .yaml)
    judge_model: str = "gpt4o_mini"          # references a model config
    attacker_model: Optional[str] = None     # overrides attack config's attacker_model when set
    attacker_models: Optional[List[str]] = None  # compare multiple attackers; overrides attacker_model
    output_dir: str = "outputs/"
    seed: int = 42                            # kept for backward compat; prefer seeds
    seeds: Optional[List[int]] = None         # if set, overrides seed; one run per seed value
    tau: float = 0.5                         # risk tolerance for lambda* computation
    n_bootstrap: int = 1000


def load_config(path: str | Path, config_class: type) -> Any:
    with open(path) as f:
        data = yaml.safe_load(f)
    return config_class(**data)


def load_model_config(name: str, configs_dir: str | Path = "configs") -> ModelConfig:
    path = Path(configs_dir) / "models" / f"{name}.yaml"
    return load_config(path, ModelConfig)


def load_attack_config(name: str, configs_dir: str | Path = "configs") -> AttackConfig:
    path = Path(configs_dir) / "attacks" / f"{name}.yaml"
    return load_config(path, AttackConfig)
