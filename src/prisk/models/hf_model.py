"""HuggingFace local model wrapper."""

from __future__ import annotations

from typing import Optional

from ..utils.config import ModelConfig
from ..utils.logging import get_logger
from .base import BaseModel

logger = get_logger(__name__)

# Plain-text continuation format for base (non-instruct) models
_BASE_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


class HFModel(BaseModel):
    """Wraps a HuggingFace causal LM with optional quantization."""

    def __init__(self, config: ModelConfig):
        self._config = config
        self._model = None
        self._tokenizer = None
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading {self._config.hf_name} (quant={self._config.quantization})")

        quant_config = None
        if self._config.quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self._config.quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.hf_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self._config.hf_name,
            quantization_config=quant_config,
            device_map=self._config.device if quant_config else None,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        if quant_config is None:
            self._model = self._model.to(self._config.device)
        self._model.eval()
        logger.info(f"Loaded {self._config.hf_name}")

    @property
    def model_id(self) -> str:
        return self._config.model_id

    def _format_prompt(self, prompt: str) -> str:
        if self._config.model_type == "base":
            return _BASE_TEMPLATE.format(prompt=prompt)

        # Use tokenizer chat template for instruct models
        if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback if no chat template
        return _BASE_TEMPLATE.format(prompt=prompt)

    def generate(self, prompt: str, **kwargs) -> str:
        import torch

        formatted = self._format_prompt(prompt)
        gen_cfg = self._config.generation
        inputs = self._tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self._config.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", gen_cfg.max_new_tokens),
                temperature=kwargs.get("temperature", gen_cfg.temperature),
                do_sample=kwargs.get("do_sample", gen_cfg.do_sample),
                top_p=kwargs.get("top_p", gen_cfg.top_p),
                repetition_penalty=kwargs.get("repetition_penalty", gen_cfg.repetition_penalty),
                pad_token_id=self._tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
