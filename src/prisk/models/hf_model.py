"""HuggingFace local model wrapper."""

from __future__ import annotations

from ..utils.config import ModelConfig
from ..utils.logging import get_logger
from .base import BaseModel

logger = get_logger(__name__)


class _SanitizeLogitsProcessor:
    """Replace NaN/Inf logits with 0 before sampling.

    GCG adversarial suffixes are optimized against a specific model's embedding
    space. When replayed on a different model (transfer experiments), some token
    embeddings can produce NaN or Inf logits, which crash torch.multinomial.
    This processor is a no-op on clean logits and costs one .isfinite() check.
    """

    def __call__(self, input_ids, scores):
        import torch
        if not torch.isfinite(scores).all():
            logger.warning("NaN/Inf logits detected — sanitizing before sampling")
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores

# Plain-text continuation format for base (non-instruct) models
_BASE_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


class HFModel(BaseModel):
    """Wraps a HuggingFace causal LM or image-text-to-text model with optional quantization."""

    def __init__(self, config: ModelConfig):
        self._config = config
        self._model = None
        self._processor = None  # AutoTokenizer for causal_lm, AutoProcessor for image_text_to_text
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import BitsAndBytesConfig

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

        if self._config.model_class == "image_text_to_text":
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self._config.hf_name,
                trust_remote_code=True,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._config.hf_name,
                quantization_config=quant_config,
                device_map=self._config.device if quant_config else None,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._processor = AutoTokenizer.from_pretrained(
                self._config.hf_name,
                trust_remote_code=True,
            )
            if self._processor.pad_token is None:
                self._processor.pad_token = self._processor.eos_token

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

    @property
    def _tokenizer(self):
        return self._processor

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt as a string (causal_lm path only)."""
        if self._config.model_type == "base":
            return _BASE_TEMPLATE.format(prompt=prompt)

        if hasattr(self._processor, "chat_template") and self._processor.chat_template:
            messages = [{"role": "user", "content": prompt}]
            kwargs = dict(tokenize=False, add_generation_prompt=True)
            if not self._config.enable_thinking:
                kwargs["enable_thinking"] = False
            return self._processor.apply_chat_template(messages, **kwargs)

        return _BASE_TEMPLATE.format(prompt=prompt)

    def _prepare_inputs_image_text(self, prompt: str) -> dict:
        """Return tokenised inputs dict for image_text_to_text models (text-only)."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {k: v.to(self._config.device) for k, v in inputs.items()}

    def _format_prompt_with_system(self, system_prompt: str, user_prompt: str) -> str:
        """Format a system + user message pair using the model's chat template."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self._processor, "chat_template") and self._processor.chat_template:
            kwargs = dict(tokenize=False, add_generation_prompt=True)
            if not self._config.enable_thinking:
                kwargs["enable_thinking"] = False
            return self._processor.apply_chat_template(messages, **kwargs)
        # Fallback: concatenate system and user into plain template
        return _BASE_TEMPLATE.format(prompt=f"{system_prompt}\n\n{user_prompt}")

    def generate(self, prompt: str, **kwargs) -> str:
        import torch

        gen_cfg = self._config.generation
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        do_sample = kwargs.get("do_sample", gen_cfg.do_sample)
        system_prompt = kwargs.get("system_prompt", None)

        # HF requires temperature > 0; temperature=0 means greedy decoding
        if temperature == 0.0:
            do_sample = False
            temperature = None

        generate_kwargs = dict(
            max_new_tokens=kwargs.get("max_new_tokens", gen_cfg.max_new_tokens),
            do_sample=do_sample,
            repetition_penalty=kwargs.get("repetition_penalty", gen_cfg.repetition_penalty),
            logits_processor=[_SanitizeLogitsProcessor()],
        )
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = kwargs.get("top_p", gen_cfg.top_p)
        else:
            # Explicitly None-out sampling params to suppress model's generation_config values
            generate_kwargs["temperature"] = None
            generate_kwargs["top_p"] = None
            generate_kwargs["top_k"] = None

        if self._config.model_class == "image_text_to_text":
            inputs = self._prepare_inputs_image_text(prompt)
            with torch.no_grad():
                output_ids = self._model.generate(**inputs, **generate_kwargs)
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            return self._processor.decode(new_tokens, skip_special_tokens=True).strip()
        else:
            if system_prompt is not None:
                formatted = self._format_prompt_with_system(system_prompt, prompt)
            else:
                formatted = self._format_prompt(prompt)
            inputs = self._processor(formatted, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self._config.device) for k, v in inputs.items()}
            generate_kwargs["pad_token_id"] = self._processor.pad_token_id
            with torch.no_grad():
                output_ids = self._model.generate(**inputs, **generate_kwargs)
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            return self._processor.decode(new_tokens, skip_special_tokens=True).strip()
