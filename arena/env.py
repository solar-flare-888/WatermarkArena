# ===============================================================
# env.py
# Description: LLM access environment via the **standard OpenAI
#              interface** (openai python SDK).
#
#              A single `LLMEnv` is passed to both the defense
#              `generate(env, request)` and the attack
#              `attack(env, task)` callables defined by the
#              competition spec, so that every LLM touch-point in
#              the arena goes through one OpenAI-compatible client.
#
#              Works with any OpenAI-compatible endpoint:
#                - OpenAI official (api.openai.com)
#                - Azure OpenAI
#                - Local vLLM / TGI / LMDeploy / Ollama / LM Studio
#                - Any service that implements /v1/chat/completions
#
#              Two access modes:
#                1. Black-box text generation  -> env.generate(...)
#                   (used by attacks, unwatermarked baselines, judge)
#                2. Logits-level access         -> env.get_transformers_config()
#                   (used by MarkLLM defenses that modify logits during
#                    generation; requires a local HF model path)
# ===============================================================

from __future__ import annotations

import os
import threading
from typing import Any, Optional

import openai


class LLMEnv:
    """Environment that exposes LLM access through the standard OpenAI interface.

    The defense and attack sides are *only* allowed to talk to the base LLM
    through this object, which keeps the "standard OpenAI interface"
    requirement centralized and decoupled from any specific watermark or
    attack implementation.
    """

    def __init__(self, config: dict):
        self.config = config or {}

        # ---- OpenAI-compatible client -------------------------------------
        llm_cfg = self.config.get("llm", {})
        self.api_key = llm_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.base_url = llm_cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.model = llm_cfg.get("model") or os.environ.get("OPENAI_MODEL") or "gpt-3.5-turbo"
        self.default_temperature = float(llm_cfg.get("temperature", 0.7))
        self.default_max_tokens = int(llm_cfg.get("max_tokens", 512))
        self.timeout = float(llm_cfg.get("timeout", 60))
        self.max_retries = int(llm_cfg.get("max_retries", 4))

        # Lazily constructed client (created on first use so that simply
        # importing the arena never requires a live key).
        self._client: Optional[openai.OpenAI] = None
        self._client_lock = threading.Lock()

        # ---- Optional local HF model for logits-based watermarking -------
        # MarkLLM watermark algorithms (KGW, SWEET, EXP, ...) modify logits
        # during generation, which a pure chat-completions API cannot do.
        # When `local_model.path` is configured we lazy-load a HuggingFace
        # model+tokenizer so the defense can build a LogitsProcessor.
        local_cfg = self.config.get("local_model", {}) or {}
        self.local_model_path: Optional[str] = local_cfg.get("path")
        self.local_device: str = local_cfg.get("device", "cpu")
        self.local_model_dtype: str = local_cfg.get("dtype", "auto")
        self._hf_cache: dict[str, Any] = {}
        self._hf_lock = threading.Lock()

        # Generation defaults forwarded into HF `.generate(...)` for the
        # watermarked / unwatermarked generation paths.
        gen_cfg = local_cfg.get("gen_kwargs", {}) or {}
        self.hf_gen_kwargs = {
            "max_new_tokens": int(gen_cfg.get("max_new_tokens", 256)),
            "do_sample": bool(gen_cfg.get("do_sample", True)),
            "temperature": float(gen_cfg.get("temperature", 0.7)),
            "top_k": int(gen_cfg.get("top_k", 0)),
            "top_p": float(gen_cfg.get("top_p", 0.9)),
            "repetition_penalty": float(gen_cfg.get("repetition_penalty", 1.1)),
        }

    # ------------------------------------------------------------------
    # Black-box OpenAI generation
    # ------------------------------------------------------------------
    @property
    def client(self) -> openai.OpenAI:
        """Return a lazily-initialized OpenAI client (standard SDK)."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        timeout=self.timeout,
                        max_retries=self.max_retries,
                    )
        return self._client

    def generate(
        self,
        prompt: str,
        system_content: Optional[str] = None,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text via the standard OpenAI chat-completions interface.

        This is the single black-box entry point used by attacks (for
        paraphrasing / rewriting) and by the arena (for unwatermarked
        baseline text and the quality judge). It is deliberately tolerant
        of endpoint differences: any OpenAI-compatible server works.
        """
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": prompt})

        request_kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
        }
        if stop:
            request_kwargs["stop"] = stop
        # Allow callers to pass through extra OpenAI params (e.g. logit_bias).
        request_kwargs.update(kwargs)

        response = self.client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Logits-level access for MarkLLM defenses
    # ------------------------------------------------------------------
    def has_local_model(self) -> bool:
        """Whether a local HF model is configured (needed for logits-based watermarking)."""
        return bool(self.local_model_path)

    def get_transformers_config(self):
        """Return a MarkLLM `TransformersConfig` backed by a local HF model.

        Lazily loads & caches the model + tokenizer. Used only by MarkLLM
        defenses (KGW / SWEET / EXP / ...). Raises a clear error when no
        local model is configured, so callers can fall back to an
        API-native defense.
        """
        if not self.has_local_model():
            raise RuntimeError(
                "No local HF model configured (local_model.path). Logits-based "
                "watermarking requires a local model; configure one in config.yaml "
                "or use an API-native defense."
            )

        with self._hf_lock:
            if "transformers_config" not in self._hf_cache:
                self._hf_cache["transformers_config"] = self._build_transformers_config()
            return self._hf_cache["transformers_config"]

    def _build_transformers_config(self):
        """Load the local HF model/tokenizer and wrap them in MarkLLM's TransformersConfig."""
        # Imported lazily so the arena can be imported without transformers.
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        from utils.transformers_config import TransformersConfig  # MarkLLM

        path = self.local_model_path
        hf_config = AutoConfig.from_pretrained(path, trust_remote_code=True)

        dtype_map = {"auto": "auto", "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        torch_dtype = dtype_map.get(self.local_model_dtype, "auto")

        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.local_device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return TransformersConfig(
            model=model,
            tokenizer=tokenizer,
            vocab_size=hf_config.vocab_size,
            device=self.local_device,
            **self.hf_gen_kwargs,
        )


def make_env(config: dict) -> LLMEnv:
    """Convenience factory used throughout the arena."""
    return LLMEnv(config)
