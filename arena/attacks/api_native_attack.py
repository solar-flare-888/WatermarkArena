# ===============================================================
# api_native_attack.py
# Description: Attacks implemented purely through the standard OpenAI
#              interface (env.generate). These are the strongest,
#              most realistic de-watermarking attacks in the arena
#              because an LLM rewrite preserves semantics while
#              thoroughly scrambling the token-level statistics that
#              logits-based watermarks (KGW/SWEET/...) rely on.
#
#              Each attack is registered independently and knows
#              nothing about which defense it will face.
# ===============================================================

from __future__ import annotations

from typing import Any

from ..attack_base import BaseAttack
from ..registry import register_attack


# ------------------------------------------------------------------
# Shared prompt builder
# ------------------------------------------------------------------
def _rewrite_prompt(text: str, language: str, strength: str = "medium") -> str:
    lang_hint = {
        "en": "English",
        "zh": "Chinese",
        "chinese": "Chinese",
        "中英": "the same mix of Chinese and English as the original",
    }.get((language or "").lower(), "the same language as the original")

    intensity = {
        "low": "Make small wording changes only.",
        "medium": "Substantially reword sentences and vary structure, but keep every fact and the full meaning.",
        "high": "Aggressively paraphrase: reorganize paragraphs, change voice and sentence order, replace most vocabulary, while preserving every fact and the task answer.",
    }.get(strength, "Substantially reword sentences and vary structure, but keep every fact and the full meaning.")

    return (
        f"Rewrite the following text. Requirements:\n"
        f"1. Preserve ALL facts, names, numbers, code and the task answer exactly.\n"
        f"2. Preserve the overall meaning and intent.\n"
        f"3. {intensity}\n"
        f"4. Output ONLY the rewritten text, no explanations.\n"
        f"5. Write in {lang_hint}.\n\n"
        f"Text:\n{text}"
    )


class LLMRewriteAttack(BaseAttack):
    """Single-pass LLM paraphrase via the standard OpenAI interface."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.model = (config or {}).get("model")
        self.temperature = float((config or {}).get("temperature", 0.9))
        self.max_tokens = int((config or {}).get("max_tokens", 1024))
        self.strength = (config or {}).get("strength", "medium")

    def attack(self, env: Any, task: dict) -> dict:
        text = task["watermarked_text"]
        language = task.get("language", "")
        prompt = _rewrite_prompt(text, language, self.strength)
        rewritten = env.generate(
            prompt,
            system_content="You are an expert editor. You rewrite text to change wording and "
            "structure while perfectly preserving meaning and factual content.",
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return {"text": rewritten}


class LLMRoundTripAttack(BaseAttack):
    """Translate to a pivot language and back, then lightly fix with the LLM.

    A classic, strong de-watermarking pipeline: round-trip translation
    destroys token-level statistics while (mostly) preserving meaning,
    and a final LLM cleanup pass repairs any awkwardness.
    """

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.model = (config or {}).get("model")
        self.pivot = (config or {}).get("pivot_lang", "Chinese")
        self.target = (config or {}).get("target_lang", "English")
        self.temperature = float((config or {}).get("temperature", 0.7))
        self.max_tokens = int((config or {}).get("max_tokens", 1024))

    def _translate(self, env, text, src, dst):
        prompt = (
            f"Translate the following text from {src} into {dst}. "
            f"Output only the translation, no notes.\n\n{text}"
        )
        return env.generate(
            prompt,
            system_content="You are a professional translator.",
            model=self.model,
            temperature=0.3,
            max_tokens=self.max_tokens,
        )

    def attack(self, env: Any, task: dict) -> dict:
        text = task["watermarked_text"]
        language = (task.get("language") or "").lower()
        target = "Chinese" if language.startswith("zh") or language == "chinese" else "English"
        pivot = "English" if target == "Chinese" else "Chinese"

        first = self._translate(env, text, target, pivot)
        if not first.strip():
            return {"text": text}
        second = self._translate(env, first, pivot, target)
        if not second.strip():
            return {"text": text}

        # Cleanup pass to restore fluency while keeping content.
        cleanup_prompt = (
            "Polish the following text for fluency and naturalness without changing any "
            "facts or meaning. Output only the polished text.\n\n" + second
        )
        cleaned = env.generate(
            cleanup_prompt,
            system_content="You are an editor.",
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return {"text": cleaned if cleaned.strip() else second}


class IdentityAttack(BaseAttack):
    """No-op attack (returns the text unchanged). Baseline for ASR lower-bound."""

    def attack(self, env: Any, task: dict) -> dict:
        return {"text": task["watermarked_text"]}


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------
@register_attack(
    "LLMRewrite",
    meta={"source": "API-native", "needs_llm": True, "requires_local_model": False},
)
def _llm_rewrite_factory(config: dict) -> BaseAttack:
    return LLMRewriteAttack(config)


@register_attack(
    "LLMRoundTrip",
    meta={"source": "API-native", "needs_llm": True, "requires_local_model": False},
)
def _llm_roundtrip_factory(config: dict) -> BaseAttack:
    return LLMRoundTripAttack(config)


@register_attack(
    "Identity",
    meta={"source": "API-native", "needs_llm": False, "requires_local_model": False},
)
def _identity_factory(config: dict) -> BaseAttack:
    return IdentityAttack(config)
