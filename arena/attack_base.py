# ===============================================================
# attack_base.py
# Description: Abstract attack interface, matching the competition
#              spec exactly (attack.py).
#
#              attack(env, task) -> {"text": str}
#
#              The attack only ever sees the *watermarked text* and the
#              language; it must not read the defense key, detector
#              threshold, or defense source. Decoupling is enforced by
#              the registry / confrontation runner: an attack is built
#              with zero knowledge of which defense it will face.
# ===============================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAttack(ABC):
    """Attack-side base class (spec: attack.py)."""

    # Subclasses may declare which languages they support; None = all.
    supported_languages: list[str] | None = None

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    def attack(self, env: Any, task: dict) -> dict:
        """Remove / weaken the watermark while preserving semantics.

        Args:
            env: LLMEnv (standard OpenAI interface) for any LLM-based
                 rewriting the attack needs.
            task: {
                "sample_id": str,
                "watermarked_text": str,
                "language": str
            }
        Returns:
            {"text": str}
        """
        raise NotImplementedError

    # Convenience: make the object callable as attack(env, task).
    def __call__(self, env: Any, task: dict) -> dict:
        result = self.attack(env, task)
        # Enforce the spec's hard constraints to avoid degenerate attacks.
        text = result.get("text", "") if isinstance(result, dict) else ""
        if not isinstance(text, str) or not text.strip():
            # Never return empty / non-text; fall back to the input.
            return {"text": task.get("watermarked_text", "")}
        return {"text": text}
