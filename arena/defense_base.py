# ===============================================================
# defense_base.py
# Description: Abstract defense interfaces, matching the competition
#              spec exactly (watermark.py / detector.py).
#
#              A "defense" is a pair:
#                 WatermarkDefense.generate(env, request) -> {"text": str}
#                 WatermarkDetector.detect(text)          -> {"is_watermarked": bool, "score": float}
#
#              `score` MUST satisfy: larger -> more likely watermarked,
#              so the arena can compute ROC-AUC / TPR@FPR.
# ===============================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class WatermarkDefense(ABC):
    """Defense-side watermark generator (spec: watermark.py)."""

    def __init__(self, config: dict):
        # config may carry secret_key, algorithm name, tokenizer, model metadata, ...
        self.config = config or {}

    @abstractmethod
    def generate(self, env: Any, request: dict) -> dict:
        """Generate watermarked text.

        Args:
            env: LLMEnv exposing the base LLM via the standard OpenAI interface
                 (and, when configured, a local HF model for logits-based methods).
            request: {
                "sample_id": str,
                "prompt": str,
                "generation_config": dict   # optional overrides
            }
        Returns:
            {"text": str}
        """
        raise NotImplementedError

    # ----- optional hooks -----
    def generate_unwatermarked(self, env: Any, request: dict) -> dict:
        """Generate *unwatermarked* text (used for FPR baselines & quality Q_base).

        Default: plain env.generate(). Defenses with a local model may override
        to use the same model without the logits processor.
        """
        prompt = request["prompt"]
        text = env.generate(prompt)
        return {"text": text}


class WatermarkDetector(ABC):
    """Defense-side watermark detector (spec: detector.py)."""

    def __init__(self, config: dict):
        self.config = config or {}

    @abstractmethod
    def detect(self, text: str) -> dict:
        """Detect watermark in text.

        Returns:
            {"is_watermarked": bool, "score": float}
            where score is monotonically related to watermark likelihood
            (larger -> more likely watermarked).
        """
        raise NotImplementedError


class DefenseBundle:
    """A (generator, detector) pair registered together.

    The arena pairs attacks with defense *bundles*; the detector always
    belongs to its generator so the pairing is honest (a detector never
    sees a key it wasn't trained with).
    """

    def __init__(self, name: str, generator: WatermarkDefense, detector: WatermarkDetector, meta: dict | None = None):
        self.name = name
        self.generator = generator
        self.detector = detector
        self.meta = meta or {}

    def __repr__(self) -> str:
        return f"<DefenseBundle '{self.name}'>"
