# ===============================================================
# markllm_defense.py
# Description: Adapter that wraps **any** MarkLLM watermark algorithm
#              (KGW, Unigram, SWEET, EXP, EXPEdit, SynthID, ...)
#              into the competition's DefenseBundle interface.
#
#              MarkLLM algorithms modify logits during generation via a
#              LogitsProcessor, so they require a local HF model. The
#              adapter pulls that model from `env.get_transformers_config()`.
#
#              The defense's private `secret_key` is injected as MarkLLM's
#              `hash_key` (and any other algorithm params can be overridden
#              per-defense through config), so each registered defense
#              carries its own private key — exactly as the threat model
#              requires.
# ===============================================================

from __future__ import annotations

import copy
import os
from typing import Any

from ..defense_base import DefenseBundle, WatermarkDefense, WatermarkDetector
from ..registry import register_defense


def _markllm_root() -> str:
    """Path to the bundled MarkLLM-main source tree (sibling of this project)."""
    here = os.path.dirname(os.path.abspath(__file__))
    # arena/ -> WatermarkArena/ -> parent dir contains MarkLLM-main
    return os.path.normpath(os.path.join(here, "..", "..", "MarkLLM-main"))


def _ensure_markllm_on_path() -> str:
    """Add MarkLLM-main to sys.path (idempotent) and return its root."""
    import sys

    root = _markllm_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _default_algorithm_config_path(algorithm_name: str) -> str:
    root = _ensure_markllm_on_path()
    return os.path.join(root, "config", f"{algorithm_name}.json")


class MarkLLMWatermarkDefense(WatermarkDefense):
    """Defense generator backed by a MarkLLM watermark algorithm."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.algorithm_name = config["algorithm"]
        self.secret_key = int(config.get("secret_key", 15485863))
        self.algorithm_config_path = config.get(
            "algorithm_config_path"
        ) or _default_algorithm_config_path(self.algorithm_name)
        # Extra per-algorithm overrides merged into the MarkLLM config dict.
        self.param_overrides: dict = config.get("params", {}) or {}
        # Cached watermark instance (built on first use, once env is known).
        self._watermark = None
        self._watermark_lock = None  # built lazily; env may differ across calls

    def _build_watermark(self, env):
        """Instantiate the MarkLLM AutoWatermark using env's local model."""
        # Fail fast with a clear message BEFORE importing MarkLLM/transformers:
        # logits-based watermarking requires a local HF model.
        if not env.has_local_model():
            raise RuntimeError(
                f"MarkLLM defense '{self.algorithm_name}' requires a local HF model "
                f"(it modifies logits during generation). Configure local_model.path "
                f"in config.yaml (or set ARENA_LOCAL_MODEL), or use the API-native "
                f"'BlackBoxGreenList' defense which works through the OpenAI interface only."
            )
        _ensure_markllm_on_path()
        try:
            from watermark.auto_watermark import AutoWatermark
        except ImportError as e:
            raise RuntimeError(
                f"Could not import MarkLLM (watermark module). Ensure MarkLLM-main is "
                f"present at {_markllm_root()!r} and that 'transformers'/'torch' are "
                f"installed (pip install -r requirements.txt). Original error: {e}"
            ) from e

        transformers_config = env.get_transformers_config()
        kwargs = dict(self.param_overrides)
        # Inject the private secret key as MarkLLM's hash_key unless the
        # defense explicitly overrides it (it shouldn't).
        kwargs.setdefault("hash_key", self.secret_key)
        watermark = AutoWatermark.load(
            algorithm_name=self.algorithm_name,
            algorithm_config=self.algorithm_config_path,
            transformers_config=transformers_config,
            **kwargs,
        )
        return watermark

    def _get_watermark(self, env):
        if self._watermark is None:
            import threading

            self._watermark_lock = threading.Lock()
            with self._watermark_lock:
                if self._watermark is None:
                    self._watermark = self._build_watermark(env)
        return self._watermark

    def generate(self, env: Any, request: dict) -> dict:
        prompt = request["prompt"]
        watermark = self._get_watermark(env)
        text = watermark.generate_watermarked_text(prompt)
        return {"text": text}

    def generate_unwatermarked(self, env: Any, request: dict) -> dict:
        """Use the *same* local model without the watermark logits processor."""
        prompt = request["prompt"]
        watermark = self._get_watermark(env)
        text = watermark.generate_unwatermarked_text(prompt)
        return {"text": text}


class MarkLLMWatermarkDetector(WatermarkDetector):
    """Detector backed by the same MarkLLM algorithm (same secret key)."""

    def __init__(self, config: dict):
        super().__init__(config)
        # Mirror the generator config so detection uses the identical key.
        self._generator = MarkLLMWatermarkDefense(config)
        self._watermark = None

    def _get_watermark(self, env):
        if self._watermark is None:
            self._watermark = self._generator._get_watermark(env)
        return self._watermark

    def detect(self, text: str, env: Any | None = None) -> dict:
        if self._watermark is None and env is None:
            raise RuntimeError(
                "Detector not initialized: call detect(text, env=env) once, "
                "or pre-bind via bind_env(env)."
            )
        if self._watermark is None:
            self._get_watermark(env)
        result = self._watermark.detect_watermark(text, return_dict=True)
        return {
            "is_watermarked": bool(result["is_watermarked"]),
            "score": float(result["score"]),
        }

    def bind_env(self, env) -> None:
        """Pre-bind the local model so detect() needs no env argument."""
        self._get_watermark(env)


def _markllm_defense_factory(algorithm_name: str):
    """Build a factory for a specific MarkLLM algorithm."""

    def factory(config: dict) -> DefenseBundle:
        cfg = copy.deepcopy(config or {})
        cfg["algorithm"] = algorithm_name
        generator = MarkLLMWatermarkDefense(cfg)
        detector = MarkLLMWatermarkDetector(cfg)
        return DefenseBundle(
            name=cfg.get("name", algorithm_name),
            generator=generator,
            detector=detector,
            meta={"algorithm": algorithm_name, "requires_local_model": True},
        )

    return factory


# ----------------------------------------------------------------------
# Register the MarkLLM-backed defenses.
# These are the logits-based methods that work uniformly through
# MarkLLM's BaseWatermark interface. (Semantic-stamp / SIR / IE / TS /
# UPV / Adaptive need extra dependencies and are registered separately
# or left for the user to enable.)
# ----------------------------------------------------------------------
_MARKLLM_DEFENSES = [
    "KGW",
    "Unigram",
    "SWEET",
    "EXP",
    "EXPGumbel",
    "EXPEdit",
    "ITSEdit",
    "EWD",
    "Unbiased",
    "SynthID",
    "PF",
    "MorphMark",
]

for _alg in _MARKLLM_DEFENSES:
    register_defense(
        _alg,
        meta={"algorithm": _alg, "requires_local_model": True, "source": "MarkLLM"},
    )(_markllm_defense_factory(_alg))
