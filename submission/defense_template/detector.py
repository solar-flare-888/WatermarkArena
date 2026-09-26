# ===============================================================
# detector.py  --  Defense submission template (spec 6.3 / 21.1)
#
#   Implements the exact competition interface:
#       class WatermarkDetector:
#           def __init__(self, config: dict): ...
#           def detect(self, text: str) -> dict: ...
#
#   `score` must satisfy: larger -> more likely watermarked.
# ===============================================================

from __future__ import annotations

import os
import sys


def _ensure_arena_on_path():
    arena_root = os.environ.get("ARENA_ROOT")
    if arena_root and arena_root not in sys.path:
        sys.path.insert(0, arena_root)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.normpath(os.path.join(here, "..", "..", "WatermarkArena"))
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, os.path.dirname(candidate))


class WatermarkDetector:
    """Competition detector interface."""

    def __init__(self, config: dict):
        self.config = config or {}
        self.algorithm = self.config.get("algorithm", "BlackBoxGreenList")
        self._bundle = None

    def _bundle_obj(self):
        if self._bundle is None:
            _ensure_arena_on_path()
            import arena  # registers everything
            from arena.registry import build_defense

            self._bundle = build_defense(self.algorithm, self.config)
            # Bind a local model if the algorithm needs one. The harness
            # passes `env` separately; for the standalone detector we build
            # a minimal env from config if a local_model.path is provided.
            if self.config.get("local_model", {}).get("path") or os.environ.get("ARENA_LOCAL_MODEL"):
                from arena.config import load_config
                from arena.env import make_env

                env = make_env(load_config())
                _bind = getattr(self._bundle.detector, "bind_env", None)
                if callable(_bind):
                    _bind(env)
        return self._bundle

    def detect(self, text: str) -> dict:
        # return: {"is_watermarked": bool, "score": float}
        bundle = self._bundle_obj()
        det = bundle.detector
        # MarkLLM-backed detectors need an env-bound model; otherwise detect()
        # works directly.
        try:
            return det.detect(text)
        except RuntimeError:
            # Build env on demand and bind.
            from arena.config import load_config
            from arena.env import make_env

            env = make_env(load_config())
            _bind = getattr(det, "bind_env", None)
            if callable(_bind):
                _bind(env)
            return det.detect(text)


Detector = WatermarkDetector
