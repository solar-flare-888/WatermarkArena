# ===============================================================
# watermark.py  --  Defense submission template (spec 6.2 / 21.1)
#
#   Implements the exact competition interface:
#       class WatermarkDefense:
#           def __init__(self, config: dict): ...
#           def generate(self, env, request: dict) -> dict: ...
#
#   This template wraps the arena's defense registry, so a submitted
#   defense can reuse any built-in algorithm (KGW, SWEET, ... or the
#   API-native BlackBoxGreenList) simply by setting `algorithm` in
#   config.yaml. Participants may also replace the body of `generate`
#   with a custom logits processor (the competition allows modifying
#   logits during generation).
# ===============================================================

from __future__ import annotations

import os
import sys


def _ensure_arena_on_path():
    """Make the sibling `WatermarkArena` project importable.

    Adjust ARENA_ROOT if you install the arena as a package.
    """
    arena_root = os.environ.get("ARENA_ROOT")
    if arena_root and arena_root not in sys.path:
        sys.path.insert(0, arena_root)
    else:
        # Default: ../WatermarkArena relative to this file.
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.normpath(os.path.join(here, "..", "..", "WatermarkArena"))
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, os.path.dirname(candidate))


class WatermarkDefense:
    """Competition defense interface."""

    def __init__(self, config: dict):
        # config may contain: secret_key, algorithm, algorithm params, ...
        self.config = config or {}
        self.algorithm = self.config.get("algorithm", "BlackBoxGreenList")

        # Lazily import the arena and build the underlying generator.
        self._bundle = None

    def _bundle_obj(self):
        if self._bundle is None:
            _ensure_arena_on_path()
            import arena  # registers everything
            from arena.registry import build_defense

            self._bundle = build_defense(self.algorithm, self.config)
        return self._bundle

    def generate(self, env, request: dict) -> dict:
        # request: {"sample_id": str, "prompt": str, "generation_config": dict}
        # return:  {"text": str}
        return self._bundle_obj().generator.generate(env, request)


# Backwards-compatible alias used by some harnesses.
Defense = WatermarkDefense
