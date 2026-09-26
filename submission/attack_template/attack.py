# ===============================================================
# attack.py  --  Attack submission template (spec 7.2 / 21.2)
#
#   Implements the exact competition interface:
#       def attack(env, task: dict) -> dict: ...
#
#   task:   {"sample_id": str, "watermarked_text": str, "language": str}
#   return: {"text": str}
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


_CONFIG = {
    # Choose a built-in attack: LLMRewrite, LLMRoundTrip, GPTParaphrase,
    # SynonymSubstitution, WordDeletion, BackTranslation, Identity
    "attack": "LLMRewrite",
    "strength": "medium",   # low | medium | high
    "temperature": 0.9,
}


def _load_yaml_config():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "config.yaml")
    if not os.path.exists(path):
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_ATTACK = None


def _get_attack():
    global _ATTACK
    if _ATTACK is None:
        _ensure_arena_on_path()
        import arena  # registers everything
        from arena.registry import build_attack

        cfg = _load_yaml_config()
        attack_name = cfg.pop("attack", _CONFIG["attack"])
        _ATTACK = build_attack(attack_name, cfg)
    return _ATTACK


def attack(env, task: dict) -> dict:
    """Remove/weaken the watermark while preserving semantics.

    The attack only sees the watermarked text + language - never the
    defense key, detector threshold, or defense source (decoupled).
    """
    return _get_attack()(env, task)


# Allow `python attack.py` for a quick self-test.
if __name__ == "__main__":
    from arena.config import load_config
    from arena.env import make_env

    env = make_env(load_config())
    demo = {
        "sample_id": "demo",
        "watermarked_text": (
            "Climate change is primarily driven by the increase of greenhouse "
            "gases such as carbon dioxide and methane in the atmosphere, largely "
            "due to human activities like burning fossil fuels and deforestation."
        ),
        "language": "en",
    }
    out = attack(env, demo)
    print("Attacked text:\n", out["text"])
