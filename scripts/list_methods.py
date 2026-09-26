#!/usr/bin/env python
# ===============================================================
# list_methods.py
# Description: List all registered attacks and defenses with their
#              metadata (source, whether they need a local model / LLM).
# ===============================================================

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import arena  # noqa: E402
from arena.registry import (  # noqa: E402
    ATTACK_REGISTRY,
    DEFENSE_REGISTRY,
)


def main():
    print("=" * 70)
    print("DEFENSES (generator + detector bundles)")
    print("=" * 70)
    for name in DEFENSE_REGISTRY.names():
        meta = DEFENSE_REGISTRY.meta(name)
        rlm = " [needs local model]" if meta.get("requires_local_model") else " [API-only]"
        src = meta.get("source", "?")
        print(f"  - {name:<22} {src:<14} {rlm}")

    print()
    print("=" * 70)
    print("ATTACKS")
    print("=" * 70)
    for name in ATTACK_REGISTRY.names():
        meta = ATTACK_REGISTRY.meta(name)
        needs_llm = " [uses LLM]" if meta.get("needs_llm") else " [no LLM]"
        src = meta.get("source", "?")
        print(f"  - {name:<22} {src:<14} {needs_llm}")


if __name__ == "__main__":
    main()
