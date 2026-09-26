#!/usr/bin/env python
# ===============================================================
# smoke_test.py
# Description: End-to-end smoke test of the arena that does NOT
#              require a real OpenAI key. It plugs a MockLLMEnv
#              (returns deterministic varied text) into the full
#              confrontation pipeline:
#
#                BlackBoxGreenList defense  x  {Identity, LLMRewrite}
#
#              and checks that every metric (AUC, ASR, semantic sim,
#              qualification) is produced. Run with:
#                  python examples/smoke_test.py
# ===============================================================

from __future__ import annotations

import os
import sys
import random

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import arena  # noqa: E402
from arena.confrontation import ConfrontationRunner, format_matrix  # noqa: E402
from arena.dataset import NaturalTextDataset, PromptDataset  # noqa: E402
from arena.registry import build_attack, build_defense  # noqa: E402


class MockLLMEnv:
    """A stand-in for the OpenAI-interface env that returns canned text.

    Implements the same surface the arena uses: `generate(prompt, ...)`,
    `has_local_model()`, and a `.client`/`.config` for the similarity
    backend's OpenAI-embedding fallback (which we disable by setting no env).
    """

    def __init__(self):
        self.config = {"llm": {}}
        self._rng = random.Random(42)
        # A small canned vocabulary so the green-list signal is real.
        self._vocab = [
            "the", "system", "model", "data", "learning", "language", "watermark",
            "text", "detection", "network", "compute", "token", "score", "attack",
            "defense", "quality", "metric", "algorithm", "robust", "signal",
            "green", "list", "hash", "key", "probability", "statistics", "sample",
            "result", "evaluation", "performance", "knowledge", "reasoning",
        ]

    def has_local_model(self):
        return False

    def generate(self, prompt, system_content=None, model=None, temperature=0.7,
                 max_tokens=512, stop=None, **kwargs):
        # Produce a deterministic-but-varied response whose token distribution
        # depends on the prompt + a per-call jitter, so best-of-N selection
        # and LLMRewrite both have something realistic to operate on.
        seed = abs(hash(prompt)) ^ int(temperature * 1000)
        rng = random.Random(seed)
        n = rng.randint(40, 70)
        words = [rng.choice(self._vocab) for _ in range(n)]
        # Echo a snippet of the prompt to keep some semantic overlap.
        head = (prompt or "").strip().split()[:8]
        text = " ".join(head + words)
        return text


def main():
    print("== Smoke test: BlackBoxGreenList x {Identity, LLMRewrite} (mock env) ==")
    env = MockLLMEnv()

    dataset = PromptDataset(max_samples=8)
    natural = NaturalTextDataset(max_samples=8)
    print(f"Loaded {len(dataset)} prompts, {len(natural)} natural texts.")

    runner = ConfrontationRunner(env=env, dataset=dataset, natural_texts=natural)

    report = runner.run_matrix(
        attack_names=["Identity", "LLMRewrite"],
        defense_names=["BlackBoxGreenList"],
        max_samples=8,
    )

    print("\n== ASR Matrix ==")
    print(format_matrix(report))

    print("\n== Defense report ==")
    for dr in report["defense_reports"]:
        if "error" in dr:
            print("ERROR:", dr["error"]); continue
        print(f"  AUC={dr['auc']:.3f}  TPR@1%FPR={dr['tpr@fpr=0.01']:.3f}  "
              f"quality_retention={dr['quality_retention']:.3f}  qualified="
              f"{dr['qualification']['qualified']}")

    print("\n== Pair reports ==")
    ok = True
    for pr in report["pair_reports"]:
        if "error" in pr:
            print("ERROR:", pr["error"]); ok = False; continue
        print(f"  {pr['attack']} vs {pr['defense']}: ASR={pr['asr']:.3f}  "
              f"sem_sim={pr['mean_semantic_similarity']:.3f}  final={pr['final_score']:.3f}  "
              f"qualified={pr['qualification']['qualified']}")
        # Sanity: Identity attack should have ~0 ASR (no change -> detector still fires).
        if pr["attack"] == "Identity" and pr["asr"] > 0.5:
            print("  [warn] Identity ASR unexpectedly high"); ok = False

    print("\nSmoke test", "PASSED" if ok else "had warnings")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
