#!/usr/bin/env python
# ===============================================================
# eval_single.py
# Description: Evaluate a single attack-vs-defense pair (or a single
#              defense alone) and print detailed per-sample output.
#              Handy for debugging and for quick experiments.
#
#   Examples:
#     python scripts/eval_single.py --defense BlackBoxGreenList --max-samples 3
#     python scripts/eval_single.py --attack LLMRewrite --defense BlackBoxGreenList --max-samples 3
# ===============================================================

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import arena  # noqa: E402
from arena.config import load_config, resolve_path  # noqa: E402
from arena.dataset import NaturalTextDataset, PromptDataset  # noqa: E402
from arena.env import make_env  # noqa: E402
from arena.confrontation import ConfrontationRunner  # noqa: E402
from arena.registry import build_attack, build_defense  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default=None)
    ap.add_argument("--defense", required=True)
    ap.add_argument("--max-samples", type=int, default=3)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    env = make_env(cfg)
    ds_cfg = cfg.get("dataset", {})
    dataset = PromptDataset(path=resolve_path(ds_cfg.get("prompts_path", "data/prompts.jsonl")), max_samples=args.max_samples)
    natural = NaturalTextDataset(path=resolve_path(ds_cfg.get("natural_path", "data/natural_texts.jsonl")))

    runner = ConfrontationRunner(env=env, dataset=dataset, natural_texts=natural, config=cfg)

    dcfg = dict(cfg.get("confrontation", {}).get("defense_configs", {}).get(args.defense, {}))
    dcfg.setdefault("name", args.defense)
    bundle = build_defense(args.defense, dcfg)

    if not args.attack:
        rep = runner.evaluate_defense(bundle, max_samples=args.max_samples)
        print("\n===== Defense-only report =====")
        import json
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return

    attack = build_attack(args.attack, cfg.get("confrontation", {}).get("attack_configs", {}).get(args.attack, {}))
    rep = runner.evaluate_pair(attack, bundle, max_samples=args.max_samples)

    print("\n===== Pair report =====")
    for k in ("attack", "defense", "n_samples", "asr", "asr_raw", "mean_semantic_similarity", "final_score", "qualification"):
        print(f"  {k}: {rep.get(k)}")

    print("\n===== Per-sample (watermarked -> attacked) =====")
    cache = runner.prepare_defense(bundle, max_samples=args.max_samples)
    for i, (wm, atk) in enumerate(zip(cache["wm_texts"], rep["attacked_texts"])):
        det_wm = bundle.detector.detect(wm)
        det_atk = bundle.detector.detect(atk)
        print(f"\n--- sample {i} ---")
        print(f"WM  [{det_wm['score']:.3f}, {det_wm['is_watermarked']}]: {wm[:160]}")
        print(f"ATK [{det_atk['score']:.3f}, {det_atk['is_watermarked']}]: {atk[:160]}")


if __name__ == "__main__":
    main()
