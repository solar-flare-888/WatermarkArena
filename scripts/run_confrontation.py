#!/usr/bin/env python
# ===============================================================
# run_confrontation.py
# Description: Main entry point - run the all-pairs Attack x Defense
#              confrontation and print/save the full report.
#
#   Examples:
#     python scripts/run_confrontation.py
#     python scripts/run_confrontation.py --attacks LLMRewrite,Identity \
#         --defenses BlackBoxGreenList --max-samples 10
#     python scripts/run_confrontation.py --config config.local.yaml
# ===============================================================

from __future__ import annotations

import argparse
import json
import os
import sys

# Bootstrap project root onto sys.path so `import arena` works.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import arena  # noqa: E402  (registers all attacks + defenses)
from arena.config import load_config, resolve_path  # noqa: E402
from arena.dataset import NaturalTextDataset, PromptDataset  # noqa: E402
from arena.env import make_env  # noqa: E402
from arena.confrontation import ConfrontationRunner, format_matrix, save_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run the attack-defense confrontation matrix.")
    parser.add_argument("--config", default=None, help="Path to a config.yaml (default: project config.yaml).")
    parser.add_argument("--attacks", default=None, help="Comma-separated attack names (default: all).")
    parser.add_argument("--defenses", default=None, help="Comma-separated defense names (default: all).")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap prompts per run.")
    parser.add_argument("--out", default=None, help="Output report JSON path.")
    parser.add_argument("--no-matrix", action="store_true", help="Only evaluate defenses (skip attack pairs).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = make_env(cfg)

    ds_cfg = cfg.get("dataset", {})
    prompts_path = resolve_path(ds_cfg.get("prompts_path", "data/prompts.jsonl"))
    natural_path = resolve_path(ds_cfg.get("natural_path", "data/natural_texts.jsonl"))
    max_samples = args.max_samples or ds_cfg.get("max_samples")

    dataset = PromptDataset(path=prompts_path, max_samples=max_samples)
    natural = NaturalTextDataset(path=natural_path)
    if len(dataset) == 0:
        print("Dataset is empty. Run `python -m arena.download_datasets` first.")
        sys.exit(1)
    print(f"Loaded {len(dataset)} prompts, {len(natural)} natural texts.")

    conf_cfg = cfg.get("confrontation", {})
    attacks = [a for a in (args.attacks or conf_cfg.get("attacks") or "").split(",") if a] or None
    defenses = [d for d in (args.defenses or conf_cfg.get("defenses") or "").split(",") if d] or None

    runner = ConfrontationRunner(env=env, dataset=dataset, natural_texts=natural, config=cfg)

    if args.no_matrix:
        from arena.registry import build_defense

        reports = []
        for d in (defenses or arena.registry.list_defenses()):
            dcfg = dict(conf_cfg.get("defense_configs", {}).get(d, {}))
            dcfg.setdefault("name", d)
            bundle = build_defense(d, dcfg)
            reports.append(runner.evaluate_defense(bundle, max_samples=max_samples))
        print("\n===== Defense Reports =====")
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return

    report = runner.run_matrix(
        attack_names=attacks,
        defense_names=defenses,
        attack_configs=conf_cfg.get("attack_configs", {}),
        defense_configs=conf_cfg.get("defense_configs", {}),
        max_samples=max_samples,
    )

    print("\n===== ASR Matrix (Attack Success Rate) =====")
    print(format_matrix(report))

    print("\n===== Defense Reports =====")
    for dr in report["defense_reports"]:
        if "error" in dr:
            print(f"  {dr['defense']}: ERROR {dr['error']}")
            continue
        q = dr.get("qualification", {})
        print(
            f"  {dr['defense']}: AUC={dr['auc']:.3f}  TPR@1%FPR={dr['tpr@fpr=0.01']:.3f}  "
            f"quality_retention={dr['quality_retention']:.3f}  qualified={q.get('qualified')}"
        )

    print("\n===== Pair Reports =====")
    for pr in report["pair_reports"]:
        if "error" in pr:
            print(f"  {pr['attack']} vs {pr['defense']}: ERROR {pr['error']}")
            continue
        q = pr.get("qualification", {})
        print(
            f"  {pr['attack']} vs {pr['defense']}: ASR={pr['asr']:.3f}  "
            f"sem_sim={pr['mean_semantic_similarity']:.3f}  qualified={q.get('qualified')}  "
            f"final={pr['final_score']:.3f}"
        )

    out_dir = resolve_path(cfg.get("output", {}).get("report_dir", "outputs"))
    out_path = args.out or os.path.join(out_dir, "confrontation_report.json")
    save_report(report, out_path)


if __name__ == "__main__":
    main()
