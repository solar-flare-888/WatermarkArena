# ===============================================================
# confrontation.py
# Description: All-pairs Attack x Defense confrontation runner
#              (spec section 12: 攻防全对阵机制).
#
#   Pipeline per defense (watermarked texts are generated ONCE and
#   reused across all attacks, since they do not depend on the attack):
#
#     prompt -> defense.generate -> watermarked_text
#                                       |
#                  +--------------------+--------------------+
#                  |                                         |
#          defense.detect (TPR baseline)            attack -> attacked_text
#                  |                                         |
#            originally_detected                   defense.detect -> ASR
#                                                            |
#                                              semantic_sim(wm, attacked)
#
#   Defense-only scoring (no attack):
#     AUC = roc_auc(wm_scores, nw_scores)   # nw = natural + unwatermarked
#     quality retention = sim(wm_text, unwatermarked_text)
#
#   Outputs:
#     - per-defense report (AUC, TPR@FPR, quality retention, qualification)
#     - per (attack, defense) report (ASR, semantic sim, qualification)
#     - ASR matrix (attacks x defenses)
# ===============================================================

from __future__ import annotations

import json
import os
import time
from typing import Sequence

from .dataset import NaturalTextDataset, PromptDataset
from .evaluation import (
    SemanticSimilarity,
    attack_qualified,
    attack_success_rate,
    defense_qualified,
    roc_auc,
    tpr_at_fpr,
)
from .registry import build_attack, build_defense, list_attacks, list_defenses


def _bind_detector(detector, env) -> None:
    """If a detector supports bind_env (MarkLLM detectors), bind it once."""
    bind = getattr(detector, "bind_env", None)
    if callable(bind):
        try:
            bind(env)
        except Exception as e:
            print(f"  [warn] detector bind_env failed: {type(e).__name__}: {e}")


class ConfrontationRunner:
    """Runs the attack-defense confrontation and produces a full report."""

    def __init__(self, env, dataset: PromptDataset | None = None, natural_texts: NaturalTextDataset | None = None,
                 config: dict | None = None):
        self.env = env
        self.dataset = dataset or PromptDataset()
        self.natural = natural_texts or NaturalTextDataset()
        self.config = config or {}
        self.sim = SemanticSimilarity(env=env)
        # Caches: defense_name -> {"wm": [...], "unwm": [...], "wm_scores": [...],
        #                           "nw_scores": [...], "orig_detected": [...]}
        self._defense_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Defense preparation (generation + detection baselines)
    # ------------------------------------------------------------------
    def prepare_defense(self, defense_bundle, max_samples: int | None = None) -> dict:
        """Generate & cache watermarked/unwatermarked texts and detector baselines."""
        name = defense_bundle.name
        if name in self._defense_cache:
            return self._defense_cache[name]

        _bind_detector(defense_bundle.detector, self.env)

        samples = self.dataset.samples
        if max_samples is not None:
            samples = samples[:max_samples]

        wm_texts, unwm_texts = [], []
        print(f"[{name}] generating watermarked + unwatermarked texts ({len(samples)} prompts)...")
        for i, s in enumerate(samples):
            req = {"sample_id": s["sample_id"], "prompt": s["prompt"], "generation_config": {}}
            try:
                wm = defense_bundle.generator.generate(self.env, req)["text"]
            except Exception as e:
                print(f"  [warn] generate failed for {s['sample_id']}: {type(e).__name__}: {e}")
                wm = ""
            try:
                unwm = defense_bundle.generator.generate_unwatermarked(self.env, req)["text"]
            except Exception:
                unwm = wm
            wm_texts.append(wm)
            unwm_texts.append(unwm if unwm else wm)

        # Detector scores
        wm_scores = [defense_bundle.detector.detect(t)["score"] for t in wm_texts]
        orig_detected = [defense_bundle.detector.detect(t)["is_watermarked"] for t in wm_texts]

        # Non-watermarked negatives: natural texts + unwatermarked generations.
        nw_texts = list(self.natural.texts) + unwm_texts
        nw_scores = [defense_bundle.detector.detect(t)["score"] for t in nw_texts]

        cache = {
            "name": name,
            "samples": samples,
            "wm_texts": wm_texts,
            "unwm_texts": unwm_texts,
            "wm_scores": wm_scores,
            "nw_scores": nw_scores,
            "orig_detected": orig_detected,
        }
        self._defense_cache[name] = cache
        return cache

    # ------------------------------------------------------------------
    # Defense-only metrics (spec 14)
    # ------------------------------------------------------------------
    def evaluate_defense(self, defense_bundle, max_samples: int | None = None) -> dict:
        cache = self.prepare_defense(defense_bundle, max_samples)
        wm_scores = cache["wm_scores"]
        nw_scores = cache["nw_scores"]

        auc = roc_auc(wm_scores, nw_scores)
        tpr_1 = tpr_at_fpr(wm_scores, nw_scores, 0.01)
        tpr_5 = tpr_at_fpr(wm_scores, nw_scores, 0.05)

        # Quality retention: similarity between watermarked and unwatermarked text.
        sims = self.sim.similarity(cache["wm_texts"], cache["unwm_texts"])
        retention = float(sum(sims) / len(sims)) if sims else 0.0

        qual = defense_qualified(auc=auc, quality_retention=retention)
        return {
            "defense": defense_bundle.name,
            "n_samples": len(cache["wm_texts"]),
            "n_negatives": len(nw_scores),
            "auc": auc,
            "tpr@fpr=0.01": tpr_1,
            "tpr@fpr=0.05": tpr_5,
            "quality_retention": retention,
            "mean_wm_score": float(sum(wm_scores) / len(wm_scores)) if wm_scores else 0.0,
            "mean_nw_score": float(sum(nw_scores) / len(nw_scores)) if nw_scores else 0.0,
            "qualification": qual,
        }

    # ------------------------------------------------------------------
    # Attack vs Defense (spec 13)
    # ------------------------------------------------------------------
    def evaluate_pair(self, attack, defense_bundle, max_samples: int | None = None) -> dict:
        cache = self.prepare_defense(defense_bundle, max_samples)
        wm_texts = cache["wm_texts"]
        samples = cache["samples"]
        orig_detected = cache["orig_detected"]

        print(f"[{attack.__class__.__name__} vs {defense_bundle.name}] attacking {len(wm_texts)} texts...")
        attacked_texts = []
        for i, wm in enumerate(wm_texts):
            task = {
                "sample_id": samples[i]["sample_id"],
                "watermarked_text": wm,
                "language": samples[i].get("language", ""),
                "reference": samples[i].get("reference"),
                "prompt": samples[i]["prompt"],
            }
            try:
                atk = attack(self.env, task)["text"]
            except Exception as e:
                print(f"  [warn] attack failed on {samples[i]['sample_id']}: {type(e).__name__}: {e}")
                atk = wm
            attacked_texts.append(atk)

        # Detection after attack.
        attacked_flags = [defense_bundle.detector.detect(t)["is_watermarked"] for t in attacked_texts]
        asr_info = attack_success_rate(
            defense_bundle.detector, attacked_texts, originally_detected_mask=orig_detected
        )

        # Semantic similarity between watermarked and attacked text.
        sims = self.sim.similarity(wm_texts, attacked_texts)
        qual = attack_qualified(sims)

        return {
            "attack": attack.__class__.__name__,
            "defense": defense_bundle.name,
            "n_samples": len(wm_texts),
            "asr": asr_info["asr"],
            "asr_raw": asr_info["asr_raw"],
            "mean_semantic_similarity": qual["mean_semantic_similarity"],
            "valid_text_ratio": qual["valid_text_ratio"],
            "qualification": qual,
            # final attack score (spec 13.4): 50% ASR + 50% text retention
            "final_score": 0.5 * asr_info["asr"] + 0.5 * qual["mean_semantic_similarity"],
            "attacked_texts": attacked_texts,  # for inspection / debugging
        }

    # ------------------------------------------------------------------
    # Full matrix
    # ------------------------------------------------------------------
    def run_matrix(
        self,
        attack_names: Sequence[str] | None = None,
        defense_names: Sequence[str] | None = None,
        attack_configs: dict | None = None,
        defense_configs: dict | None = None,
        max_samples: int | None = None,
    ) -> dict:
        attack_names = list(attack_names) if attack_names else list_attacks()
        defense_names = list(defense_names) if defense_names else list_defenses()
        attack_configs = attack_configs or {}
        defense_configs = defense_configs or {}

        # Build defenses first (so each is prepared once).
        defenses = []
        for dn in defense_names:
            cfg = dict(defense_configs.get(dn, {}))
            cfg.setdefault("name", dn)
            defenses.append(build_defense(dn, cfg))

        # Defense-only reports.
        defense_reports = []
        for d in defenses:
            try:
                defense_reports.append(self.evaluate_defense(d, max_samples=max_samples))
            except Exception as e:
                print(f"[!] defense {d.name} eval failed: {type(e).__name__}: {e}")
                defense_reports.append({"defense": d.name, "error": str(e)})

        # Pair reports + ASR matrix.
        pair_reports = []
        asr_matrix: dict[str, dict[str, float]] = {}
        for an in attack_names:
            attack = build_attack(an, attack_configs.get(an, {}))
            asr_matrix[an] = {}
            for d in defenses:
                try:
                    rep = self.evaluate_pair(attack, d, max_samples=max_samples)
                    # Strip the heavy attacked_texts from the stored report.
                    rep_clean = {k: v for k, v in rep.items() if k != "attacked_texts"}
                    pair_reports.append(rep_clean)
                    asr_matrix[an][d.name] = rep["asr"]
                except Exception as e:
                    print(f"[!] pair {an} vs {d.name} failed: {type(e).__name__}: {e}")
                    pair_reports.append({"attack": an, "defense": d.name, "error": str(e)})
                    asr_matrix[an][d.name] = None

        return {
            "attacks": attack_names,
            "defenses": defense_names,
            "defense_reports": defense_reports,
            "pair_reports": pair_reports,
            "asr_matrix": asr_matrix,
        }


def format_matrix(report: dict) -> str:
    """Render the ASR matrix as a text table (spec 12.2)."""
    defenses = report["defenses"]
    attacks = report["attacks"]
    matrix = report["asr_matrix"]
    name_w = max(16, max((len(a) for a in attacks), default=8) + 2)
    col_w = max(10, max((len(d) for d in defenses), default=10) + 1)
    header = "Attack\\Defense".ljust(name_w) + "".join(d.ljust(col_w) for d in defenses)
    lines = [header, "-" * len(header)]
    for a in attacks:
        row = a.ljust(name_w)
        for d in defenses:
            v = matrix.get(a, {}).get(d)
            cell = f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"
            row += cell.ljust(col_w)
        lines.append(row)
    return "\n".join(lines)


def save_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")
