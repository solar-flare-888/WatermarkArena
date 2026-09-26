# ===============================================================
# evaluation.py
# Description: Evaluation engine for the attack-defense arena.
#
#   Implements the metrics defined by the competition spec:
#     - ROC-AUC of the detector score (spec 8.1, 14.1)
#     - TPR @ fixed FPR (spec 8)
#     - Attack Success Rate, ASR (spec 13.2)
#     - Semantic similarity between original & attacked text
#       (qualification line >= 0.85, spec 9.2 / 13.1)
#     - Quality judge via the standard OpenAI interface (spec 9.1)
#
#   Semantic similarity is computed with a pluggable backend:
#     1. sentence-transformers (multilingual) if available
#     2. OpenAI embeddings API via env.generate-compatible client
#     3. Lexical fallback (char-level Jaccard) so the arena always
#        produces a number even fully offline.
# ===============================================================

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------
def roc_auc(wm_scores: Sequence[float], nw_scores: Sequence[float]) -> float:
    """ROC-AUC of detector scores: watermarked (pos) vs non-watermarked (neg)."""
    from sklearn.metrics import roc_auc_score

    y = [1] * len(wm_scores) + [0] * len(nw_scores)
    s = list(wm_scores) + list(nw_scores)
    if len(set(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, s))


def tpr_at_fpr(wm_scores: Sequence[float], nw_scores: Sequence[float], target_fpr: float) -> float:
    """TPR achievable at a given FPR (higher detector score = more positive)."""
    nw = np.sort(np.asarray(nw_scores, dtype=float))
    if len(nw) == 0:
        return 0.0
    # Threshold = the (1-target_fpr) quantile of negative scores.
    k = int(np.ceil((1.0 - target_fpr) * len(nw))) - 1
    k = max(0, min(k, len(nw) - 1))
    threshold = nw[k]
    wm = np.asarray(wm_scores, dtype=float)
    tpr = float(np.mean(wm > threshold)) if len(wm) else 0.0
    return tpr


def attack_success_rate(
    detector, attacked_texts: Sequence[str], originally_detected_mask: Sequence[bool] | None = None
) -> dict:
    """ASR: fraction of attacked texts where the detector no longer fires.

    Returns raw ASR (over all) and conditional ASR (over texts the detector
    originally flagged - the standard, fair definition).
    """
    flags = [not detector.detect(t)["is_watermarked"] for t in attacked_texts]
    raw = float(np.mean(flags)) if flags else 0.0
    if originally_detected_mask is not None:
        sel = [flags[i] for i, m in enumerate(originally_detected_mask) if m]
        cond = float(np.mean(sel)) if sel else 0.0
    else:
        cond = raw
    return {"asr_raw": raw, "asr": cond, "n": len(flags)}


# ------------------------------------------------------------------
# Semantic similarity backend
# ------------------------------------------------------------------
class SemanticSimilarity:
    """Pluggable semantic similarity between two text lists.

    Tries sentence-transformers -> OpenAI embeddings -> lexical fallback.
    """

    def __init__(self, env=None, model_name: str | None = None):
        self.env = env
        self.model_name = model_name or "paraphrase-multilingual-MiniLM-L12-v2"
        self._backend = None
        self._st = None
        self._init_backend()

    def _init_backend(self):
        # 1) sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(self.model_name)
            self._backend = "st"
            return
        except Exception:
            self._st = None
        # 2) OpenAI embeddings via env - only if an embedding model is
        #    configured AND a real OpenAI client is reachable. Otherwise
        #    fall through to the lexical backend so the arena always works.
        if self.env is not None:
            llm_cfg = (getattr(self.env, "config", {}) or {}).get("llm", {}) or {}
            has_emb = bool(llm_cfg.get("embedding_model"))
            # Accept a real api key (non-empty, non-EMPTY sentinel) as a signal
            # that the OpenAI embeddings endpoint is actually usable.
            api_key = llm_cfg.get("api_key") or ""
            key_usable = bool(api_key) and api_key != "EMPTY"
            if has_emb and key_usable:
                self._backend = "openai"
                return
        # 3) lexical fallback (offline, language-agnostic)
        self._backend = "lexical"
        # 3) lexical fallback
        self._backend = "lexical"

    def _embed_openai(self, texts: Sequence[str]) -> np.ndarray:
        client = self.env.client
        model = (self.env.config.get("llm", {}).get("embedding_model")) or "text-embedding-3-small"
        # Embed in small batches.
        out = []
        batch = 64
        for i in range(0, len(texts), batch):
            chunk = [t if t else " " for t in texts[i : i + batch]]
            resp = client.embeddings.create(model=model, input=chunk)
            out.extend([d.embedding for d in resp.data])
        return np.asarray(out, dtype=float)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return (an * bn).sum(axis=1)

    def _lexical(self, a: str, b: str) -> float:
        """Char-level Jaccard fallback (0..1). Offline, language-agnostic."""
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        # Combine word Jaccard with length ratio to soften.
        j = inter / union
        lr = min(len(a), len(b)) / max(len(a), len(b)) if a and b else 0.0
        return float(0.7 * j + 0.3 * lr)

    def similarity(self, texts_a: Sequence[str], texts_b: Sequence[str]) -> list[float]:
        """Return per-pair cosine similarity in [0,1] (clipped)."""
        if len(texts_a) != len(texts_b):
            raise ValueError("text lists must have equal length")
        if len(texts_a) == 0:
            return []
        if self._backend == "st":
            emb = self._st.encode(list(texts_a) + list(texts_b), convert_to_numpy=True)
            a, b = emb[: len(texts_a)], emb[len(texts_a) :]
            sims = self._cosine(a, b)
        elif self._backend == "openai":
            a = self._embed_openai(list(texts_a))
            b = self._embed_openai(list(texts_b))
            sims = self._cosine(a, b)
        else:
            sims = np.asarray([self._lexical(a, b) for a, b in zip(texts_a, texts_b)])
        return [float(max(0.0, min(1.0, s))) for s in sims]


# ------------------------------------------------------------------
# Quality judge (OpenAI interface, spec 9.1)
# ------------------------------------------------------------------
def judge_quality(env, original: str, candidate: str, prompt: str | None = None) -> dict:
    """Ask the LLM (via standard OpenAI interface) to score a candidate text.

    Returns scores in [0,1] for: semantic_consistency, factual_consistency,
    relevance, fluency, completeness, and an overall Q.
    """
    dims = ["semantic_consistency", "factual_consistency", "relevance", "fluency", "completeness"]
    prompt_text = (
        "You are a strict judge. Compare the CANDIDATE text to the ORIGINAL text.\n"
        "Score each dimension from 0 to 1 (1 = perfect). "
        "Be harsh on any factual change or meaning loss.\n\n"
        f"ORIGINAL:\n{original}\n\nCANDIDATE:\n{candidate}\n\n"
        + (f"TASK PROMPT (for reference):\n{prompt}\n\n" if prompt else "")
        + "Return ONLY a JSON object with keys "
        + ", ".join(f'"{d}"' for d in dims)
        + " and \"overall\". Example: "
        + "{" + ", ".join(f'"{d}": 0.9' for d in dims) + ', "overall": 0.9}'
    )
    raw = env.generate(prompt_text, temperature=0.0, max_tokens=200)
    import json as _json
    import re as _re

    try:
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        obj = _json.loads(m.group(0)) if m else {}
    except Exception:
        obj = {}
    result = {d: float(max(0.0, min(1.0, obj.get(d, 0.5)))) for d in dims}
    result["overall"] = float(max(0.0, min(1.0, obj.get("overall", sum(result.values()) / len(dims)))))
    return result


# ------------------------------------------------------------------
# Qualification checks (spec 13.1 / 15)
# ------------------------------------------------------------------
def attack_qualified(similarities: Sequence[float], threshold: float = 0.85, valid_ratio: float = 0.95) -> dict:
    """Attack qualification: mean semantic sim >= threshold AND valid-text ratio >= 0.95."""
    sims = np.asarray(similarities, dtype=float)
    mean_sim = float(sims.mean()) if len(sims) else 0.0
    valid = float(np.mean(sims > 0.1)) if len(sims) else 0.0
    return {
        "qualified": bool(mean_sim >= threshold and valid >= valid_ratio),
        "mean_semantic_similarity": mean_sim,
        "valid_text_ratio": valid,
        "threshold": threshold,
    }


def defense_qualified(auc: float, quality_retention: float, auc_thr: float = 0.8, qual_thr: float = 0.9) -> dict:
    """Defense qualification: AUC > 0.8 AND quality retention >= 0.9 (spec 15)."""
    return {
        "qualified": bool(auc > auc_thr and quality_retention >= qual_thr),
        "auc": auc,
        "quality_retention": quality_retention,
        "auc_threshold": auc_thr,
        "quality_threshold": qual_thr,
    }
