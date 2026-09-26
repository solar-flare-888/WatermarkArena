# ===============================================================
# api_native_defense.py
# Description: A defense that works *purely* through the standard
#              OpenAI interface - no local model, no logits access.
#
#              Technique: black-box "best-of-N green-list" watermark.
#              1. Generate N candidate completions from the LLM
#                 (env.generate, OpenAI chat completions, temperature>0).
#              2. Score each candidate by a green-list z-score computed
#                 from a secret-keyed PRF over tokens (KGW-style statistic,
#                 but evaluated on the *output text* rather than logits).
#              3. Return the candidate with the highest z-score.
#
#              Detection re-computes the same z-score; larger score ->
#              more likely watermarked (spec compliant).
#
#              This is a genuine black-box watermark (selection biases the
#              emitted token distribution toward the green list) and lets
#              the whole attack-defense loop run end-to-end with nothing
#              but an OpenAI-compatible endpoint. When a local model is
#              available, prefer the MarkLLM logits-based defenses
#              (registered separately) for a stronger signal.
# ===============================================================

from __future__ import annotations

import hashlib
from typing import Any, List

from ..defense_base import DefenseBundle, WatermarkDefense, WatermarkDetector
from ..registry import register_defense


# ------------------------------------------------------------------
# Tokenizer abstraction: prefer tiktoken (BPE, matches OpenAI), else
# fall back to whitespace word splitting. The only requirement is that
# generation-scoring and detection use the *same* tokenizer.
# ------------------------------------------------------------------
class _Tokenizer:
    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self._tk = None
        if backend in ("auto", "tiktoken"):
            try:
                import tiktoken

                self._tk = tiktoken.get_encoding("cl100k_base")
                self.backend = "tiktoken"
            except Exception:
                if backend == "tiktoken":
                    raise
                self.backend = "whitespace"

    def encode(self, text: str) -> List[int]:
        if self._tk is not None:
            return self._tk.encode(text)
        # Whitespace fallback: hash each word to a stable int id.
        return [int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % (1 << 24) for w in text.split()]


class GreenListScorer:
    """KGW-style green-list z-score evaluated on token ids.

    greenness(token_i) = (PRF(secret_key, token_{i-1}) + token_i) mod V < gamma * V
    The z-score compares the observed green fraction against the null (gamma).
    """

    def __init__(self, secret_key: int, gamma: float = 0.5, prefix_length: int = 1, vocab_size: int = 1 << 24):
        self.secret_key = int(secret_key)
        self.gamma = float(gamma)
        self.prefix_length = int(prefix_length)
        self.vocab_size = int(vocab_size)
        self.green_size = int(self.vocab_size * self.gamma)
        self._tokenizer = _Tokenizer()

    def _prf(self, prev_token: int) -> int:
        """Deterministic PRF keyed by the secret, mixing the previous token."""
        h = hashlib.sha256(f"{self.secret_key}|{prev_token}".encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big") % self.vocab_size

    def _is_green(self, prev_token: int, token: int) -> bool:
        """O(1) green-list membership check.

        The green list for a given previous token is the contiguous block
        { (offset + k) mod V | 0 <= k < green_size }. A token `t` is green
        iff ((t - offset) mod V) < green_size. This avoids materializing an
        8M-element set per token (vocab_size = 2**24).
        """
        offset = self._prf(prev_token)
        return ((token - offset) % self.vocab_size) < self.green_size

    def score(self, text: str) -> tuple[float, int, int]:
        """Return (z_score, green_count, num_scored)."""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= self.prefix_length:
            return 0.0, 0, 0
        green_count = 0
        num_scored = len(tokens) - self.prefix_length
        for i in range(self.prefix_length, len(tokens)):
            if self._is_green(tokens[i - 1], tokens[i]):
                green_count += 1
        # z-score under null Bernoulli(gamma)
        from math import sqrt

        denom = sqrt(num_scored * self.gamma * (1 - self.gamma))
        z = (green_count - self.gamma * num_scored) / denom if denom > 0 else 0.0
        return z, green_count, num_scored


class BlackBoxGreenListDefense(WatermarkDefense):
    """Best-of-N green-list watermark using only env.generate (OpenAI API)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.secret_key = int(config.get("secret_key", 15485863))
        self.gamma = float(config.get("gamma", 0.5))
        self.num_candidates = int(config.get("num_candidates", 4))
        self.temperature = float(config.get("temperature", 0.8))
        self.max_tokens = int(config.get("max_tokens", 256))
        self.prefix_length = int(config.get("prefix_length", 1))
        self.system_content = config.get(
            "system_content", "You are a helpful assistant. Answer the user's request."
        )
        self.scorer = GreenListScorer(
            secret_key=self.secret_key,
            gamma=self.gamma,
            prefix_length=self.prefix_length,
        )

    def _candidate(self, env, prompt: str) -> str:
        return env.generate(
            prompt,
            system_content=self.system_content,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def generate(self, env: Any, request: dict) -> dict:
        prompt = request["prompt"]
        gen_cfg = request.get("generation_config", {}) or {}
        n = int(gen_cfg.get("num_candidates", self.num_candidates))
        temp = float(gen_cfg.get("temperature", self.temperature))

        if n <= 1:
            text = self._candidate(env, prompt) if temp == self.temperature else env.generate(
                prompt, system_content=self.system_content, temperature=temp, max_tokens=self.max_tokens
            )
            return {"text": text}

        # Best-of-N: keep the candidate with the highest green-list z-score.
        best_text, best_z = "", -1e9
        for _ in range(n):
            cand = env.generate(
                prompt,
                system_content=self.system_content,
                temperature=temp,
                max_tokens=self.max_tokens,
            )
            if not cand.strip():
                continue
            z, _, _ = self.scorer.score(cand)
            if z > best_z:
                best_z, best_text = z, cand
        return {"text": best_text if best_text.strip() else cand}


class BlackBoxGreenListDetector(WatermarkDetector):
    """Detector for the best-of-N green-list watermark (same secret key)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.secret_key = int(config.get("secret_key", 15485863))
        self.gamma = float(config.get("gamma", 0.5))
        self.prefix_length = int(config.get("prefix_length", 1))
        self.z_threshold = float(config.get("z_threshold", 4.0))
        self.scorer = GreenListScorer(
            secret_key=self.secret_key,
            gamma=self.gamma,
            prefix_length=self.prefix_length,
        )

    def detect(self, text: str) -> dict:
        z, _, _ = self.scorer.score(text)
        return {
            "is_watermarked": z > self.z_threshold,
            "score": float(z),
        }


@register_defense(
    "BlackBoxGreenList",
    meta={"algorithm": "BlackBoxGreenList", "requires_local_model": False, "source": "API-native"},
)
def _bb_greenlist_factory(config: dict) -> DefenseBundle:
    cfg = dict(config or {})
    generator = BlackBoxGreenListDefense(cfg)
    detector = BlackBoxGreenListDetector(cfg)
    return DefenseBundle(
        name=cfg.get("name", "BlackBoxGreenList"),
        generator=generator,
        detector=detector,
        meta={"algorithm": "BlackBoxGreenList", "requires_local_model": False},
    )
