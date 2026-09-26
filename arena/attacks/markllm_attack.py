# ===============================================================
# markllm_attack.py
# Description: Adapter that wraps MarkLLM's TextEditor attack suite
#              (evaluation/tools/text_editor.py) into the competition's
#              attack interface.
#
#              MarkLLM ships: GPTParaphraser, DipperParaphraser,
#              WordDeletion, SynonymSubstitution,
#              ContextAwareSynonymSubstitution, BackTranslationTextEditor,
#              RandomWalkAttack, ...
#
#              Each is exposed as an independently-registered attack so
#              the confrontation runner can pair it with any defense.
#              Attacks know nothing about defenses (decoupled).
#
#              Robustness: MarkLLM's text_editor.py imports heavy deps
#              (transformers/torch) at module level. When those are not
#              installed, each attack falls back to a self-contained
#              implementation that mirrors MarkLLM's algorithm, so the
#              arena always has working attacks. When the full deps ARE
#              installed, the real MarkLLM editor is used.
# ===============================================================

from __future__ import annotations

import os
import random
from typing import Any

from ..attack_base import BaseAttack
from ..registry import register_attack


def _markllm_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "MarkLLM-main"))


def _ensure_markllm_on_path() -> str:
    import sys

    root = _markllm_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _try_import_markllm_editor(module_path: str, attr: str):
    """Import a MarkLLM TextEditor subclass; return None on failure."""
    try:
        _ensure_markllm_on_path()
        mod = __import__(module_path, fromlist=[attr])
        return getattr(mod, attr)
    except Exception:
        return None


class _TextEditorAttack(BaseAttack):
    """Bridge a MarkLLM TextEditor into the attack() interface."""

    needs_reference = False

    def __init__(self, config: dict | None = None):
        super().__init__(config)

    def _build_editor(self, env: Any):
        raise NotImplementedError

    def attack(self, env: Any, task: dict) -> dict:
        text = task["watermarked_text"]
        try:
            editor = self._build_editor(env)
        except ImportError as e:
            raise RuntimeError(
                f"Attack {self.__class__.__name__} needs MarkLLM dependencies that are "
                f"not installed ({e}). Install them with: pip install -r requirements.txt."
            ) from e
        reference = task.get("reference") or task.get("prompt")
        if self.needs_reference and reference is not None:
            edited = editor.edit(text, reference)
        else:
            edited = editor.edit(text)
        return {"text": edited}


# ------------------------------------------------------------------
# Self-contained fallback editors (mirror MarkLLM's algorithms).
# Used when MarkLLM's text_editor.py cannot be imported.
# ------------------------------------------------------------------
class _WordDeletionEditor:
    def __init__(self, ratio: float):
        self.ratio = ratio

    def edit(self, text, reference=None):
        if not text:
            return text
        words = text.split()
        kept = [w for w in words if random.random() >= self.ratio]
        return " ".join(kept) if kept else " ".join(words)


class _SynonymSubstitutionEditor:
    # Class-level cache so the (network-bounded) wordnet probe runs once.
    _wordnet = None
    _wordnet_probed = False

    def __init__(self, ratio: float):
        self.ratio = ratio
        if not _SynonymSubstitutionEditor._wordnet_probed:
            _SynonymSubstitutionEditor._wordnet_probed = True
            try:
                import socket

                socket.setdefaulttimeout(5)
                import nltk
                from nltk.corpus import wordnet

                try:
                    wordnet.synsets("test")
                    _SynonymSubstitutionEditor._wordnet = wordnet
                except LookupError:
                    try:
                        nltk.download("wordnet", quiet=True)
                        wordnet.synsets("test")
                        _SynonymSubstitutionEditor._wordnet = wordnet
                    except Exception:
                        _SynonymSubstitutionEditor._wordnet = None
                socket.setdefaulttimeout(None)
            except Exception:
                _SynonymSubstitutionEditor._wordnet = None

    @property
    def _wordnet_ok(self):
        return _SynonymSubstitutionEditor._wordnet is not None

    def _synonyms(self, word):
        if not self._wordnet_ok:
            return []
        syns = set()
        for s in _SynonymSubstitutionEditor._wordnet.synsets(word):
            for lemma in s.lemmas():
                if lemma.name().replace("_", " ") != word:
                    syns.add(lemma.name().replace("_", " "))
        return list(syns)

    def edit(self, text, reference=None):
        words = text.split()
        if not words:
            return text
        replaceable = [i for i, w in enumerate(words) if self._synonyms(w)]
        n = min(int(self.ratio * len(words)), len(replaceable))
        if n > 0:
            for i in random.sample(replaceable, n):
                syns = self._synonyms(words[i])
                if syns:
                    words[i] = random.choice(syns)
        return " ".join(words)


class _BackTranslationEditor:
    def __init__(self, intermediary_lang: str = "zh"):
        self.intermediary_lang = intermediary_lang
        self._translator_ok = False
        try:
            from translate import Translator

            self._Translator = Translator
            self._translator_ok = True
        except Exception:
            self._translator_ok = False

    def edit(self, text, reference=None):
        if not self._translator_ok:
            return text
        import socket

        src = "en"
        inter = self.intermediary_lang
        socket.setdefaulttimeout(8)  # bound the translation API call
        try:
            mid = self._Translator(from_lang=src, to_lang=inter).translate(text)
            out = self._Translator(from_lang=inter, to_lang=src).translate(mid)
            socket.setdefaulttimeout(None)
            return out if out else text
        except Exception:
            socket.setdefaulttimeout(None)
            return text


# ------------------------------------------------------------------
# GPT paraphrase attack - uses the standard OpenAI interface via env.
# Fully self-contained (no MarkLLM import needed): the only thing it
# does is call env.generate with a paraphrase prompt.
# ------------------------------------------------------------------
class GPTParaphraseAttack(_TextEditorAttack):
    """LLM paraphrase attack routed through the arena's LLMEnv (OpenAI)."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.model = (config or {}).get("model")
        self.prompt = (config or {}).get(
            "prompt",
            "Rewrite the following text to preserve its full meaning, facts and "
            "task answer, but change the wording and sentence structure as much as "
            "possible. Keep the same language. Output only the rewritten text.\nText:\n",
        )
        self.temperature = float((config or {}).get("temperature", 0.7))
        self.max_tokens = int((config or {}).get("max_tokens", 1024))

    def _build_editor(self, env: Any):
        arena_env = env
        arena_model = self.model
        arena_prompt = self.prompt
        temperature = self.temperature
        max_tokens = self.max_tokens

        class _EnvParaphraser:
            def edit(self, text, reference=None):
                return arena_env.generate(
                    arena_prompt + text,
                    system_content="You are a helpful assistant that rewrites text "
                    "while preserving meaning and factual content.",
                    model=arena_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

        return _EnvParaphraser()


# ------------------------------------------------------------------
# Lightweight attacks: try MarkLLM editor, fall back to self-contained.
# ------------------------------------------------------------------
class WordDeletionAttack(_TextEditorAttack):
    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.ratio = float((config or {}).get("ratio", 0.15))

    def _build_editor(self, env: Any):
        cls = _try_import_markllm_editor("evaluation.tools.text_editor", "WordDeletion")
        if cls is not None:
            return cls(ratio=self.ratio)
        return _WordDeletionEditor(ratio=self.ratio)


class SynonymSubstitutionAttack(_TextEditorAttack):
    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.ratio = float((config or {}).get("ratio", 0.3))

    def _build_editor(self, env: Any):
        cls = _try_import_markllm_editor("evaluation.tools.text_editor", "SynonymSubstitution")
        if cls is not None:
            return cls(ratio=self.ratio)
        return _SynonymSubstitutionEditor(ratio=self.ratio)


class BackTranslationAttack(_TextEditorAttack):
    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.intermediary = (config or {}).get("intermediary_lang", "zh")

    def _build_editor(self, env: Any):
        cls = _try_import_markllm_editor("evaluation.tools.text_editor", "BackTranslationTextEditor")
        if cls is not None:
            from translate import Translator

            inter = self.intermediary
            return cls(
                translate_to_intermediary=Translator(from_lang="en", to_lang=inter).translate,
                translate_to_source=Translator(from_lang=inter, to_lang="en").translate,
            )
        return _BackTranslationEditor(intermediary_lang=self.intermediary)


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------
@register_attack(
    "GPTParaphrase",
    meta={"source": "MarkLLM/GPTParaphraser", "needs_llm": True, "requires_local_model": False},
)
def _gpt_paraphrase_factory(config: dict) -> BaseAttack:
    return GPTParaphraseAttack(config)


@register_attack(
    "WordDeletion",
    meta={"source": "MarkLLM/WordDeletion", "needs_llm": False, "requires_local_model": False},
)
def _word_deletion_factory(config: dict) -> BaseAttack:
    return WordDeletionAttack(config)


@register_attack(
    "SynonymSubstitution",
    meta={"source": "MarkLLM/SynonymSubstitution", "needs_llm": False, "requires_local_model": False},
)
def _synonym_factory(config: dict) -> BaseAttack:
    return SynonymSubstitutionAttack(config)


@register_attack(
    "BackTranslation",
    meta={"source": "MarkLLM/BackTranslation", "needs_llm": False, "requires_local_model": False},
)
def _back_translation_factory(config: dict) -> BaseAttack:
    return BackTranslationAttack(config)
