# ===============================================================
# dataset.py
# Description: Load the locally-built competition Prompt dataset
#              (data/prompts.jsonl) and the non-watermarked natural
#              text set (data/natural_texts.jsonl) for FPR testing.
#
#              Each prompt sample follows the spec (section 5.1):
#                  {sample_id, prompt, task_type, language, reference}
# ===============================================================

from __future__ import annotations

import json
import os
from typing import Iterator, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, ".."))
DEFAULT_PROMPTS = os.path.join(PROJECT_ROOT, "data", "prompts.jsonl")
DEFAULT_NATURAL = os.path.join(PROJECT_ROOT, "data", "natural_texts.jsonl")


class PromptDataset:
    """Iterable competition Prompt dataset.

    Lazily reads the JSONL produced by `download_datasets.py`.
    """

    def __init__(self, path: str = DEFAULT_PROMPTS, max_samples: Optional[int] = None):
        self.path = path
        self.max_samples = max_samples
        self._samples: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self._samples is None:
            if not os.path.exists(self.path):
                raise FileNotFoundError(
                    f"Prompt dataset not found at {self.path}. "
                    f"Run `python -m arena.download_datasets` first."
                )
            samples = []
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    samples.append(json.loads(line))
            if self.max_samples is not None:
                samples = samples[: self.max_samples]
            self._samples = samples
        return self._samples

    @property
    def samples(self) -> list[dict]:
        return self._load()

    def __len__(self) -> int:
        return len(self._load())

    def __iter__(self) -> Iterator[dict]:
        return iter(self._load())

    def __getitem__(self, idx: int) -> dict:
        return self._load()[idx]

    def by_task_type(self, task_type: str) -> list[dict]:
        return [s for s in self._load() if s.get("task_type") == task_type]

    def by_language(self, language: str) -> list[dict]:
        return [s for s in self._load() if s.get("language") == language]


class NaturalTextDataset:
    """Non-watermarked text set (human / unwatermarked) for FPR testing."""

    def __init__(self, path: str = DEFAULT_NATURAL, max_samples: Optional[int] = None):
        self.path = path
        self.max_samples = max_samples
        self._texts: list[str] | None = None

    def _load(self) -> list[str]:
        if self._texts is None:
            texts: list[str] = []
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        t = obj.get("text") if isinstance(obj, dict) else None
                        if t:
                            texts.append(t)
            if self.max_samples is not None:
                texts = texts[: self.max_samples]
            self._texts = texts
        return self._texts

    @property
    def texts(self) -> list[str]:
        return self._load()

    def __len__(self) -> int:
        return len(self._load())

    def __iter__(self) -> Iterator[str]:
        return iter(self._load())

    def __getitem__(self, idx: int) -> str:
        return self._load()[idx]
