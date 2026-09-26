# ===============================================================
# download_datasets.py
# Description: Build the competition Prompt dataset **locally**.
#
#   Output schema (per spec section 5.1):
#       {
#         "sample_id": str,
#         "prompt": str,
#         "task_type": str,
#         "language": str,
#         "reference": str | None
#       }
#
#   Sources, in priority order (each is best-effort; failures are
#   logged and skipped so the script always produces a usable file):
#     1. A built-in curated prompt set covering EVERY required task
#        type (open QA, summarization, rewriting, knowledge QA,
#        reasoning, creative writing, structured generation, long
#        text, Chinese-English mixed). Always available, no network.
#     2. HuggingFace `datasets` downloads (trivia_qa, gsm8k,
#        cnn_dailymail, ...) - fetched and cached locally when the
#        network is reachable.
#     3. MarkLLM's bundled local datasets (C4, CNN/DailyMail,
#        HumanEval, WMT16) reformatted into the schema.
#
#   Usage:
#       python -m arena.download_datasets --out data/prompts.jsonl
#       python -m arena.download_datasets --max-per-source 50
# ===============================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable, Optional

# ------------------------------------------------------------------
# Output location
# ------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, ".."))
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data", "prompts.jsonl")
NATURAL_OUT = os.path.join(PROJECT_ROOT, "data", "natural_texts.jsonl")


def _markllm_root() -> str:
    return os.path.normpath(os.path.join(PROJECT_ROOT, "..", "MarkLLM-main"))


# ------------------------------------------------------------------
# 1. Built-in curated prompt set (always available, no network).
#    Covers every task type required by the spec.
# ------------------------------------------------------------------
CURATED_PROMPTS = [
    # --- open-ended QA ---
    {"prompt": "What are the main causes of climate change? Explain in a few sentences.",
     "task_type": "open_qa", "language": "en", "reference": None},
    {"prompt": "How does the human immune system respond to a viral infection?",
     "task_type": "open_qa", "language": "en", "reference": None},
    # --- knowledge QA ---
    {"prompt": "Who invented the telephone, and in what year?",
     "task_type": "knowledge_qa", "language": "en", "reference": "Alexander Graham Bell, 1876"},
    {"prompt": "What is the chemical formula of glucose?",
     "task_type": "knowledge_qa", "language": "en", "reference": "C6H12O6"},
    # --- reasoning ---
    {"prompt": "If a train travels 60 km in 45 minutes, what is its average speed in km/h? Show your reasoning.",
     "task_type": "reasoning", "language": "en", "reference": "80 km/h"},
    {"prompt": "Three friends split a bill of $87. If they want to leave a 15% tip, how much does each pay?",
     "task_type": "reasoning", "language": "en", "reference": "$33.35"},
    # --- summarization ---
    {"prompt": "Summarize the following article in two sentences: "
               "The James Webb Space Telescope (JWST) is the largest optical telescope in space. "
               "Its high resolution and sensitivity allow it to view objects too old, distant, "
               "or faint for the Hubble Space Telescope. Launched in December 2021, JWST orbits "
               "the Sun near the second Lagrange point and observes in the near-infrared spectrum.",
     "task_type": "summarization", "language": "en", "reference": None},
    # --- rewriting / paraphrase ---
    {"prompt": "Rewrite the following sentence to make it more concise: "
               "'Due to the fact that the weather conditions were not favorable, the event was "
               "postponed by the organizers to a later date.'",
     "task_type": "rewriting", "language": "en", "reference": "The organizers postponed the event due to bad weather."},
    # --- creative writing ---
    {"prompt": "Write a short science fiction story (about 150 words) about a city that floats above the clouds.",
     "task_type": "creative_writing", "language": "en", "reference": None},
    {"prompt": "Compose a short poem about the changing of seasons.",
     "task_type": "creative_writing", "language": "en", "reference": None},
    # --- structured generation ---
    {"prompt": "List three benefits of regular exercise as a JSON object with a 'benefits' array of objects "
               "each having 'title' and 'description' fields.",
     "task_type": "structured_generation", "language": "en", "reference": None},
    {"prompt": "Write a Python function that returns the n-th Fibonacci number. Include a docstring and type hints.",
     "task_type": "structured_generation", "language": "en", "reference": None},
    # --- long text generation ---
    {"prompt": "Write a 300-word essay discussing the social impacts of artificial intelligence on employment.",
     "task_type": "long_text", "language": "en", "reference": None},
    # --- Chinese-English mixed / Chinese tasks ---
    {"prompt": "请用中文简要解释什么是大语言模型，并列出三个主要应用场景。",
     "task_type": "knowledge_qa", "language": "zh", "reference": None},
    {"prompt": "请把下面这段话改写得更简洁，但保留全部含义："
               "'由于天气条件不太理想的原因，主办方决定将本次活动推迟到另一个较晚的日期举行。'",
     "task_type": "rewriting", "language": "zh", "reference": "因天气不佳，主办方推迟了活动。"},
    {"prompt": "写一首关于秋天的短诗。",
     "task_type": "creative_writing", "language": "zh", "reference": None},
    {"prompt": "请写一段约200字的中文短文，论述阅读习惯对个人成长的影响。",
     "task_type": "long_text", "language": "zh", "reference": None},
    {"prompt": "Translate the following English sentence into Chinese and then explain its meaning in English: "
               "'Knowledge is power.'",
     "task_type": "open_qa", "language": "中英", "reference": "知识就是力量。"},
    {"prompt": "用中英混合的方式介绍你自己（假设你是一个AI助手），约100字。",
     "task_type": "creative_writing", "language": "中英", "reference": None},
    {"prompt": "If a box contains 5 red, 3 blue, and 2 green balls, what is the probability of drawing a blue ball? "
               "请用中文回答并给出推理过程。",
     "task_type": "reasoning", "language": "中英", "reference": "3/10 = 0.3"},
]


def _curated_samples() -> Iterable[dict]:
    for i, p in enumerate(CURATED_PROMPTS):
        yield {
            "sample_id": f"curated-{i:04d}",
            "prompt": p["prompt"],
            "task_type": p["task_type"],
            "language": p["language"],
            "reference": p.get("reference"),
        }


# ------------------------------------------------------------------
# 2. HuggingFace datasets (best-effort, cached locally by the HF lib)
# ------------------------------------------------------------------
def _try_hf(max_per_source: int) -> Iterable[dict]:
    """Download a few small HF datasets and convert to schema. Best-effort."""
    sources = [
        ("trivia_qa", "rc.nocontext", "knowledge_qa", "en", _triviaqa_conv),
        ("gsm8k", "main", "reasoning", "en", _gsm8k_conv),
        ("openbookqa", "main", "reasoning", "en", _openbookqa_conv),
        ("cnn_dailymail", "3.0.0", "summarization", "en", _cnn_conv),
    ]
    for name, config, task_type, lang, conv in sources:
        try:
            from datasets import load_dataset

            ds = load_dataset(name, config, split="train", streaming=True)
            count = 0
            sid = name
            for row in ds:
                try:
                    converted = conv(row)
                except Exception:
                    converted = None
                if converted is None:
                    continue
                yield {
                    "sample_id": f"{sid}-{count:05d}",
                    "prompt": converted["prompt"],
                    "task_type": task_type,
                    "language": lang,
                    "reference": converted.get("reference"),
                }
                count += 1
                if count >= max_per_source:
                    break
            print(f"  [HF] {name}: +{count} samples")
        except Exception as e:  # network / dataset issues
            print(f"  [HF] {name}: skipped ({type(e).__name__}: {str(e)[:80]})")


def _triviaqa_conv(row):
    q = row.get("question")
    a = row.get("answer", {})
    ref = a.get("value") if isinstance(a, dict) else None
    return {"prompt": q, "reference": ref} if q else None


def _gsm8k_conv(row):
    q = row.get("question")
    a = row.get("answer", "")
    # Extract final numeric answer after "####"
    ref = a.split("####")[-1].strip() if "####" in a else None
    return {"prompt": q, "reference": ref} if q else None


def _openbookqa_conv(row):
    q = row.get("question_stem")
    choices = row.get("choices", {})
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    ans = row.get("answerKey")
    if not q or not texts:
        return None
    options = " ".join(f"({l}) {t}" for l, t in zip(labels, texts))
    prompt = f"{q}\nOptions: {options}\nGive the correct option label."
    ref = ans
    return {"prompt": prompt, "reference": ref}


def _cnn_conv(row):
    art = row.get("article")
    if not art:
        return None
    prompt = "Please summarize the following article:\n" + art[:2000]
    return {"prompt": prompt, "reference": row.get("highlights")}


# ------------------------------------------------------------------
# 3. MarkLLM's bundled local datasets -> schema
# ------------------------------------------------------------------
def _markllm_local(max_per_source: int) -> Iterable[dict]:
    root = _markllm_root()
    plans = [
        ("c4", os.path.join(root, "dataset", "c4", "processed_c4.json"), "open_qa", "en", _c4_local_conv),
        ("cnn", os.path.join(root, "dataset", "cnn_dailymail", "test-00000-of-00001.jsonl"), "summarization", "en", _cnn_local_conv),
        ("humaneval", os.path.join(root, "dataset", "human_eval", "test.jsonl"), "structured_generation", "en", _humaneval_local_conv),
        ("wmt16", os.path.join(root, "dataset", "wmt16_de_en", "validation.jsonl"), "rewriting", "en", _wmt_local_conv),
    ]
    for sid, path, task_type, lang, conv in plans:
        if not os.path.exists(path):
            print(f"  [MarkLLM] {sid}: file not found ({path})")
            continue
        count = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    converted = conv(row)
                    if converted is None:
                        continue
                    yield {
                        "sample_id": f"{sid}-{count:05d}",
                        "prompt": converted["prompt"],
                        "task_type": task_type,
                        "language": lang,
                        "reference": converted.get("reference"),
                    }
                    count += 1
                    if count >= max_per_source:
                        break
            print(f"  [MarkLLM] {sid}: +{count} samples")
        except Exception as e:
            print(f"  [MarkLLM] {sid}: error ({type(e).__name__}: {str(e)[:80]})")


def _c4_local_conv(row):
    p = row.get("prompt")
    nat = row.get("natural_text")
    return {"prompt": p, "reference": nat} if p else None


def _cnn_local_conv(row):
    art = row.get("article")
    if not art:
        return None
    return {"prompt": "Please summarize the following article: " + art[:2000], "reference": row.get("highlights")}


def _humaneval_local_conv(row):
    prompt = row.get("prompt")
    return {"prompt": prompt, "reference": None} if prompt else None


def _wmt_local_conv(row):
    de = row.get("de")
    en = row.get("en")
    if not de:
        return None
    return {"prompt": f"Translate the following German text into English:\n{de}", "reference": en}


# ------------------------------------------------------------------
# Non-watermarked (natural) texts for FPR testing (spec 5.2 / 20.1)
# ------------------------------------------------------------------
def _collect_natural_texts(max_n: int = 200) -> list[str]:
    root = _markllm_root()
    path = os.path.join(root, "dataset", "c4", "processed_c4.json")
    texts: list[str] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    nat = row.get("natural_text")
                    if nat:
                        texts.append(nat)
                    if len(texts) >= max_n:
                        break
        except Exception:
            pass
    return texts


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def build_dataset(out_path: str = DEFAULT_OUT, max_per_source: int = 50, include_hf: bool = True) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    samples: list[dict] = []

    print("Building competition prompt dataset ->", out_path)
    print("[1/3] Curated prompts (always available)...")
    samples.extend(_curated_samples())

    if include_hf:
        print("[2/3] HuggingFace datasets (best-effort)...")
        try:
            samples.extend(_try_hf(max_per_source))
        except Exception as e:
            print(f"  HF step skipped entirely: {type(e).__name__}: {e}")
    else:
        print("[2/3] HuggingFace datasets: disabled")

    print("[3/3] MarkLLM bundled local datasets...")
    samples.extend(_markllm_local(max_per_source))

    # Deduplicate by (prompt) to avoid exact duplicates across sources.
    seen = set()
    deduped = []
    for s in samples:
        key = s["prompt"].strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    with open(out_path, "w", encoding="utf-8") as f:
        for s in deduped:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Natural (non-watermarked) texts for FPR
    nats = _collect_natural_texts()
    with open(NATURAL_OUT, "w", encoding="utf-8") as f:
        for t in nats:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    # Stats
    by_type: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    for s in deduped:
        by_type[s["task_type"]] = by_type.get(s["task_type"], 0) + 1
        by_lang[s["language"]] = by_lang.get(s["language"], 0) + 1
    print(f"\nWrote {len(deduped)} unique prompts to {out_path}")
    print(f"Wrote {len(nats)} natural texts to {NATURAL_OUT}")
    print("By task_type:", dict(sorted(by_type.items())))
    print("By language:", dict(sorted(by_lang.items())))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download/build the competition prompt dataset locally.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSONL path for prompts.")
    parser.add_argument("--max-per-source", type=int, default=50, help="Max samples per external source.")
    parser.add_argument("--no-hf", action="store_true", help="Skip HuggingFace downloads (offline mode).")
    args = parser.parse_args()
    build_dataset(out_path=args.out, max_per_source=args.max_per_source, include_hf=not args.no_hf)


if __name__ == "__main__":
    main()
