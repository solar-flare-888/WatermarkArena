# ===============================================================
# config.py
# Description: Load arena configuration from config.yaml with
#              environment-variable overrides for secrets.
# ===============================================================

from __future__ import annotations

import os
from typing import Any

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, ".."))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None = None) -> dict:
    """Load config.yaml and apply environment-variable overrides.

    Env overrides (optional, for secrets / CI):
      OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
      ARENA_LOCAL_MODEL (path), ARENA_DEVICE
    """
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(cfg_path):
        cfg = {}
    else:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Env overrides for the LLM client.
    llm = cfg.setdefault("llm", {})
    if os.environ.get("OPENAI_API_KEY"):
        llm["api_key"] = os.environ["OPENAI_API_KEY"]
    if os.environ.get("OPENAI_BASE_URL"):
        llm["base_url"] = os.environ["OPENAI_BASE_URL"]
    if os.environ.get("OPENAI_MODEL"):
        llm["model"] = os.environ["OPENAI_MODEL"]

    # Env overrides for the local HF model.
    local = cfg.setdefault("local_model", {})
    if os.environ.get("ARENA_LOCAL_MODEL"):
        local["path"] = os.environ["ARENA_LOCAL_MODEL"]
    if os.environ.get("ARENA_DEVICE"):
        local["device"] = os.environ["ARENA_DEVICE"]

    return cfg


def resolve_path(p: str) -> str:
    """Resolve a path that may be relative to the project root."""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(PROJECT_ROOT, p))
