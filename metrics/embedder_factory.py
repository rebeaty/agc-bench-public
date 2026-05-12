"""Embedder factory: returns the active embedding backend.

Backends
--------
  - gemini  : GeminiEmbedder over `gemini-embedding-001` (768-dim).
              Default for release-set fidelity; availability and cost depend
              on the caller's Google API account.
  - qwen    : SentenceTransformer over `Qwen/Qwen3-Embedding-0.6B` (1024-dim).
              Local fallback for runs without a Google embedding API path.
              Auto-detects device (cuda if available, else cpu).
              Override the device via `AGC_QWEN_EMBEDDING_DEVICE` (e.g.
              `cpu`, `cuda`, `cuda:1`).
              Override the model via `AGC_QWEN_EMBEDDING_MODEL` (e.g.
              `Qwen/Qwen3-Embedding-4B` or `Qwen/Qwen3-Embedding-8B`).
  - sentence_transformers : Generic SentenceTransformer with a caller-supplied
              model_name. Use only to reproduce a published number against
              the original benchmark's embedder.

Selection
---------
Set `AGC_EMBEDDING_BACKEND` env var:
    AGC_EMBEDDING_BACKEND=gemini    (default)
    AGC_EMBEDDING_BACKEND=qwen
    AGC_EMBEDDING_BACKEND=sentence_transformers

The legacy `ABC_*` names remain readable as fallbacks so older local
environment files don't break.

Each metric calls `get_embedder(model_name=...)` once at first use and
caches the result. The model_name argument is honored by the
sentence_transformers backend (e.g., "all-mpnet-base-v2") and ignored by
gemini and qwen (which use fixed defaults).
"""
from __future__ import annotations

import os
import threading
from typing import Optional

_lock = threading.Lock()
_cached: dict = {}

_QWEN_DEFAULT = "Qwen/Qwen3-Embedding-0.6B"


def _read_env_with_alias(canonical: str, alias: str, default: str) -> str:
    """Read env var, preferring the canonical AGC_-prefixed name but falling
    back to the legacy ABC_-prefixed alias for backward compatibility with
    pre-rename setups."""
    return os.environ.get(canonical, os.environ.get(alias, default)).strip()


def get_backend_name() -> str:
    return _read_env_with_alias(
        "AGC_EMBEDDING_BACKEND", "ABC_EMBEDDING_BACKEND", "gemini"
    ).lower()


def _detect_device() -> str:
    override = _read_env_with_alias(
        "AGC_QWEN_EMBEDDING_DEVICE", "ABC_QWEN_EMBEDDING_DEVICE", ""
    )
    if override:
        return override
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_embedder(model_name: Optional[str] = None):
    """Return the active embedder. Caches per-process per-model_name.

    Args:
        model_name: SentenceTransformer model identifier. Used only when the
                    backend is `sentence_transformers`. Ignored by gemini and
                    qwen.
    """
    backend = get_backend_name()
    cache_key = (backend, model_name or "default")
    with _lock:
        if cache_key in _cached:
            return _cached[cache_key]
        if backend == "gemini":
            from clients.gemini_embedding_client import GeminiEmbedder
            inst = GeminiEmbedder()
        elif backend == "qwen":
            from sentence_transformers import SentenceTransformer
            qwen_model = _read_env_with_alias(
                "AGC_QWEN_EMBEDDING_MODEL", "ABC_QWEN_EMBEDDING_MODEL", _QWEN_DEFAULT
            )
            inst = SentenceTransformer(qwen_model, device=_detect_device())
            # Qwen3-Embedding defaults to max_seq_length=8192. With batch=32
            # and pad-to-longest, that consumes >60 GiB of activation memory
            # on long CAP responses. CAP outputs are ≤4 K chars (~1 K tokens),
            # so 1024 is a safe ceiling. Override via AGC_QWEN_MAX_SEQ.
            try:
                inst.max_seq_length = int(_read_env_with_alias(
                    "AGC_QWEN_MAX_SEQ", "ABC_QWEN_MAX_SEQ", "1024"
                ))
            except Exception:
                inst.max_seq_length = 1024
        elif backend in {"sentence_transformers", "st", "local"}:
            from sentence_transformers import SentenceTransformer
            inst = SentenceTransformer(model_name or "all-MiniLM-L6-v2",
                                       device=_detect_device())
        else:
            raise ValueError(f"Unknown AGC_EMBEDDING_BACKEND={backend!r}")
        _cached[cache_key] = inst
        return inst
