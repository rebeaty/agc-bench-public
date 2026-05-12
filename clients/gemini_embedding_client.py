"""SentenceTransformer-compatible adapter over Google's Gemini embeddings API.

Why this exists
---------------
The production sweep wants embedding calls to go through `gemini-embedding-001`
rather than local SentenceTransformer / AutoModel models. Reasons:
  - Cost: free under the Google credit allotment available to the project; paid HF/local has either bandwidth
    cost (HF Hub download) or OOM risk on Slurm tasks.
  - Reliability: removes HF Hub cold-start dependency on production launch.
  - Reproducibility: a single fixed model version (`gemini-embedding-001`,
    `output_dimensionality=768`) across all metrics and the entire sweep.

Metric compatibility note
-------------------------
Most upstream creativity benchmarks (e.g., aidanbench, mops, semantic-
diversity, sdat) report cosine similarity computed against
`all-MiniLM-L6-v2` or `all-mpnet-base-v2` embeddings. Switching to
`gemini-embedding-001` (768-dim) breaks numerical comparability with those
papers' published numbers. The within-our-sweep ranking of models against
each other is preserved because all models are scored under the same
embedder. The release scoring notes disclose the embedder substitution.

API contract
------------
`GeminiEmbedder` mimics SentenceTransformer's surface so existing metrics
need a one-line import swap rather than a logic rewrite:

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts, convert_to_numpy=True)

becomes

    from metrics.embedder_factory import get_embedder
    model = get_embedder()
    embs = model.encode(texts, convert_to_numpy=True)

The factory returns either GeminiEmbedder or a real SentenceTransformer
based on `ABC_EMBEDDING_BACKEND` (default: gemini).
"""
from __future__ import annotations

import os
import random
import threading
import time
from functools import lru_cache
from typing import Iterable, List, Sequence, Union

import numpy as np

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as e:
    raise ImportError(
        "GeminiEmbedder requires the google-genai package. "
        "Install via `uv pip install google-genai`."
    ) from e


_DEFAULT_MODEL = os.environ.get("ABC_GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
_DEFAULT_DIM = int(os.environ.get("ABC_GEMINI_EMBEDDING_DIM", "768"))
_MAX_RETRIES = int(os.environ.get("ABC_GEMINI_EMBEDDING_MAX_RETRIES", "8"))
_RETRY_BASE = float(os.environ.get("ABC_GEMINI_EMBEDDING_RETRY_BASE_SECONDS", "2.0"))
_RETRY_MAX = float(os.environ.get("ABC_GEMINI_EMBEDDING_RETRY_MAX_SECONDS", "60.0"))
_CONCURRENCY = max(1, int(os.environ.get("ABC_GEMINI_EMBEDDING_CONCURRENCY", "1")))

_client_lock = threading.Lock()
_client: genai.Client | None = None
_semaphore = threading.BoundedSemaphore(_CONCURRENCY)


def _get_client() -> genai.Client:
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY is required for Gemini embeddings (ABC_EMBEDDING_BACKEND=gemini)"
                )
            _client = genai.Client(api_key=api_key)
        return _client


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    text = str(exc).upper()
    return status in {429, 500, 503} or "RESOURCE_EXHAUSTED" in text or "UNAVAILABLE" in text


def _sleep_backoff(attempt: int) -> None:
    delay = min(_RETRY_MAX, _RETRY_BASE * (2 ** attempt))
    time.sleep(delay + random.uniform(0.0, min(1.0, delay * 0.1)))


@lru_cache(maxsize=20000)
def _embed_one(text: str, model_name: str, dim: int) -> tuple[float, ...]:
    """Embed a single text. Cached so repeat calls don't re-bill the API."""
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            with _semaphore:
                response = _get_client().models.embed_content(
                    model=model_name,
                    contents=text,
                    config=genai_types.EmbedContentConfig(output_dimensionality=dim),
                )
            return tuple(float(v) for v in response.embeddings[0].values)
        except Exception as exc:
            last_error = exc
            if attempt >= _MAX_RETRIES - 1 or not _is_retryable(exc):
                raise
            _sleep_backoff(attempt)
    raise last_error or RuntimeError("Gemini embedding failed without an exception")


class GeminiEmbedder:
    """Drop-in replacement for SentenceTransformer.encode() returning np.ndarray.

    Only the subset of the SentenceTransformer API used in this codebase is
    implemented. Specifically: encode(texts, convert_to_numpy=True,
    show_progress_bar=...) returning a 2-D float32 array of shape (N, dim)
    or 1-D for a single-string input.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, dim: int = _DEFAULT_DIM):
        self.model_name = model_name
        self.dim = dim

    def encode(
        self,
        sentences: Union[str, Sequence[str]],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            vec = _embed_one(sentences, self.model_name, self.dim)
            arr = np.asarray(vec, dtype=np.float32)
            return arr if convert_to_numpy else arr.tolist()
        out = np.empty((len(sentences), self.dim), dtype=np.float32)
        for i, text in enumerate(sentences):
            out[i] = _embed_one(text or "", self.model_name, self.dim)
        return out if convert_to_numpy else out.tolist()

    # AutoModel/AutoTokenizer-style alias used by mops_diversity_metric
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)
