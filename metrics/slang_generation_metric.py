"""Benchmark-specific metric for the freeform slang-generation proxy."""

import json
import re
import threading
from functools import lru_cache
from typing import List, Optional

import torch
from nltk.corpus import wordnet as wn
from transformers import AutoModel, AutoTokenizer

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_WORDNET_LOCK = threading.Lock()
_EMBED_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _get_embedder():
    from metrics.embedder_factory import get_embedder as _factory_get_embedder
    return _factory_get_embedder(_MODEL_NAME)


def _embed_texts(texts: List[str]) -> torch.Tensor:
    """Returns mean-pooled embeddings as a torch.Tensor of shape (N, dim).

    Routes through the active embedder backend (Gemini by default; see
    metrics/embedder_factory.py).
    """
    import numpy as np
    with _EMBED_LOCK:
        embedder = _get_embedder()
        embeddings = embedder.encode(list(texts), convert_to_numpy=True)
    return torch.from_numpy(np.asarray(embeddings, dtype=np.float32))


def _try_json_parse(text: str) -> Optional[dict]:
    cleaned = text.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


def _extract_field(parsed: dict, key: str) -> str:
    value = parsed.get(key, "")
    if isinstance(value, list):
        return " | ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def _fallback_parse(text: str) -> dict:
    fields = {"word": "", "definition": "", "usage_context": ""}
    patterns = {
        "word": r"(?:^|\n)\s*(?:word|slang word)\s*[:\-]\s*(.+)",
        "definition": r"(?:^|\n)\s*definition\s*[:\-]\s*(.+)",
        "usage_context": r"(?:^|\n)\s*(?:usage(?:_context)?|usage example)\s*[:\-]\s*(.+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            fields[key] = match.group(1).strip()
    return fields


def _parse_generation(text: str) -> dict:
    parsed = _try_json_parse(text)
    if parsed is None:
        return _fallback_parse(text)
    return {
        "word": _extract_field(parsed, "word"),
        "definition": _extract_field(parsed, "definition"),
        "usage_context": _extract_field(parsed, "usage_context"),
    }


def _normalize_lookup_forms(word: str) -> List[str]:
    forms = []
    cleaned = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9-]+$", "", word.strip())
    if cleaned:
        forms.append(cleaned)
        lower = cleaned.lower()
        if lower not in forms:
            forms.append(lower)
        morphy = wn.morphy(lower)
        if morphy and morphy not in forms:
            forms.append(morphy)
    return forms


def _component_forms(word: str, definition: str) -> List[str]:
    parts = []
    for piece in re.split(r"[^A-Za-z]+", word.lower()):
        piece = piece.strip()
        if len(piece) >= 3:
            parts.append(piece)
    if word.isupper():
        for phrase in re.findall(r"'([^']+)'|\"([^\"]+)\"", definition):
            text = next((p for p in phrase if p), "")
            for piece in re.split(r"[^A-Za-z]+", text.lower()):
                piece = piece.strip()
                if len(piece) >= 3:
                    parts.append(piece)
    deduped = []
    seen = set()
    for part in parts:
        if part not in seen:
            seen.add(part)
            deduped.append(part)
    return deduped


@lru_cache(maxsize=4096)
def _get_standard_glosses_cached(word: str) -> List[str]:
    glosses = []
    with _WORDNET_LOCK:
        synsets = list(wn.synsets(word))
    for synset in synsets:
        gloss = synset.definition().strip()
        if gloss:
            glosses.append(gloss)
    deduped = []
    seen = set()
    for gloss in glosses:
        if gloss not in seen:
            seen.add(gloss)
            deduped.append(gloss)
    return deduped


def _get_standard_glosses(word: str, definition: str) -> List[str]:
    glosses: List[str] = []
    for form in _normalize_lookup_forms(word):
        glosses.extend(_get_standard_glosses_cached(form))
    if glosses:
        return list(dict.fromkeys(glosses))

    fallback_glosses: List[str] = []
    for component in _component_forms(word, definition):
        fallback_glosses.extend(_get_standard_glosses_cached(component))
    return list(dict.fromkeys(fallback_glosses))


class SlangGenerationMetric(Metric):
    """Parser-aware semantic novelty metric for the local freeform proxy."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()
        parsed = _parse_generation(completion)
        has_all_fields = all(parsed[key] for key in ("word", "definition", "usage_context"))

        stats = [Stat(MetricName("slang_parse_rate")).add(1.0 if has_all_fields else 0.0)]
        if not has_all_fields:
            stats.append(Stat(MetricName("slang_novelty_coverage")).add(0.0))
            return stats

        glosses = _get_standard_glosses(parsed["word"], parsed["definition"])
        has_glosses = 1.0 if glosses else 0.0
        stats.append(Stat(MetricName("slang_novelty_coverage")).add(has_glosses))
        if not glosses:
            return stats

        embeddings = _embed_texts([parsed["definition"], *glosses])
        definition_embedding = embeddings[0]
        gloss_embeddings = embeddings[1:]
        novelty = torch.norm(gloss_embeddings - definition_embedding, p=2, dim=1).mean().item()
        stats.append(Stat(MetricName("slang_semantic_novelty")).add(novelty))
        return stats
