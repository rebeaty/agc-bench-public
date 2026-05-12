"""Benchmark-specific metric for the local MATDESIGN proxy."""

import ast
import json
import re
import threading
from functools import lru_cache
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _get_embedder():
    from metrics.embedder_factory import get_embedder as _factory_get_embedder
    return _factory_get_embedder(_MODEL_NAME)


def _embed_texts(texts: List[str]) -> torch.Tensor:
    """Returns L2-normalized embeddings as a torch.Tensor of shape (N, dim).

    Routes through the active embedder backend (Gemini by default; see
    metrics/embedder_factory.py). The torch.Tensor return type is preserved
    so downstream callers using torch ops keep working.
    """
    import numpy as np
    with _EMBED_LOCK:
        embedder = _get_embedder()
        embeddings = embedder.encode(list(texts), convert_to_numpy=True)
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return torch.from_numpy(arr / norms)


def _extract_text(value) -> str:
    if isinstance(value, list):
        parts = [_extract_text(item) for item in value]
        return " ".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = [_extract_text(item) for item in value.values()]
        return " ".join(part for part in parts if part)
    return str(value).strip()


def _normalize_suggestion(value) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {"materials": "", "methods": "", "reasoning": _extract_text(value)}

    fields = {str(key).strip().lower(): value[key] for key in value}
    return {
        "materials": _extract_text(fields.get("materials", "")),
        "methods": _extract_text(
            fields.get("methods_to_develop_the_materials_suggested", "")
            or fields.get("methods", "")
            or fields.get("method", "")
        ),
        "reasoning": _extract_text(fields.get("reasoning", "")),
    }


def _sorted_suggestion_items(payload: dict) -> List[Dict[str, str]]:
    items = []
    for key, value in payload.items():
        key_text = str(key).strip()
        if not key_text.lower().startswith("suggestion"):
            continue
        match = re.search(r"(\d+)", key_text)
        order = int(match.group(1)) if match else 999
        items.append((order, _normalize_suggestion(value)))
    items.sort(key=lambda item: item[0])
    return [item[1] for item in items]


def _try_json_parse(text: str) -> Optional[dict]:
    cleaned = text.strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _extract_field(block: str, field_names: List[str]) -> str:
    for field_name in field_names:
        pattern = re.compile(
            rf"{re.escape(field_name)}\s*:\s*(.*?)(?=\n\s*[A-Za-z_][A-Za-z0-9_ ]*\s*:|\Z)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(block)
        if match:
            return match.group(1).strip(" \n\r\t,{}")
    return ""


def _fallback_parse(text: str) -> List[Dict[str, str]]:
    pattern = re.compile(
        r"(Suggestion[_ ]?\d+)\s*[:\-]?\s*(.*?)(?=(?:\n\s*Suggestion[_ ]?\d+\s*[:\-]?)|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    suggestions = []
    for _, block in pattern.findall(text):
        suggestions.append(
            {
                "materials": _extract_field(block, ["Materials"]),
                "methods": _extract_field(
                    block,
                    [
                        "Methods_to_develop_the_materials_suggested",
                        "Methods to develop the materials suggested",
                        "Methods",
                    ],
                ),
                "reasoning": _extract_field(block, ["Reasoning"]),
            }
        )
    return suggestions


def _parse_suggestions(text: str) -> List[Dict[str, str]]:
    parsed = _try_json_parse(text)
    if parsed is not None:
        suggestions = _sorted_suggestion_items(parsed)
        if suggestions:
            return suggestions
        normalized = _normalize_suggestion(parsed)
        if any(normalized.values()):
            return [normalized]
    return _fallback_parse(text)


def _combine_suggestion_text(suggestion: Dict[str, str]) -> str:
    return "\n".join(
        [
            f"Materials: {suggestion['materials']}",
            f"Methods: {suggestion['methods']}",
            f"Reasoning: {suggestion['reasoning']}",
        ]
    ).strip()


class MatDesignMetric(Metric):
    """Structured-output and semantic-alignment proxy for MATDESIGN."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()
        suggestions = _parse_suggestions(completion)
        parsed_count = len(suggestions)
        complete_suggestions = [
            suggestion
            for suggestion in suggestions
            if suggestion["materials"] and suggestion["methods"] and suggestion["reasoning"]
        ]
        complete_count = len(complete_suggestions)

        stats = [
            Stat(MetricName("matdesign_parse_rate")).add(1.0 if parsed_count > 0 else 0.0),
            Stat(MetricName("matdesign_suggestion_coverage")).add(min(parsed_count, 20) / 20.0),
            Stat(MetricName("matdesign_complete_suggestion_rate")).add(min(complete_count, 20) / 20.0),
        ]

        if not complete_suggestions:
            stats.extend(
                [
                    Stat(MetricName("matdesign_goal_alignment")).add(0.0),
                    Stat(MetricName("matdesign_constraint_coverage")).add(0.0),
                    Stat(MetricName("matdesign_reference_closeness")).add(0.0),
                ]
            )
            return stats

        extra_data = request_state.instance.extra_data or {}
        goal_statement = str(extra_data.get("goal_statement", "")).strip()
        constraints = [
            str(item).strip()
            for item in extra_data.get("constraints_list", [])
            if str(item).strip()
        ]
        reference_text = str(extra_data.get("reference_text", "")).strip()

        suggestion_texts = [_combine_suggestion_text(suggestion) for suggestion in complete_suggestions]
        embed_inputs = list(suggestion_texts)
        goal_index = None
        constraint_start = None
        reference_index = None

        if goal_statement:
            goal_index = len(embed_inputs)
            embed_inputs.append(goal_statement)
        if constraints:
            constraint_start = len(embed_inputs)
            embed_inputs.extend(constraints)
        if reference_text:
            reference_index = len(embed_inputs)
            embed_inputs.append(reference_text)

        embeddings = _embed_texts(embed_inputs)
        suggestion_embeddings = embeddings[: len(suggestion_texts)]

        goal_alignment = 0.0
        if goal_index is not None:
            goal_embedding = embeddings[goal_index]
            goal_alignment = torch.matmul(suggestion_embeddings, goal_embedding).mean().item()

        constraint_coverage = 0.0
        if constraints:
            constraint_embeddings = embeddings[constraint_start : constraint_start + len(constraints)]
            coverage_scores = torch.matmul(suggestion_embeddings, constraint_embeddings.T)
            constraint_coverage = coverage_scores.max(dim=0).values.mean().item()

        reference_closeness = 0.0
        if reference_index is not None:
            reference_embedding = embeddings[reference_index]
            reference_closeness = torch.matmul(suggestion_embeddings, reference_embedding).max().item()

        stats.extend(
            [
                Stat(MetricName("matdesign_goal_alignment")).add(goal_alignment),
                Stat(MetricName("matdesign_constraint_coverage")).add(constraint_coverage),
                Stat(MetricName("matdesign_reference_closeness")).add(reference_closeness),
            ]
        )
        return stats
