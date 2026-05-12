"""
Creative Process metric: process-analysis evaluator for response sequences.

Paper: "Characterising the Creative Process in Humans and Large Language Models"
       https://arxiv.org/abs/2405.00899
Repo:  https://github.com/surabhisnath/Creative_Process

This metric restores the benchmark's core local semantics more faithfully than
the prior generic LLM-judge setup by:
- parsing the generated bracketed response list
- removing obvious invalid repetitions / empty items
- encoding responses with the paper's `thenlper/gte-large` model
- measuring consecutive semantic similarity
- assigning categories by nearest released human response / category exemplar
- computing category jumps, semantic-similarity jumps, and combined jumps
- reporting jump-profile-style summary statistics

Adaptation notes:
- The paper's categories were built globally over the released corpus. Rather
  than reclustering the full mixed human/LLM universe on every run, this metric
  reuses the released human-cleaned task CSVs and their provided category labels
  as the reference side for nearest-category assignment.
- `jump_SS` uses the paper-aligned threshold noted in the paper / repo notes and
  the existing benchmark review (`SS < 0.8`).
- The paper's participant-clustering analysis over the first 18 responses is not
  reproduced as a HELM metric here; this runnable metric focuses on the released
  sequence-level jump summaries that can be scored directly from each output.
"""

from __future__ import annotations

import csv
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


@dataclass(frozen=True)
class _ReferenceConfig:
    url: str
    response_fields: Sequence[str]
    category_field: str
    expected_count: int


@dataclass
class _ReferenceData:
    responses: List[str]
    categories: List[int]
    embeddings: np.ndarray


class CreativeProcessMetric(Metric):
    _CATEGORY_LOCK = threading.Lock()
    _MODEL_LOCK = threading.Lock()
    _REFERENCE_CACHE: Dict[str, _ReferenceData] = {}

    _REFERENCE_CONFIGS: Dict[str, _ReferenceConfig] = {
        "vf": _ReferenceConfig(
            url="https://raw.githubusercontent.com/surabhisnath/Creative_Process/main/csvs/data_humans_vf_cleaned.csv",
            response_fields=("response_corrected", "response", "original_response_cleaned", "original_response"),
            category_field="clusters_hier",
            expected_count=30,
        ),
        "aut_brick": _ReferenceConfig(
            url="https://raw.githubusercontent.com/surabhisnath/Creative_Process/main/csvs/data_humans_autbrick_cleaned.csv",
            response_fields=(
                "response_English_corrected",
                "response_English_cleaned",
                "response_English",
                "response",
            ),
            category_field="cluster",
            expected_count=30,
        ),
        "aut_paperclip": _ReferenceConfig(
            url="https://raw.githubusercontent.com/surabhisnath/Creative_Process/main/csvs/data_humans_autpaperclip_cleaned.csv",
            response_fields=("response_corrected", "response", "original_response_cleaned", "original_response"),
            category_field="clusters_hier",
            expected_count=20,
        ),
    }

    _LIST_SPLIT_RE = re.compile(r",|\n|;")
    _NUMBERED_PREFIX_RE = re.compile(r"^\s*\d+[\).\:-]\s*")
    _WS_RE = re.compile(r"\s+")

    def __init__(self, model_name: str = "thenlper/gte-large", jump_ss_threshold: float = 0.8):
        super().__init__()
        self.model_name = model_name
        self.jump_ss_threshold = jump_ss_threshold
        self._model: Optional[object] = None

    def _get_model(self):
        if self._model is None:
            with self._MODEL_LOCK:
                if self._model is None:
                    from metrics.embedder_factory import get_embedder
                    self._model = get_embedder(self.model_name)
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        embedder = self._get_model()
        embeddings = embedder.encode(list(texts), convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (embeddings / norms).astype(np.float32)

    @classmethod
    def _normalize_response(cls, text: str) -> str:
        text = text.strip().strip("[](){}")
        text = cls._NUMBERED_PREFIX_RE.sub("", text)
        text = text.strip(" \"'`.,;:!?")
        text = cls._WS_RE.sub(" ", text)
        return text.lower().strip()

    @classmethod
    def _coerce_category(cls, value: str) -> Optional[int]:
        value = (value or "").strip()
        if not value or value.lower() == "nan":
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    @classmethod
    def _is_invalid_row(cls, row: Dict[str, str]) -> bool:
        raw = (row.get("invalid") or "").strip()
        if not raw or raw.lower() == "nan":
            return False
        try:
            return float(raw) != 0.0
        except ValueError:
            return raw.lower() not in {"false", "no"}

    def _load_reference_data(self, task: str, eval_cache_path: str) -> _ReferenceData:
        if task in self._REFERENCE_CACHE:
            return self._REFERENCE_CACHE[task]

        config = self._REFERENCE_CONFIGS[task]
        cache_dir = os.path.join(eval_cache_path, "creative_process_reference_cache")
        ensure_directory_exists(cache_dir)
        csv_path = os.path.join(cache_dir, f"{task}.csv")
        ensure_file_downloaded(source_url=config.url, target_path=csv_path)

        response_category_votes: Dict[str, Counter[int]] = {}
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if self._is_invalid_row(row):
                    continue
                category = self._coerce_category(row.get(config.category_field, ""))
                if category is None:
                    continue

                response = ""
                for field_name in config.response_fields:
                    candidate = self._normalize_response(row.get(field_name, ""))
                    if candidate:
                        response = candidate
                        break
                if not response:
                    continue

                response_category_votes.setdefault(response, Counter())[category] += 1

        responses = sorted(response_category_votes.keys())
        categories = [response_category_votes[response].most_common(1)[0][0] for response in responses]
        embeddings = self._encode(responses)
        embeddings = self._normalize_embeddings(embeddings)

        data = _ReferenceData(responses=responses, categories=categories, embeddings=embeddings)
        with self._CATEGORY_LOCK:
            self._REFERENCE_CACHE.setdefault(task, data)
        return self._REFERENCE_CACHE[task]

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    def _parse_list_items(self, completion: str) -> List[str]:
        text = completion.strip()
        if "[" in text:
            text = text.split("[", 1)[1]
        if "]" in text:
            text = text.split("]", 1)[0]

        raw_items = [self._normalize_response(part) for part in self._LIST_SPLIT_RE.split(text)]
        raw_items = [item for item in raw_items if item]

        deduped_items: List[str] = []
        seen = set()
        for item in raw_items:
            if item in seen:
                continue
            seen.add(item)
            deduped_items.append(item)
        return deduped_items

    def _assign_categories(
        self, items: Sequence[str], reference_data: _ReferenceData
    ) -> tuple[List[int], List[float]]:
        if not items:
            return [], []

        item_embeddings = self._encode(items)
        item_embeddings = self._normalize_embeddings(item_embeddings)
        similarity_matrix = np.matmul(item_embeddings, reference_data.embeddings.T)

        nearest_indices = np.argmax(similarity_matrix, axis=1)
        nearest_scores = [float(similarity_matrix[row_index, ref_index]) for row_index, ref_index in enumerate(nearest_indices)]
        categories = [reference_data.categories[index] for index in nearest_indices]
        return categories, nearest_scores

    def _consecutive_similarities(self, items: Sequence[str]) -> List[float]:
        if len(items) < 2:
            return []
        embeddings = self._encode(items)
        embeddings = self._normalize_embeddings(embeddings)
        return [float(np.dot(embeddings[index], embeddings[index + 1])) for index in range(len(embeddings) - 1)]

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()
        task = str(request_state.instance.extra_data.get("task", "vf"))
        expected_count = int(request_state.instance.extra_data.get("expected_count", self._REFERENCE_CONFIGS[task].expected_count))

        items = self._parse_list_items(completion)
        reference_data = self._load_reference_data(task, eval_cache_path)
        categories, nearest_scores = self._assign_categories(items, reference_data)
        consecutive_sims = self._consecutive_similarities(items)

        jump_ss = [1.0 if similarity < self.jump_ss_threshold else 0.0 for similarity in consecutive_sims]
        jump_cat = [
            1.0 if categories[index] != categories[index + 1] else 0.0
            for index in range(max(len(categories) - 1, 0))
        ]
        jumps = [
            1.0 if jump_cat[index] and jump_ss[index] else 0.0
            for index in range(min(len(jump_cat), len(jump_ss)))
        ]

        item_count = len(items)
        valid_list_rate = 1.0 if item_count >= 2 else 0.0
        jump_total = float(sum(jumps))
        jump_rate = jump_total / len(jumps) if jumps else 0.0
        unique_category_count = float(len(set(categories))) if categories else 0.0
        unique_category_rate = unique_category_count / item_count if item_count else 0.0
        jump_profile_slope = jump_total / max(item_count - 1, 1) if item_count >= 2 else 0.0

        stats = [
            Stat(MetricName("creative_process_valid_list_rate")).add(valid_list_rate),
            Stat(MetricName("creative_process_item_count")).add(float(item_count)),
            Stat(MetricName("creative_process_valid_item_count")).add(float(item_count)),
            Stat(MetricName("creative_process_item_count_ratio")).add(float(item_count / expected_count if expected_count else 0.0)),
            Stat(MetricName("creative_process_mean_reference_similarity")).add(float(mean(nearest_scores) if nearest_scores else 0.0)),
            Stat(MetricName("creative_process_mean_consecutive_similarity")).add(float(mean(consecutive_sims) if consecutive_sims else 0.0)),
            Stat(MetricName("creative_process_jump_cat_rate")).add(float(mean(jump_cat) if jump_cat else 0.0)),
            Stat(MetricName("creative_process_jump_ss_rate")).add(float(mean(jump_ss) if jump_ss else 0.0)),
            Stat(MetricName("creative_process_jump_rate")).add(jump_rate),
            Stat(MetricName("creative_process_total_jumps")).add(jump_total),
            Stat(MetricName("creative_process_unique_category_count")).add(unique_category_count),
            Stat(MetricName("creative_process_unique_category_rate")).add(unique_category_rate),
            Stat(MetricName("creative_process_jump_profile_slope")).add(jump_profile_slope),
        ]
        return stats
