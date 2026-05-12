"""NoveltyBench metric implementing classifier partitioning and utility scoring."""

import bisect
import json
import os
from functools import cache
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


_REWARD_THRESHOLDS = [
    -7.71875,
    -6.28125,
    -6.0,
    -5.71875,
    -5.5,
    -5.0,
    -4.375,
    -3.4375,
    -2.046875,
]


def _classifier_model_name() -> str:
    return os.environ.get("NOVELTYBENCH_CLASSIFIER_MODEL_OVERRIDE", "yimingzhang/deberta-v3-large-generation-similarity")


def _classifier_tokenizer_name() -> str:
    return os.environ.get("NOVELTYBENCH_CLASSIFIER_TOKENIZER_OVERRIDE", "microsoft/deberta-v3-large")


def _classifier_device() -> str:
    override = os.environ.get("NOVELTYBENCH_CLASSIFIER_DEVICE_OVERRIDE", "").strip()
    if override:
        return override
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _reward_model_name() -> str:
    return os.environ.get("NOVELTYBENCH_REWARD_MODEL_OVERRIDE", "Skywork/Skywork-Reward-Gemma-2-27B-v0.2")


def _reward_dtype() -> torch.dtype:
    dtype_name = os.environ.get("NOVELTYBENCH_REWARD_DTYPE", "").strip()
    if dtype_name:
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unsupported NOVELTYBENCH_REWARD_DTYPE={dtype_name}")
        return dtype
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _reward_device_map() -> str:
    return os.environ.get("NOVELTYBENCH_REWARD_DEVICE_MAP", "auto")


def _reward_max_memory():
    raw = os.environ.get("NOVELTYBENCH_REWARD_MAX_MEMORY_JSON", "").strip()
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("NOVELTYBENCH_REWARD_MAX_MEMORY_JSON must decode to a JSON object")
    normalized = {}
    for key, value in parsed.items():
        normalized_key = key
        if isinstance(key, str):
            if key.isdigit():
                normalized_key = int(key)
            elif key.startswith("cuda:") and key[5:].isdigit():
                normalized_key = int(key[5:])
        normalized[normalized_key] = value
    return normalized


def _reward_offload_folder() -> str | None:
    folder = os.environ.get("NOVELTYBENCH_REWARD_OFFLOAD_FOLDER", "").strip()
    if not folder:
        return None
    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


@cache
def _load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(_classifier_tokenizer_name(), use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(_classifier_model_name()).to(_classifier_device())
    model.eval()
    return tokenizer, model


@cache
def _load_reward_model():
    model_name = _reward_model_name()
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": _reward_dtype(),
        "device_map": _reward_device_map(),
        "attn_implementation": "eager",
        "num_labels": 1,
    }
    max_memory = _reward_max_memory()
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    offload_folder = _reward_offload_folder()
    if offload_folder is not None:
        model_kwargs["offload_folder"] = offload_folder

    reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    reward_model.eval()
    return reward_model, tokenizer


def _transform_raw_reward(reward: float) -> int:
    return bisect.bisect_left(_REWARD_THRESHOLDS, reward) + 1


def _maybe_test_equality(response_a: str, response_b: str) -> Optional[bool]:
    unigram_a = response_a.strip().lower().split()
    unigram_b = response_b.strip().lower().split()
    max_len = max(len(unigram_a), len(unigram_b))
    if max_len <= 5:
        common_unigrams = set(unigram_a) & set(unigram_b)
        return len(common_unigrams) * 2 >= max_len
    return None


@torch.inference_mode()
def _classifier_score(response_a: str, response_b: str) -> float:
    tokenizer, model = _load_classifier()
    input_ids = [tokenizer.cls_token_id]
    for response in [response_a, response_b]:
        input_ids.extend(
            tokenizer.encode(
                response,
                truncation=True,
                max_length=128,
                add_special_tokens=False,
            )
        )
        input_ids.append(tokenizer.sep_token_id)
        prompt_len = input_ids.index(tokenizer.sep_token_id) + 1

    token_type_ids = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)
    device = _classifier_device()
    iids = torch.tensor(input_ids, device=device, dtype=torch.int64)
    tids = torch.tensor(token_type_ids, device=device, dtype=torch.int64)
    outputs = model(input_ids=iids.unsqueeze(0), token_type_ids=tids.unsqueeze(0))
    return outputs["logits"].softmax(-1)[0, 1].item()


def _equivalent(response_a: str, response_b: str) -> bool:
    equality = _maybe_test_equality(response_a, response_b)
    if equality is not None:
        return equality
    return _classifier_score(response_a, response_b) > 0.102


def _partition_responses(responses: List[str]) -> List[int]:
    partition = [-1] * len(responses)
    representatives: List[str] = []

    for index, response in enumerate(responses):
        if partition[index] >= 0:
            continue

        class_id = len(representatives)
        representatives.append(response)
        partition[index] = class_id

        for compare_index in range(index + 1, len(responses)):
            if partition[compare_index] == -1 and _equivalent(response, responses[compare_index]):
                partition[compare_index] = class_id

    return partition


@torch.inference_mode()
def _score_partition(prompt: str, generations: List[str], partition: List[int]) -> tuple[List[int], List[int]]:
    reward_model, tokenizer = _load_reward_model()
    conversations = [
        [
            {"content": prompt, "role": "user"},
            {"content": generation, "role": "assistant"},
        ]
        for generation in generations
    ]
    batch = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    ).to(reward_model.device)

    raw_rewards = reward_model(**batch).logits[:, 0].tolist()
    scores = [_transform_raw_reward(reward) for reward in raw_rewards]

    generation_scores: List[int] = []
    partition_scores: List[int] = []
    for score, class_id in zip(scores, partition, strict=False):
        if class_id == len(partition_scores):
            generation_scores.append(score)
            partition_scores.append(score)
        else:
            generation_scores.append(0)

    return generation_scores, partition_scores


class NoveltyBenchMetric(Metric):
    """Compute NoveltyBench distinct_k and utility_k from one prompt's generations."""

    def __init__(self, num_generations: int = 10, patience: float = 0.8):
        super().__init__()
        self.num_generations = num_generations
        self.patience = patience
        self._distinct_metric_name = f"distinct_{num_generations}"
        self._utility_metric_name = f"utility_{num_generations}"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None

        generations = [completion.text.strip() for completion in request_state.result.completions if completion.text.strip()]
        if not generations:
            return [
                Stat(MetricName(self._distinct_metric_name)).add(0.0),
                Stat(MetricName(self._utility_metric_name)).add(0.0),
            ]

        partition = _partition_responses(generations)
        generation_scores, partition_scores = _score_partition(
            request_state.instance.input.text,
            generations,
            partition,
        )
        utility = float(
            np.average(
                generation_scores,
                weights=self.patience ** np.arange(len(generations)),
            )
        )
        distinct = float(len(partition_scores))

        return [
            Stat(MetricName(self._distinct_metric_name)).add(distinct),
            Stat(MetricName(self._utility_metric_name)).add(utility),
        ]
