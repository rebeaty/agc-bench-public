"""Benchmark-specific held-out inference annotator for HypoBench real tasks."""

from __future__ import annotations

import json
import os
import re
from string import Template
from typing import Any, Dict, List, Sequence, Tuple

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request


_HYPOTHESIS_RE = re.compile(r"\d+\.\s*(.+?)(?=\n\s*\d+\.\s*|\Z)", flags=re.DOTALL)
# Hypobench's "inference call" is structurally a fast classifier (predict a
# label given a hypothesis + a row of test data), NOT a thoughtful rater.
# It is paper-canonical for this bench to default to a small, fast model
# (gemini-3-flash-preview) rather than AGC-Judge, even when AGC_JUDGE_OVERRIDE
# is set globally for the 23 *true* LLM-judge benches. Users who want a
# specific inference model can set HYPOBENCH_INFERENCE_MODEL_OVERRIDE; the
# global AGC_JUDGE_OVERRIDE deliberately does not propagate here because a
# 30B-MoE model is the wrong tool for ~4000 per-row predictions per model.
_DEFAULT_INFERENCE_MODEL = "google/gemini-3-flash-preview"
_BACKUP_MODEL = _DEFAULT_INFERENCE_MODEL
_MODEL_OVERRIDE = os.environ.get("HYPOBENCH_INFERENCE_MODEL_OVERRIDE", "").strip() or None


def _columnar_to_rows(payload: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(payload.keys())
    if not keys:
        return []
    row_count = len(payload[keys[0]])
    return [{key: payload[key][index] for key in keys} for index in range(row_count)]


def _safe_rows(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return _columnar_to_rows(json.load(handle))


def _extract_hypotheses(text: str) -> List[str]:
    hypotheses = [item.strip() for item in _HYPOTHESIS_RE.findall(text or "") if item.strip()]
    # Deduplicate while preserving order.
    unique: List[str] = []
    seen = set()
    for hypothesis in hypotheses:
        if hypothesis in seen:
            continue
        seen.add(hypothesis)
        unique.append(hypothesis)
    return unique


def _macro_f1(predictions: Sequence[str], labels: Sequence[str], valid_labels: Sequence[str]) -> float:
    if not labels:
        return 0.0

    scores: List[float] = []
    for label in valid_labels:
        tp = sum(1 for pred, gold in zip(predictions, labels) if pred == label and gold == label)
        fp = sum(1 for pred, gold in zip(predictions, labels) if pred == label and gold != label)
        fn = sum(1 for pred, gold in zip(predictions, labels) if pred != label and gold == label)
        if tp == 0 and fp == 0 and fn == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _normalize_label(task: str, label: str) -> str:
    text = " ".join((label or "").strip().split())
    text = text.replace("Human", "HUMAN")
    lowered = text.lower()
    if task == "headline_binary":
        if "headline 1" in lowered:
            return "Headline 1 has more clicks than Headline 2."
        if "headline 2" in lowered:
            return "Headline 2 has more clicks than Headline 1."
    if task in {"gptgc_detect", "llamagc_detect"}:
        if lowered == "human":
            return "HUMAN"
        if lowered == "ai":
            return "AI"
    return text


def _extract_prediction(task: str, text: str) -> str:
    lowered = (text or "").lower()

    if task == "deceptive_reviews":
        match = re.findall(r"final answer:\s+(truthful|deceptive|other)", lowered)
        return match[-1] if match else "other"

    if task == "headline_binary":
        match = re.findall(r"answer:\s+(headline 1|headline 2|other)", lowered)
        if not match:
            final = re.findall(r"final answer:\s+(headline 1|headline 2|other)", lowered)
            match = final
        if not match:
            return "other"
        answer = match[-1]
        if answer == "headline 1":
            return "Headline 1 has more clicks than Headline 2."
        if answer == "headline 2":
            return "Headline 2 has more clicks than Headline 1."
        return "other"

    if task in {"gptgc_detect", "llamagc_detect"}:
        match = re.findall(r"final answer:\s+(ai|human)", lowered)
        if not match:
            return "other"
        return "AI" if match[-1] == "ai" else "HUMAN"

    if task == "dreaddit":
        patterns = [r"final answer:\s+(has stress|no stress)", r"answer:\s+(has stress|no stress)"]
        for pattern in patterns:
            match = re.findall(pattern, lowered)
            if match:
                return match[-1]
        return "other"

    if task == "persuasive_pairs":
        patterns = [
            r"final answer:\s+(?:the )?(first|second) argument",
            r"answer:\s+(?:the )?(first|second) argument",
        ]
        for pattern in patterns:
            match = re.findall(pattern, lowered)
            if match:
                return match[-1]
        return "other"

    if task == "retweet":
        patterns = [
            r"final answer:\s+the (first|second) tweet",
            r"answer:\s+the (first|second) tweet",
        ]
        for pattern in patterns:
            match = re.findall(pattern, lowered)
            if match:
                return match[-1]
        return "other"

    return "other"


def _apply_template(template: Dict[str, str], values: Dict[str, Any]) -> str:
    system_prompt = Template(str(template.get("system", ""))).safe_substitute(values).strip()
    user_prompt = Template(str(template.get("user", ""))).safe_substitute(values).strip()
    if system_prompt and user_prompt:
        return f"{system_prompt}\n\n{user_prompt}"
    return system_prompt or user_prompt


class HypoBenchInferenceAnnotator(Annotator):
    """Run the benchmark's held-out inference pipeline on generated hypotheses."""

    def __init__(self, auto_client: AutoClient, inference_temperature: float = 0.0, inference_max_new_tokens: int = 512):
        self._auto_client = auto_client
        self.inference_temperature = inference_temperature
        self.inference_max_new_tokens = inference_max_new_tokens
        self.name = "hypobench_inference"

    def _call_model(self, model_name: str, prompt: str) -> str:
        request = Request(
            model=model_name,
            model_deployment=model_name,
            prompt=prompt,
            temperature=self.inference_temperature,
            max_tokens=self.inference_max_new_tokens,
            num_completions=1,
        )
        result = self._auto_client.make_request(request)
        if not result.success:
            raise RuntimeError(f"HypoBench inference call failed for model {model_name}")
        return result.completions[0].text if result.completions else ""

    def _score_split(
        self,
        *,
        task: str,
        rows: Sequence[Dict[str, Any]],
        hypotheses: Sequence[str],
        template: Dict[str, str],
        label_values: Sequence[str],
        model_name: str,
    ) -> Tuple[float, float, float]:
        if not rows:
            return 0.0, 0.0, 0.0

        predictions: List[str] = []
        labels: List[str] = []
        parsed = 0

        numbered_hypotheses = "\n".join(f"{index + 1}. {hypothesis}" for index, hypothesis in enumerate(hypotheses))
        for row in rows:
            prompt_values = dict(row)
            prompt_values["hypotheses"] = numbered_hypotheses
            prompt_values["hypothesis"] = hypotheses[0] if hypotheses else ""
            prompt = _apply_template(template, prompt_values)

            try:
                raw_prediction = self._call_model(model_name, prompt)
            except Exception:
                try:
                    raw_prediction = self._call_model(_BACKUP_MODEL, prompt)
                except Exception:
                    raw_prediction = ""

            prediction = _extract_prediction(task, raw_prediction)
            gold = _normalize_label(task, str(row.get("label", "")))
            if prediction != "other":
                parsed += 1
            predictions.append(_normalize_label(task, prediction))
            labels.append(gold)

        accuracy = sum(1 for pred, gold in zip(predictions, labels) if pred == gold) / len(labels)
        f1 = _macro_f1(predictions, labels, [_normalize_label(task, label) for label in label_values])
        parsed_rate = parsed / len(labels)
        return accuracy, f1, parsed_rate

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result is not None
        extra_data = request_state.instance.extra_data or {}
        task = str(extra_data.get("task") or "")
        hypotheses = _extract_hypotheses(request_state.result.completions[0].text if request_state.result.completions else "")
        label_values = list(extra_data.get("label_values") or [])
        model_name = _MODEL_OVERRIDE or _DEFAULT_INFERENCE_MODEL

        ind_limit = int(os.environ.get("HYPOBENCH_MAX_IND_EXAMPLES", "0") or "0")
        ood_limit = int(os.environ.get("HYPOBENCH_MAX_OOD_EXAMPLES", "0") or "0")

        ind_rows = _safe_rows(str(extra_data.get("test_data_path", "")))
        ood_rows = _safe_rows(str(extra_data.get("ood_data_path", "")))
        if ind_limit > 0:
            ind_rows = ind_rows[:ind_limit]
        if ood_limit > 0:
            ood_rows = ood_rows[:ood_limit]

        multi_hyp_template = extra_data.get("multiple_hypotheses_inference") or {}
        single_hyp_template = extra_data.get("inference") or {}
        template = multi_hyp_template if multi_hyp_template else single_hyp_template

        if not hypotheses or not template:
            return {
                "hypobench_ind_accuracy": 0.0,
                "hypobench_ind_f1": 0.0,
                "hypobench_ind_parsed_label_rate": 0.0,
                "hypobench_ood_accuracy": 0.0,
                "hypobench_ood_f1": 0.0,
                "hypobench_ood_parsed_label_rate": 0.0,
                "hypobench_hypothesis_count": float(len(hypotheses)),
                "hypobench_parsed_hypothesis_rate": 0.0,
            }

        ind_accuracy, ind_f1, ind_parsed_rate = self._score_split(
            task=task,
            rows=ind_rows,
            hypotheses=hypotheses,
            template=template,
            label_values=label_values,
            model_name=model_name,
        )
        ood_accuracy, ood_f1, ood_parsed_rate = self._score_split(
            task=task,
            rows=ood_rows,
            hypotheses=hypotheses,
            template=template,
            label_values=label_values,
            model_name=model_name,
        )

        return {
            "hypobench_ind_accuracy": ind_accuracy,
            "hypobench_ind_f1": ind_f1,
            "hypobench_ind_parsed_label_rate": ind_parsed_rate,
            "hypobench_ood_accuracy": ood_accuracy,
            "hypobench_ood_f1": ood_f1,
            "hypobench_ood_parsed_label_rate": ood_parsed_rate,
            "hypobench_hypothesis_count": float(len(hypotheses)),
            "hypobench_parsed_hypothesis_rate": 1.0,
        }
