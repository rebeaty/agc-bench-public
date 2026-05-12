"""Permutation-style scoring for PermPST structured reasoning outputs."""

import ast
import json
import re
from typing import Any, List, Optional

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat


class PerMPSTScoreMetric(EvaluateInstancesMetric):
    """
    Population-level score evaluator for PerMPST.

    This mirrors the official PerSE scoring path:
    - parse generated JSON reviews
    - extract the numeric `Score`
    - drop malformed outputs from correlation computation
    - report score-level correlations on the valid subset
    """

    @staticmethod
    def _get_valid_entry(content: str) -> Any:
        try:
            return ast.literal_eval("[" + content + "]")
        except Exception:
            key_value_pattern = r'([^:]+):(("[^"]+")|([^,]+)),?'
            valid_entry = {}
            for match in re.findall(key_value_pattern, content):
                key = match[0].strip().lstrip("{")
                value = match[1].strip().rstrip(";").rstrip("}")
                candidate = "{" + key + ":" + value + "}"
                try:
                    parsed_candidate = ast.literal_eval(candidate)
                    if isinstance(parsed_candidate, dict):
                        valid_entry.update(parsed_candidate)
                except Exception:
                    continue
            return valid_entry

    @classmethod
    def _check_json(cls, response: str) -> tuple[bool, Any]:
        response = response.replace("\n", "").replace("\r", "")
        response = response.replace("  ", "")
        code_env = r"```([\s\S]*?)```"
        match_content = re.findall(code_env, response)
        if match_content:
            response = match_content[-1].lstrip("json").lstrip("css").lstrip("javascript").lstrip("vbnet")
            response = response.replace(': "', ':@').replace('","Sco', '@,"Sco').replace('"', "'").replace("@", '"')
        else:
            json_env = r"({([\s\S]*?)}+)"
            match_content = re.findall(json_env, response)
            if match_content:
                response = match_content[-1][0]

        try:
            content = ast.literal_eval(response)
            response = json.dumps(content)
        except Exception:
            repaired_response = response
            json_env = r"{([\s\S]*?)}"
            inner_content = response.strip()[1:-1]
            for candidate in re.findall(json_env, inner_content):
                valid_entries = cls._get_valid_entry(candidate.strip())
                valid_str = json.dumps(valid_entries)
                repaired_response = response.replace("{" + candidate + "}", valid_str)
            response = repaired_response

        try:
            content = json.loads(response)
        except Exception:
            return False, response

        return True, content

    @classmethod
    def _extract_predicted_score(cls, response_text: str) -> Optional[float]:
        response_text = response_text.strip()

        candidate_texts = [response_text]
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, flags=re.IGNORECASE)
        if code_block_match:
            candidate_texts.insert(0, code_block_match.group(1).strip())
        brace_match = re.search(r"\{[\s\S]*\}", response_text)
        if brace_match:
            candidate_texts.insert(0, brace_match.group(0).strip())
        if response_text.startswith('"Review"'):
            wrapped_response = "{" + response_text
            if not response_text.endswith("}"):
                wrapped_response += "}"
            candidate_texts.insert(0, wrapped_response)

        content = None
        status = False
        for candidate in candidate_texts:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    content = parsed
                    status = True
                    break
            except Exception:
                status, content = cls._check_json(candidate)
                if status and isinstance(content, dict):
                    break

        if not status or not isinstance(content, dict):
            return None
        if "Score" not in content:
            return None

        score = content["Score"]
        if isinstance(score, bool):
            return None
        if isinstance(score, (int, float)):
            return float(score)
        if isinstance(score, str):
            try:
                return float(score.strip())
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_reference_score(request_state: RequestState) -> Optional[float]:
        references = request_state.instance.references
        if not references:
            return None
        try:
            return float(references[0].output.text.strip())
        except ValueError:
            return None

    @staticmethod
    def _safe_correlation(values: List[float], references: List[float], fn) -> float:
        if len(values) < 2:
            return 0.0
        if len(set(values)) < 2 or len(set(references)) < 2:
            return 0.0
        result = fn(values, references)[0]
        if result is None or np.isnan(result):
            return 0.0
        return float(result)

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        predicted_scores: List[float] = []
        reference_scores: List[float] = []
        valid_predictions = 0
        exact_score_matches = 0

        for request_state in request_states:
            if request_state.request_mode == "calibration":
                continue
            assert request_state.result is not None
            if len(request_state.result.completions) != 1:
                raise ValueError("PerMPSTScoreMetric expects exactly one completion per instance")

            predicted_score = self._extract_predicted_score(request_state.result.completions[0].text)
            reference_score = self._extract_reference_score(request_state)

            if predicted_score is None or reference_score is None:
                continue

            valid_predictions += 1
            predicted_scores.append(predicted_score)
            reference_scores.append(reference_score)
            if predicted_score == reference_score:
                exact_score_matches += 1

        total_predictions = len(request_states)
        valid_rate = valid_predictions / total_predictions if total_predictions else 0.0
        score_accuracy = exact_score_matches / valid_predictions if valid_predictions else 0.0

        return [
            Stat(MetricName("pearson_correlation")).add(
                self._safe_correlation(predicted_scores, reference_scores, pearsonr)
            ),
            Stat(MetricName("spearman_correlation")).add(
                self._safe_correlation(predicted_scores, reference_scores, spearmanr)
            ),
            Stat(MetricName("kendall_tau_correlation")).add(
                self._safe_correlation(predicted_scores, reference_scores, kendalltau)
            ),
            Stat(MetricName("score_accuracy")).add(score_accuracy),
            Stat(MetricName("valid_score_json_rate")).add(valid_rate),
        ]
