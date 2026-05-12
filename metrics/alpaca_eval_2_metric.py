"""Benchmark-specific metrics for AlpacaEval 2.0."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import sklearn
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat

_GLM_INFO = {
    "regularize_to_baseline_lambda": 0.2,
    "n_splits": 5,
}


def _safe_sem(series: Sequence[float]) -> float:
    value = pd.Series(series).sem()
    if pd.isna(value):
        return 0.0
    return float(value)


def _adjusted_preferences(preferences: Sequence[float]) -> pd.Series:
    return pd.Series(preferences, dtype=float).replace({0.0: 1.5})


def _describe_head2head(preferences: Sequence[float]) -> Dict[str, float]:
    adjusted = _adjusted_preferences(preferences)
    adjusted = adjusted[adjusted.notna()]
    if adjusted.empty:
        return {
            "win_rate": 0.0,
            "standard_error": 0.0,
            "n_wins": 0.0,
            "n_wins_base": 0.0,
            "n_draws": 0.0,
            "n_total": 0.0,
            "discrete_win_rate": 0.0,
        }

    win_values = adjusted - 1.0
    discrete = adjusted.apply(lambda value: 2.0 if value > 1.5 else (1.0 if value < 1.5 else 1.5)) - 1.0

    return {
        "win_rate": float(win_values.mean() * 100.0),
        "standard_error": float(_safe_sem(win_values) * 100.0),
        "n_wins": float((win_values > 0.5).sum()),
        "n_wins_base": float((win_values < 0.5).sum()),
        "n_draws": float((adjusted == 1.5).sum()),
        "n_total": float(len(adjusted)),
        "discrete_win_rate": float(discrete.mean() * 100.0),
    }


def _load_instruction_difficulty() -> pd.Series:
    path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="instruction_difficulty.csv",
        repo_type="dataset",
    )
    return pd.read_csv(path, index_col=0).squeeze("columns")


def _load_gamed_annotations() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="df_gamed.csv",
        repo_type="dataset",
    )
    return pd.read_csv(path).drop(columns=["model"])


def _build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "np.tanh(std_delta_len)": np.tanh(df["std_delta_len"].astype(float)),
            "instruction_difficulty": df["instruction_difficulty"].astype(float),
            "not_gamed_baseline.astype(float)": df["not_gamed_baseline"].astype(float),
        }
    )


def _logloss(y_true, y_pred, sample_weight=None):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    all_logloss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    if sample_weight is not None:
        all_logloss = all_logloss * sample_weight
    return -np.mean(all_logloss)


def _logloss_continuous(y_true, y_pred, true_prob, true_sample_weight=None):
    y_true = np.where(y_true == 1, true_prob, 1 - true_prob)
    return _logloss(y_true, y_pred, sample_weight=true_sample_weight)


def _fit_logistic_regression_cv(
    data_x: pd.DataFrame,
    labels: pd.Series,
    sample_weight: Optional[np.ndarray],
    n_splits: int,
):
    sklearn.set_config(enable_metadata_routing=True)
    kwargs = dict(
        random_state=123,
        dual=False,
        penalty="l1",
        solver="liblinear",
        n_jobs=None,
        fit_intercept=False,
    )

    base = data_x.reset_index(drop=True).copy()
    base["preference"] = labels.reset_index(drop=True)
    base = base.reset_index(drop=False, names=["group"])

    data_1 = base.copy()
    data_1["y"] = 1
    data_0 = base.copy()
    data_0["preference"] = 1 - data_0["preference"]
    data_0["y"] = 0
    data_dup = pd.concat([data_1, data_0], axis=0).reset_index(drop=True)
    true_prob = data_dup["preference"].to_numpy()

    if sample_weight is None:
        true_sample_weight = None
        fit_sample_weight = true_prob
    else:
        true_sample_weight = np.concatenate([sample_weight, sample_weight], axis=0)
        fit_sample_weight = true_prob * true_sample_weight

    x = data_dup.drop(columns=["group", "preference", "y"])
    y = data_dup["y"]
    n_groups = data_dup["group"].nunique()

    if n_groups >= 2 and n_splits >= 2:
        effective_splits = min(n_splits, n_groups)
        scorer = make_scorer(
            _logloss_continuous,
            response_method="predict_proba",
            greater_is_better=False,
        ).set_score_request(true_sample_weight=True, true_prob=True)
        model = LogisticRegressionCV(cv=GroupKFold(n_splits=effective_splits), scoring=scorer, **kwargs)
        model.set_fit_request(sample_weight=True)
        model.fit(
            x,
            y,
            sample_weight=fit_sample_weight,
            groups=data_dup["group"],
            true_sample_weight=true_sample_weight,
            true_prob=true_prob,
        )
        return model

    model = LogisticRegression(C=100, **kwargs)
    model.fit(x, y, sample_weight=fit_sample_weight)
    return model


def _get_featurized_data(df_annotations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Optional[np.ndarray]]:
    df_gamed = _load_gamed_annotations()
    instruction_difficulty = _load_instruction_difficulty()

    df = df_annotations.reset_index(drop=True)
    len_1 = df["output_1"].str.len()
    len_2 = df["output_2"].str.len()
    std_delta_len = len_1 - len_2
    std_value = std_delta_len.std()
    if not std_value or pd.isna(std_value):
        std_value = 1.0

    glm_df = df[["preference", "index"]].copy()
    glm_df["std_delta_len"] = std_delta_len / std_value
    glm_df["preference"] = glm_df["preference"].astype(float).replace({0.0: 1.5}) - 1.0
    glm_df["instruction_difficulty"] = glm_df["index"].map(instruction_difficulty)
    glm_df["not_gamed_baseline"] = True

    df_test = glm_df[["instruction_difficulty", "not_gamed_baseline"]].copy()
    df_test["std_delta_len"] = 0.0

    df_gamed_and_model = pd.concat([df_gamed, glm_df], axis=0, ignore_index=True)
    train_x = _build_feature_matrix(df_gamed_and_model)
    test_x = _build_feature_matrix(df_test)

    sample_weight = (df_gamed_and_model["not_gamed_baseline"]).astype(float) + (
        _GLM_INFO["regularize_to_baseline_lambda"] * (~df_gamed_and_model["not_gamed_baseline"])
    ).astype(float) / 2.0

    return train_x, test_x, sample_weight.to_numpy(dtype=float), df_gamed_and_model["preference"]


def _compute_length_controlled_metrics(df: pd.DataFrame) -> Dict[str, float]:
    metrics = _describe_head2head(df["preference"])
    try:
        train_x, test_x, sample_weight, labels = _get_featurized_data(df)
        valid = labels.notna()
        train_x = train_x[valid].reset_index(drop=True)
        labels = labels[valid].reset_index(drop=True)
        sample_weight = sample_weight[valid.to_numpy()]
        model = _fit_logistic_regression_cv(
            train_x,
            labels,
            sample_weight=sample_weight,
            n_splits=_GLM_INFO["n_splits"],
        )
        predicted_preferences = model.predict_proba(test_x)[:, 1]
        metrics["length_controlled_winrate"] = float(predicted_preferences.mean() * 100.0)
        metrics["length_controlled_win_rate"] = metrics["length_controlled_winrate"]
        metrics["lc_standard_error"] = float(_safe_sem(predicted_preferences) * 100.0)
        metrics["length_controlled_fallback"] = 0.0
    except Exception:
        metrics["length_controlled_winrate"] = metrics["win_rate"]
        metrics["length_controlled_win_rate"] = metrics["win_rate"]
        metrics["lc_standard_error"] = metrics["standard_error"]
        metrics["length_controlled_fallback"] = 1.0
    return metrics


def _get_annotation(state: RequestState) -> Dict[str, Any]:
    annotations = state.annotations or {}
    annotation = annotations.get("weighted_alpaca_eval_gpt4_turbo", {}) or {}
    if isinstance(annotation, dict) and "preference" in annotation:
        return annotation
    return {}


class AlpacaEval2Metric(EvaluateInstancesMetric):
    """Expose AlpacaEval 2.0 raw and length-controlled win rates."""

    def __init__(self, **_: Any):
        super().__init__()

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        rows: List[Dict[str, Any]] = []
        parse_failures = 0

        for state in request_states:
            if state.request_mode == "calibration" or state.result is None:
                continue

            annotation = _get_annotation(state)
            preference = annotation.get("preference")
            if pd.isna(preference):
                parse_failures += 1
                continue

            extra_data = state.instance.extra_data or {}
            baseline_output = state.instance.references[0].output.text if state.instance.references else ""
            candidate_output = state.result.completions[0].text.strip() if state.result.completions else ""
            rows.append(
                {
                    "instruction": state.instance.input.text,
                    "output_1": baseline_output,
                    "output_2": candidate_output,
                    "generator_1": extra_data.get("baseline_generator", annotation.get("baseline_generator", "gpt-4-turbo-2024-04-09")),
                    "generator_2": state.request.model_deployment or state.request.model,
                    "annotator": "weighted_alpaca_eval_gpt4_turbo",
                    "index": int(annotation.get("alpaca_eval_index", extra_data.get("alpaca_eval_index", -1))),
                    "preference": float(preference),
                }
            )

        if rows:
            metrics = _compute_length_controlled_metrics(pd.DataFrame(rows))
        else:
            metrics = _describe_head2head([])
            metrics["length_controlled_winrate"] = 0.0
            metrics["length_controlled_win_rate"] = 0.0
            metrics["lc_standard_error"] = 0.0
            metrics["length_controlled_fallback"] = 0.0

        total = int(metrics.get("n_total", 0.0))
        parse_rate = 0.0 if total + parse_failures == 0 else total / float(total + parse_failures)

        return [
            Stat(MetricName("win_rate")).add(metrics.get("win_rate", 0.0)),
            Stat(MetricName("length_controlled_winrate")).add(metrics.get("length_controlled_winrate", 0.0)),
            Stat(MetricName("length_controlled_win_rate")).add(metrics.get("length_controlled_win_rate", 0.0)),
            Stat(MetricName("discrete_win_rate")).add(metrics.get("discrete_win_rate", 0.0)),
            Stat(MetricName("standard_error")).add(metrics.get("standard_error", 0.0)),
            Stat(MetricName("lc_standard_error")).add(metrics.get("lc_standard_error", 0.0)),
            Stat(MetricName("n_total")).add(metrics.get("n_total", 0.0)),
            Stat(MetricName("n_wins")).add(metrics.get("n_wins", 0.0)),
            Stat(MetricName("n_wins_base")).add(metrics.get("n_wins_base", 0.0)),
            Stat(MetricName("n_draws")).add(metrics.get("n_draws", 0.0)),
            Stat(MetricName("preference_parse_rate")).add(parse_rate),
            Stat(MetricName("length_controlled_fallback")).add(metrics.get("length_controlled_fallback", 0.0)),
        ]
