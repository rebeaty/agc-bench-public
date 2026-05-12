"""HELM Run Specs for thenextchapter."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from llm_judge._judge_override import resolve_judge

_JUDGE_MODEL = resolve_judge("openai/gpt-4", bench_var="THENEXTCHAPTER_JUDGE_MODEL_OVERRIDE")

_DIMENSIONS = {
    "thenextchapter_fluency": "Grammatical correctness, readability, and natural language flow.",
    "thenextchapter_coherence": "Internal consistency and narrative flow across the continuation.",
    "thenextchapter_relatedness": "How well the continuation follows from and stays connected to the given condition.",
    "thenextchapter_logicality": "Plausibility of events and cause-and-effect relationships.",
    "thenextchapter_interestingness": "Creativity, engagement, and entertainment value.",
}


def _build_rubric(label: str, definition: str) -> str:
    return f"""\
Evaluate the {label.upper()} of the generated story continuation for The Next Chapter benchmark.
Consider only this single dimension.

{label.title()}: {definition}

Score 1: Very poor
Score 2: Poor
Score 3: Adequate
Score 4: Good
Score 5: Excellent
"""


def _build_thenextchapter_spec(name: str, subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.thenextchapter_scenario.TheNextChapterScenario",
        args={"subset": subset},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",  # NOTE: scenario passes the raw condition directly.
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=512,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
            args={"names": ["bleu_4", "rouge_1", "rouge_2", "rouge_l"]},
        ),
        MetricSpec(class_name="metrics.meteor_metric.MeteorMetric", args={}),
        MetricSpec(
            class_name="metrics.bert_score_metric.BertScoreMetric",
            args={"model_type": "bert-base-uncased"},
        ),
        *[
            MetricSpec(
                class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
                args={"metric_name": metric_name},
            )
            for metric_name in _DIMENSIONS
        ],
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.thenextchapter_annotator.TheNextChapterJudgeAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": metric_name,
                "rubric": _build_rubric(metric_name.replace("thenextchapter_", ""), definition),
            },
        )
        for metric_name, definition in _DIMENSIONS.items()
    ]

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "thenextchapter"],
        annotators=annotators,
    )


@run_spec_function("thenextchapter")
def get_thenextchapter_spec() -> RunSpec:
    return _build_thenextchapter_spec(name="thenextchapter", subset="roc")


@run_spec_function("thenextchapter_wp")
def get_thenextchapter_wp_spec() -> RunSpec:
    return _build_thenextchapter_spec(name="thenextchapter_wp", subset="wp")


@run_spec_function("thenextchapter_cnn")
def get_thenextchapter_cnn_spec() -> RunSpec:
    return _build_thenextchapter_spec(name="thenextchapter_cnn", subset="cnn")
