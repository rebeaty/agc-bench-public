"""HELM Run Specs for arastories."""

import os

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from llm_judge._judge_override import resolve_judge


# ── Rubrics ──────────────────────────────────────────────────────────────────

_JUDGE_MODEL_NAME = resolve_judge("google/gemini-2.5-flash-lite", bench_var="ARASTORIES_JUDGE_MODEL_OVERRIDE")

_DIMENSIONS = {
    "fluency": "How smooth and natural the text is, including appropriate grammar, vocabulary, and sentence structure.",
    "coherence": "The logical connection and flow of sentences and ideas, making the text easy to understand.",
    "following_instructions": "How well the text adheres to the provided instructions or task requirements.",
    "consistency": "How consistently accurate and uniform the information and style are throughout the text.",
    "variety": "How well does the model generate story in the required Arabic variety.",
}


def _make_paper_aligned_rubric(metric_name: str, definition: str) -> str:
    criterion_label = metric_name.replace("_", " ").title()
    return f"""\
You are an expert in Arabic language, its dialects, and storytelling. I would like your help in evaluating a story written by a student based on a set of instructions.

Evaluate only this criterion on a 1-5 scale:
{criterion_label}: {definition}

Give the score directly without explanations or additions.
"""


@run_spec_function("arastories")
def get_arastories_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.arastories_scenario.AraStoriesScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,  # ASSUMPTION: zero-shot, no TRAIN_SPLIT seen
        num_outputs=1,
        max_tokens=2048,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(
            class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
            args={"metric_name": metric_name},
        )
        for metric_name in _DIMENSIONS
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL_NAME,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": metric_name,
                "rubric": _make_paper_aligned_rubric(metric_name, definition),
            },
        )
        for metric_name, definition in _DIMENSIONS.items()
    ]

    return RunSpec(
        name="arastories",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "arastories"],
        annotators=annotators,
    )
