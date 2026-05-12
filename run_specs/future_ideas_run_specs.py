"""HELM Run Specs for future_ideas."""

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

_JUDGE_MODEL = resolve_judge("openai/gpt-4", bench_var="FUTURE_IDEAS_JUDGE_MODEL_OVERRIDE")


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_LLM_JUDGE_NOVELTY = """\
Evaluate the NOVELTY of the generated future scientific or technological idea.
Consider how unprecedented, original, and forward-thinking the idea is.

Score 1: Idea is entirely conventional or already widely known
Score 2: Mostly incremental with minimal novelty
Score 3: Somewhat novel with some original elements
Score 4: Notably novel idea that goes beyond current common thinking
Score 5: Highly novel, unprecedented idea demonstrating exceptional forward-thinking creativity
"""

_RUBRIC_LLM_JUDGE_RELEVANCE = """\
Evaluate the RELEVANCE of the generated future idea to the given domain or topic.
Consider how well the idea addresses the specified area and aligns with the domain's challenges.

Score 1: Idea has no relevance to the specified domain or topic
Score 2: Minimal relevance with major off-topic elements
Score 3: Somewhat relevant but with notable gaps or off-topic elements
Score 4: Mostly relevant and well-aligned with the domain
Score 5: Perfectly relevant and highly focused on the domain's core challenges
"""

_RUBRIC_LLM_JUDGE_FEASIBILITY = """\
Evaluate the FEASIBILITY of the generated future idea.
Consider whether the proposed direction is realistic, scientifically grounded, and plausibly investigable.

Score 1: Idea is infeasible or scientifically implausible
Score 2: Idea has major feasibility problems or missing prerequisites
Score 3: Idea is partially feasible but has notable practical or scientific gaps
Score 4: Idea is mostly feasible with reasonable methods or assumptions
Score 5: Idea is highly feasible, well-grounded, and realistic to investigate
"""


def _build_future_ideas_spec(name: str, domain: str) -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.future_ideas_scenario.FutureIdeasScenario",
        args={"domain": domain},
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
        max_tokens=3072,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_novelty"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_relevance"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_feasibility"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_novelty",
                "rubric": _RUBRIC_LLM_JUDGE_NOVELTY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_relevance",
                "rubric": _RUBRIC_LLM_JUDGE_RELEVANCE,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_feasibility",
                "rubric": _RUBRIC_LLM_JUDGE_FEASIBILITY,
            },
        ),
    ]

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "future_ideas"],
        annotators=annotators,
    )


@run_spec_function("future_ideas")
def get_future_ideas_spec() -> RunSpec:
    return _build_future_ideas_spec(name="future_ideas", domain="all")


@run_spec_function("future_ideas_chemistry")
def get_future_ideas_chemistry_spec() -> RunSpec:
    return _build_future_ideas_spec(name="future_ideas_chemistry", domain="chemistry")


@run_spec_function("future_ideas_computer")
def get_future_ideas_computer_spec() -> RunSpec:
    return _build_future_ideas_spec(name="future_ideas_computer", domain="computer")


@run_spec_function("future_ideas_economics")
def get_future_ideas_economics_spec() -> RunSpec:
    return _build_future_ideas_spec(name="future_ideas_economics", domain="economics")


@run_spec_function("future_ideas_medical")
def get_future_ideas_medical_spec() -> RunSpec:
    return _build_future_ideas_spec(name="future_ideas_medical", domain="medical")


@run_spec_function("future_ideas_physics")
def get_future_ideas_physics_spec() -> RunSpec:
    return _build_future_ideas_spec(name="future_ideas_physics", domain="physics")
