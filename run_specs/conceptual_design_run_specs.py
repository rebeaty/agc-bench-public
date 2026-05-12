"""HELM Run Specs for conceptual_design."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from llm_judge._judge_override import resolve_judge


_JUDGE_MODEL = resolve_judge("openai/gpt-4o", bench_var="CONCEPTUAL_DESIGN_JUDGE_MODEL_OVERRIDE")

_RUBRIC_CONCEPTUAL_DESIGN = """\
Evaluate the full set of generated conceptual design solutions for an engineering problem.
Consider the set as a whole and score the following dimensions on the anchored 0-2 scale.

1. feasibility: 0 = infeasible or not implementable, 1 = partially feasible, 2 = feasible and implementable
2. novelty: 0 = common or repetitive, 1 = somewhat novel, 2 = clearly novel and distinct from the reference pool
3. usefulness: 0 = off-topic or unhelpful, 1 = somewhat useful, 2 = useful and relevant to the prompt

Respond with a JSON object containing only the three integer scores.
"""


@run_spec_function("conceptual_design")
def get_conceptual_design_spec(prompt_variant: str = "base") -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.conceptual_design_scenario.ConceptualDesignScenario",
        args={"prompt_variant": prompt_variant},
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
        max_tokens=4000,
        temperature=0.9,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(
            class_name="metrics.conceptual_design_metric.ConceptualDesignMetric",
            args={},
        ),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.conceptual_design_annotator.ConceptualDesignAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 512,
                "rubric": _RUBRIC_CONCEPTUAL_DESIGN,
            },
        ),
    ]

    return RunSpec(
        name="conceptual_design",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "conceptual_design"],
        annotators=annotators,
    )
