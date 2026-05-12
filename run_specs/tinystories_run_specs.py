"""HELM Run Specs for tinystories."""

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

@run_spec_function("tinystories")
def get_tinystories_spec() -> RunSpec:
    output_override = os.environ.get("TINYSTORIES_NUM_OUTPUTS_OVERRIDE", "").strip()
    num_outputs = int(output_override) if output_override.isdigit() and int(output_override) == 1 else 1

    scenario_spec = ScenarioSpec(
        class_name="scenarios.tinystories_scenario.TinyStoriesScenario",
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
        num_outputs=num_outputs,
        max_tokens=1024,
        temperature=1.0,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="llm_judge.tinystories_metric.TinyStoriesMetric"),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.tinystories_annotator.TinyStoriesAnnotator",
            args={
                "judge_model_name": resolve_judge("google/gemini-2.5-flash-lite", bench_var="TINYSTORIES_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 128,
            },
        ),
    ]

    return RunSpec(
        name="tinystories",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "tinystories"],
        annotators=annotators,
    )
