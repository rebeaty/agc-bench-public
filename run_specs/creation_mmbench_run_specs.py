"""HELM Run Specs for creation_mmbench."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION_MULTIMODAL,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("creation_mmbench")
def get_creation_mmbench_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.creation_mmbench_scenario.CreationMMBenchScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION_MULTIMODAL,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,  # ASSUMPTION: zero-shot, no TRAIN_SPLIT seen
        num_outputs=1,
        max_tokens=4096,
        temperature=0.0,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.creation_mmbench_metric.CreationMMBenchMetric", args={}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.creation_mmbench_annotator.CreationMMBenchAnnotator",
            args={
                "judge_model_name": "openai/gpt-4o",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 2048,
            },
        ),
    ]

    return RunSpec(
        name="creation_mmbench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "creation_mmbench"],
        annotators=annotators,
    )
