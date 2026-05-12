"""HELM Run Specs for eqbench_creative_writing_v3."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from llm_judge._judge_override import resolve_judge


@run_spec_function("eqbench_creative_writing_v3")
def get_eqbench_creative_writing_v3_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.eqbench_creative_writing_v3_scenario.EQBenchCreativeWritingV3Scenario",
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
        max_tokens=4000,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.eqbench_creative_writing_v3_metric.EQBenchCreativeWritingV3Metric", args={}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.eqbench_creative_writing_v3_annotator.EQBenchCreativeWritingV3Annotator",
            args={
                "judge_model_name": resolve_judge("google/gemini-3-flash-preview", bench_var="EQBENCH_CREATIVE_WRITING_V3_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 4096,
            },
        ),
    ]

    return RunSpec(
        name="eqbench_creative_writing_v3",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "eqbench_creative_writing_v3"],
        annotators=annotators,
    )
