"""HELM Run Specs for rebus_puzzle."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION_MULTIMODAL,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("rebus_puzzle")
def get_rebus_puzzle_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.rebus_puzzle_scenario.RebusPuzzleScenario",
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
        max_tokens=256,
        temperature=0.0,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.rebus_puzzle_metric.RebusPuzzleMetric"),
        MetricSpec(class_name="llm_judge.rebus_puzzle_metric.RebusPuzzleJudgeMetric"),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.rebus_puzzle_annotator.RebusPuzzleAnnotator",
            args={
                "judge_model_name": "google/gemini-2.5-flash-lite",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 16,
            },
        ),
    ]

    return RunSpec(
        name="rebus_puzzle",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "rebus_puzzle"],
        annotators=annotators,
    )
