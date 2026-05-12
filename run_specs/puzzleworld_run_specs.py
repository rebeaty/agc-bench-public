"""HELM Run Specs for puzzleworld."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION_MULTIMODAL,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("puzzleworld")
def get_puzzleworld_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.puzzleworld_scenario.PuzzleWorldScenario",
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
        MetricSpec(class_name="metrics.puzzleworld_answer_metric.PuzzleWorldAnswerMetric"),
    ]

    return RunSpec(
        name="puzzleworld",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "puzzleworld"],
        annotators=None,
    )
