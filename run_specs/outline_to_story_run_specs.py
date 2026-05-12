"""HELM Run Specs for outline_to_story."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("outline_to_story")
def get_outline_to_story_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.outline_to_story_scenario.OutlineToStoryScenario",
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
        max_tokens=1024,
        temperature=0.95,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
            args={"names": ["rouge_1", "rouge_2", "rouge_l", "bleu_4"]},
        ),
        MetricSpec(class_name="metrics.outline_to_story_metric.OutlineToStoryMetric", args={}),
    ]

    return RunSpec(
        name="outline_to_story",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "outline_to_story"],
        annotators=None,
    )
