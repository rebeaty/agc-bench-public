"""HELM Run Specs for story_generation_rocstories."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("story_generation_rocstories")
def get_story_generation_rocstories_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.story_generation_rocstories_scenario.StoryGenerationScenario",
        args={"dataset": "roc"},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=2048,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
            args={"names": ["rouge_1", "rouge_2", "rouge_l", "bleu_4"]},
        ),
    ]

    return RunSpec(
        name="story_generation_rocstories",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "story_generation_rocstories"],
        annotators=None,
    )
