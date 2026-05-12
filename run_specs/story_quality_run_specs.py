"""HELM Run Specs for story_quality."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("story_quality")
def get_story_quality_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.story_quality_scenario.StoryQualityScenario",
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
        max_tokens=512,
        # Lowered from 0.7 to 0.0 (2026-04-25 audit): output is a numeric
        # rating, sampling temperature shouldn't add noise.
        temperature=0.0,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.correlation_metric.CorrelationMetric", args={"correlation_type": "pearson"}),
        MetricSpec(class_name="metrics.correlation_metric.CorrelationMetric", args={"correlation_type": "spearman"}),
    ]

    return RunSpec(
        name="story_quality",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "story_quality"],
        annotators=None,
    )
