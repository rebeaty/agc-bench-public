"""HELM Run Specs for arn."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("arn")
def get_arn_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.arn_scenario.ARNScenario",
        args={},
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
        max_tokens=1024,
        temperature=0.0,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(
            class_name="metrics.arn_metric.ARNMetric",
            args={},
        ),
    ]

    return RunSpec(
        name="arn",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "arn"],
        annotators=None,
    )
