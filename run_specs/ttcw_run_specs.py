"""HELM Run Specs for ttcw."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("ttcw")
def get_ttcw_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.ttcw_scenario.TTCWScenario",
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
        max_tokens=2048,
        temperature=0.0,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.ttcw_metric.TTCWMetric", args={}),
    ]

    return RunSpec(
        name="ttcw",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "ttcw"],
        annotators=None,
    )
