"""HELM Run Specs for grapheval_iclr."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("grapheval_iclr")
def get_grapheval_iclr_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.grapheval_iclr_scenario.GraphEvalICLRScenario",
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
        temperature=0.1,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.grapheval_decision_metric.GraphEvalDecisionMetric", args={}),
    ]

    return RunSpec(
        name="grapheval_iclr",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "grapheval_iclr"],
        annotators=None,
    )
