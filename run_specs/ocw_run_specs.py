"""HELM Run Specs for ocw."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("ocw")
def get_ocw_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.ocw_scenario.OnlyConnectWallScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        # HELM's generic generation adapter currently inserts `n/a` for OCW's
        # train examples here, so zero-shot is the strictest non-misleading
        # setting until a task-specific adapter reproduces the upstream prompt.
        max_train_instances=0,
        num_outputs=1,
        max_tokens=144,
        temperature=0.0,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.group_match_score_metric.GroupMatchScoreMetric", args={}),
    ]

    return RunSpec(
        name="ocw",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "ocw"],
        annotators=None,
    )
