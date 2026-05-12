"""HELM Run Specs for humor_transfer."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def _humor_transfer_spec(*, subset: str, name: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.humor_transfer_scenario.HumorTransferScenario",
        args={"subset": subset},
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
        max_tokens=16,
        temperature=0.0,
        stop_sequences=["\n"],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.humor_transfer_metric.HumorTransferMetric", args={}),
    ]

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "humor_transfer"],
        annotators=None,
    )


@run_spec_function("humor_transfer")
def get_humor_transfer_spec() -> RunSpec:
    return _humor_transfer_spec(subset="sarcasm_headlines", name="humor_transfer")


@run_spec_function("humor_transfer_amazon_questions")
def get_humor_transfer_amazon_questions_spec() -> RunSpec:
    return _humor_transfer_spec(
        subset="amazon_questions",
        name="humor_transfer_amazon_questions",
    )
