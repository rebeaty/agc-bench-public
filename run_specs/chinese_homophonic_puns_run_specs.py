"""HELM Run Specs for chinese_homophonic_puns."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("chinese_homophonic_puns")
def get_chinese_homophonic_puns_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.chinese_homophonic_puns_scenario.ChineseHomophonicPunsScenario",
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
        max_tokens=16,
        temperature=0.0,
        stop_sequences=["\n"],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.chinese_homophonic_puns_metric.ChineseHomophonicPunsMetric"),
    ]

    return RunSpec(
        name="chinese_homophonic_puns",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "chinese_homophonic_puns"],
        annotators=None,
    )
