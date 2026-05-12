"""HELM Run Specs for slang_generation."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("slang_generation")
def get_slang_generation_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.slang_generation_scenario.SlangGenerationScenario",
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
        max_tokens=256,
        temperature=1.0,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(
            class_name="metrics.slang_generation_metric.SlangGenerationMetric",
            args={},
        ),
    ]

    return RunSpec(
        name="slang_generation",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "slang_generation"],
        annotators=None,
    )
