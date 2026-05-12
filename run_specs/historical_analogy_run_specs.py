"""HELM Run Specs for historical_analogy."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("historical_analogy")
def get_historical_analogy_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.historical_analogy_scenario.HistoricalAnalogyScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="",
        output_prefix="",
        output_suffix="",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=32,
        temperature=0.0,
        stop_sequences=["\n"],
    )

    metric_specs = [
        MetricSpec(
            class_name="metrics.historical_analogy_metric.HistoricalAnalogyMetric",
            args={},
        ),
    ]

    return RunSpec(
        name="historical_analogy",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "historical_analogy"],
    )
