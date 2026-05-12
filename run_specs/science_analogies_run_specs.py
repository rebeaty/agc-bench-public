"""HELM Run Specs for science_analogies."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def _science_analogies_spec(*, subset: str, name: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.science_analogies_scenario.ScienceAnalogiesScenario",
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
        max_tokens=256,
        temperature=0.0,
        stop_sequences=["\n\n"],
    )

    metric_specs = [
        MetricSpec(
            class_name="metrics.science_analogies_metric.ScienceAnalogiesAutomaticMetric",
            args={},
        ),
    ]

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "science_analogies"],
        annotators=None,
    )


@run_spec_function("science_analogies")
def get_science_analogies_spec() -> RunSpec:
    return _science_analogies_spec(subset="nosrc", name="science_analogies")


@run_spec_function("science_analogies_wsrc")
def get_science_analogies_wsrc_spec() -> RunSpec:
    return _science_analogies_spec(subset="wsrc", name="science_analogies_wsrc")
