"""HELM Run Specs for moh_x."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("moh_x")
def get_moh_x_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.moh_x_scenario.MOHXScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Respond with only one word: Yes or No.",
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,  # ASSUMPTION: zero-shot, no TRAIN_SPLIT seen
        num_outputs=1,
        max_tokens=16,
        temperature=0.0,
        stop_sequences=["\n"],
        output_mapping_pattern=r"\b(Yes|No)\b",
    )

    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
            args={"names": ["exact_match"]},
        ),
    ]

    return RunSpec(
        name="moh_x",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "moh_x"],
        annotators=None,
    )
