"""HELM Run Specs for simile_generation."""

import os

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("simile_generation")
def get_simile_generation_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.simile_generation_scenario.SimileGenerationScenario",
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
        max_tokens=48,
        temperature=0.7,
        stop_sequences=["\n"],
    )

    metric_specs = [
        MetricSpec(
            class_name="metrics.simile_generation_metric.SimileGenerationMetric",
            args={"bert_score_model": os.environ.get("SIMILE_GENERATION_BERTSCORE_MODEL_OVERRIDE", "roberta-large")},
        ),
    ]

    return RunSpec(
        name="simile_generation",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "simile_generation"],
        annotators=None,
    )
