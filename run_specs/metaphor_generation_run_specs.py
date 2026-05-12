"""HELM Run Specs for metaphor_generation."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("metaphor_generation")
def get_metaphor_generation_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.metaphor_generation_scenario.MetaphorGenerationScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=(
            "Rewrite the literal sentence below as exactly one metaphorical sentence. "
            "Preserve the original meaning while making the main verb phrase more metaphorical. "
            "Output only the rewritten sentence."
        ),
        input_prefix="Literal sentence: ",
        input_suffix="\nMetaphorical sentence:",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,  # ASSUMPTION: zero-shot, no TRAIN_SPLIT seen
        num_outputs=1,
        max_tokens=48,
        temperature=0.7,
        stop_sequences=["\n"],
    )

    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
            args={"names": ["bleu_4", "rouge_l", "f1_score"]},
        ),
    ]

    return RunSpec(
        name="metaphor_generation",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "metaphor_generation"],
        annotators=None,
    )
