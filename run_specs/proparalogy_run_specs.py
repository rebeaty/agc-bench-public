"""HELM Run Specs for proparalogy."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_MULTIPLE_CHOICE_JOINT,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("proparalogy")
def get_proparalogy_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.proparalogy_scenario.ProparaLogyScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Choose the single best answer. Respond with only one letter: A, B, or C.",
        input_prefix="",
        input_suffix="\n",
        output_prefix="Answer: ",
        output_suffix="\n",
        max_train_instances=0,  # ASSUMPTION: zero-shot, no TRAIN_SPLIT seen
        num_outputs=1,
        max_tokens=16,
        temperature=0.0,
        stop_sequences=["\n"],
        output_mapping_pattern=r"\b([ABC])\b",
    )

    metric_specs = [
        MetricSpec(class_name="metrics.markdown_normalized_classification_metric.MarkdownNormalizedMCQClassificationMetric", args={}),
    ]

    return RunSpec(
        name="proparalogy",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "proparalogy"],
        annotators=None,
    )
