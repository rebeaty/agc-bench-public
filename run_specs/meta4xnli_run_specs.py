"""HELM Run Specs for meta4xnli."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION, ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def _interpretation_spec(*, subset: str, name: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.meta4xnli_scenario.Meta4XNLIScenario",
        args={"subset": subset},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="",
        input_prefix="",
        input_suffix="\n",
        output_prefix="Answer: ",
        output_suffix="\n",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=16,
        temperature=0.0,
        stop_sequences=["\n"],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.markdown_normalized_classification_metric.MarkdownNormalizedMCQClassificationMetric", args={}),
    ]

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "meta4xnli"],
        annotators=None,
    )


def _detection_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.meta4xnli_scenario.Meta4XNLIScenario",
        args={"subset": "detection_en"},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=512,
        temperature=0.0,
        stop_sequences=["\n"],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.meta4xnli_detection_metric.Meta4XNLIDetectionMetric", args={}),
    ]

    return RunSpec(
        name="meta4xnli_detection_en",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "meta4xnli"],
        annotators=None,
    )


@run_spec_function("meta4xnli")
def get_meta4xnli_spec() -> RunSpec:
    return _interpretation_spec(subset="interpretation_en", name="meta4xnli")


@run_spec_function("meta4xnli_interpretation_es")
def get_meta4xnli_interpretation_es_spec() -> RunSpec:
    return _interpretation_spec(subset="interpretation_es", name="meta4xnli_interpretation_es")


@run_spec_function("meta4xnli_interpretation_en_cot")
def get_meta4xnli_interpretation_en_cot_spec() -> RunSpec:
    return _interpretation_spec(subset="interpretation_en_cot", name="meta4xnli_interpretation_en_cot")


@run_spec_function("meta4xnli_interpretation_es_cot")
def get_meta4xnli_interpretation_es_cot_spec() -> RunSpec:
    return _interpretation_spec(subset="interpretation_es_cot", name="meta4xnli_interpretation_es_cot")


@run_spec_function("meta4xnli_detection_en")
def get_meta4xnli_detection_en_spec() -> RunSpec:
    return _detection_spec()
