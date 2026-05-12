"""HELM Run Specs for puntuguese."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_MULTIPLE_CHOICE_JOINT,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("puntuguese")
def get_puntuguese_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.puntuguese_scenario.PuntugueseScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=(
            "Leia o texto em português e classifique se ele é humorístico. "
            "Responda apenas com Sim ou Não."
        ),
        input_prefix="",
        input_suffix="\n",
        output_prefix="Resposta: ",
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
        name="puntuguese",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "puntuguese"],
        annotators=None,
    )
