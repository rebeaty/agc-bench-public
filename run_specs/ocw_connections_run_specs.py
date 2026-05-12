"""HELM Run Specs for ocw_connections."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("ocw_connections")
def get_ocw_connections_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.ocw_connections_scenario.OnlyConnectConnectionsScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=5,  # ASSUMPTION: few-shot, TRAIN_SPLIT seen
        num_outputs=1,
        max_tokens=512,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
            args={"names": ["exact_match", "rouge_1", "rouge_l"]},
        ),
        MetricSpec(
            class_name="metrics.bert_score_metric.BertScoreMetric",
            args={},
        ),
    ]

    return RunSpec(
        name="ocw_connections",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "ocw_connections"],
        annotators=None,
    )
