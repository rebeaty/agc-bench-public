"""HELM Run Specs for scimon."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("scimon")
def get_scimon_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.scimon_scenario.SciMONScenario",
        args={},
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
        max_tokens=96,
        temperature=0.2,
        stop_sequences=["\n"],
    )

    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
            args={"names": ["rouge_l", "bleu_4"]},
        ),
        MetricSpec(
            class_name="metrics.bert_score_metric.BertScoreMetric",
            args={"model_type": "allenai/scibert_scivocab_uncased"},
        ),
    ]

    return RunSpec(
        name="scimon",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "scimon"],
        annotators=None,
    )
