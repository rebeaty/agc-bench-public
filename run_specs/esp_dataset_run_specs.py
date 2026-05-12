"""HELM Run Specs for esp_dataset."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION_MULTIMODAL,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("esp_dataset")
def get_esp_dataset_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.esp_dataset_scenario.ESPDatasetScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION_MULTIMODAL,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,  # ASSUMPTION: zero-shot, no TRAIN_SPLIT seen
        num_outputs=1,
        max_tokens=192,
        temperature=0.2,
        stop_sequences=["\n\n"],
    )

    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
            args={"names": ["bleu_4", "cider"]},
        ),
        MetricSpec(class_name="metrics.meteor_metric.MeteorMetric", args={}),
        MetricSpec(
            class_name="metrics.bert_score_metric.BertScoreMetric",
            args={"model_type": "bert-base-uncased"},
        ),
    ]

    return RunSpec(
        name="esp_dataset",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "esp_dataset"],
        annotators=None,
    )
