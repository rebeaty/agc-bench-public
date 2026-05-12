"""HELM Run Specs for futuregen."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def _futuregen_spec(*, prompt_style: str, name: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.futuregen_scenario.FuturegenScenario",
        args={"prompt_style": prompt_style},
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
        max_tokens=192,
        temperature=0.0,
        stop_sequences=["\n\n"],
    )

    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
            args={"names": ["rouge_l", "bleu_4"]},
        ),
        MetricSpec(
            class_name="metrics.bert_score_metric.BertScoreMetric",
            args={"model_type": "bert-base-uncased"},
        ),
        MetricSpec(
            class_name="metrics.futuregen_similarity_metric.FutureGenSimilarityMetric",
            args={},
        ),
    ]

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "futuregen"],
        annotators=None,
    )


@run_spec_function("futuregen")
def get_futuregen_spec() -> RunSpec:
    return _futuregen_spec(prompt_style="top3", name="futuregen")


@run_spec_function("futuregen_all_sections")
def get_futuregen_all_sections_spec() -> RunSpec:
    return _futuregen_spec(prompt_style="all_sections", name="futuregen_all_sections")
