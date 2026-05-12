"""HELM Run Specs for unfun_corpus."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("unfun_corpus")
def get_unfun_corpus_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.unfun_corpus_scenario.UnfunCorpusScenario",
        args={"prompt_style": "chat", "context_size": 8, "seed": 1234, "include_validation": False},
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
        max_tokens=64,
        temperature=0.0,
        stop_sequences=["\n"],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.unfun_automatic_metric.UnfunAutomaticMetric", args={}),
    ]

    return RunSpec(
        name="unfun_corpus",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "unfun_corpus"],
        annotators=None,
    )
