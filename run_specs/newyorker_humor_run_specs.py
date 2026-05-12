"""HELM run specs for the New Yorker humor benchmark family."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def _newyorker_humor_spec(*, task: str, name: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.newyorker_humor_scenario.NewYorkerHumorScenario",
        args={"task": task, "prefer_official_from_description": True},
    )

    max_tokens = 16 if task in {"matching", "ranking"} else 256
    stop_sequences = ["\n", "."]

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="Answer with only the single capital-letter option. Do not provide words, explanations, or punctuation.",
        input_prefix="",
        input_suffix="\n",
        output_prefix="Answer: ",
        output_suffix="\n",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=max_tokens,
        temperature=0.0,
        stop_sequences=stop_sequences,
    )

    metric_specs = [
        MetricSpec(class_name="metrics.newyorker_humor_metric.NewYorkerHumorMetric", args={"task": task}),
    ]

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "newyorker_humor"],
        annotators=None,
    )


@run_spec_function("newyorker_humor")
def get_newyorker_humor_spec() -> RunSpec:
    return _newyorker_humor_spec(task="matching", name="newyorker_humor")


@run_spec_function("newyorker_humor_matching")
def get_newyorker_humor_matching_spec() -> RunSpec:
    return _newyorker_humor_spec(task="matching", name="newyorker_humor_matching")


@run_spec_function("newyorker_humor_ranking")
def get_newyorker_humor_ranking_spec() -> RunSpec:
    return _newyorker_humor_spec(task="ranking", name="newyorker_humor_ranking")
