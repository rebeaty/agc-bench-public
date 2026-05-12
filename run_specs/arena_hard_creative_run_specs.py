"""HELM Run Specs for arena_hard_creative."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from llm_judge._judge_override import resolve_judge


@run_spec_function("arena_hard_creative")
def get_arena_hard_creative_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.arena_hard_creative_scenario.ArenaHardCreativeScenario",
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
        max_tokens=2048,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.arena_hard_pairwise_metric.ArenaHardPairwiseMetric", args={}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.arena_hard_pairwise_annotator.ArenaHardPairwiseAnnotator",
            args={
                "category": "creative_writing",
                "judge_model_name": resolve_judge("openai/gpt-4.1", bench_var="ARENA_HARD_CREATIVE_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 16000,
            },
        ),
    ]

    return RunSpec(
        name="arena_hard_creative",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "arena_hard_creative"],
        annotators=annotators,
    )
