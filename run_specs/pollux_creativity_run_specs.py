"""HELM Run Specs for pollux_creativity."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from llm_judge._judge_override import resolve_judge


# ── Rubrics ──────────────────────────────────────────────────────────────────

@run_spec_function("pollux_creativity")
def get_pollux_creativity_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.pollux_creativity_scenario.POLLUXCreativityScenario",
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
        MetricSpec(
            class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
            args={"metric_name": "pollux_score"},
        ),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.pollux_creativity_annotator.POLLUXCreativityAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4.1-mini", bench_var="POLLUX_CREATIVITY_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 512,
            },
        ),
    ]

    return RunSpec(
        name="pollux_creativity",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "pollux_creativity"],
        annotators=annotators,
    )
