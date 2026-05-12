"""HELM Run Specs for the HypoGen bit-flip proxy."""

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

_RUBRIC_LLM_JUDGE_PROXY_NOVELTY = """\
Evaluate the NOVELTY of the generated hypothesis on the local HypoGen bit-flip proxy task.
Score only whether the candidate proposes an original and interesting "flip" relative to the
provided abstract and conventional limitation ("bit"). Do not treat this as a score for the
full AI-Researcher ideation benchmark from the paper.

Score 1: Hypothesis shows no novel thinking; follows the conventional assumption
Score 2: Minimal novelty; makes only slight modifications to the assumption
Score 3: Somewhat novel hypothesis with partial inversion of the assumption
Score 4: Notably novel hypothesis clearly inverting the assumption in an interesting way
Score 5: Highly novel hypothesis presenting a completely original and insightful inversion
"""


@run_spec_function("hypogen")
def get_hypogen_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.hypogen_scenario.HypoGenScenario",
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
            args={"metric_name": "hypogen_proxy_novelty"},
        ),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4o-mini", bench_var="HYPOGEN_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "hypogen_proxy_novelty",
                "rubric": _RUBRIC_LLM_JUDGE_PROXY_NOVELTY,
            },
        ),
    ]

    return RunSpec(
        name="hypogen",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "hypogen", "proxy"],
        annotators=annotators,
    )
