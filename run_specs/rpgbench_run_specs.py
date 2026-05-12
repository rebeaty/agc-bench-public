"""HELM Run Specs for rpgbench."""

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

_RUBRIC_INTERESTINGNESS = """\
Evaluate the INTERESTINGNESS of the generated RPG game content.
Consider how engaging, surprising, and captivating the game content would be to players.

Score 1: Content is completely boring with no interesting elements
Score 2: Minimally interesting with poor engagement value
Score 3: Moderately interesting with some engaging elements
Score 4: Genuinely interesting content that would engage most players
Score 5: Highly interesting, captivating content with exceptional player engagement value
"""


@run_spec_function("rpgbench")
def get_rpgbench_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.rpgbench_scenario.RpgBenchScenario",
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
        max_tokens=4096,
        temperature=0.0,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.json_validity_metric.JsonValidityMetric", args={}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "interestingness"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4o", bench_var="RPGBENCH_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "interestingness",
                "rubric": _RUBRIC_INTERESTINGNESS,
            },
        ),
    ]

    return RunSpec(
        name="rpgbench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "rpgbench"],
        annotators=annotators,
    )
