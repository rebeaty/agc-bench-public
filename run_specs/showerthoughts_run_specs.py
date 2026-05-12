"""HELM run spec for showerthoughts."""

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

_RUBRIC_LLM_JUDGE_GENERAL_SCORE = """\
You are rating a single Showerthought in the style of the paper's human survey.
Use the paper's six-point Likert scale:
1 = strongly disagree / very poor
2 = disagree / poor
3 = somewhat disagree / below average
4 = somewhat agree / above average
5 = agree / good
6 = strongly agree / excellent

Rate the statement: "I like this Showerthought."
Judge the overall quality of the Showerthought as a standalone short text.
"""

_RUBRIC_LLM_JUDGE_LOGICAL_VALIDITY = """\
You are rating a single Showerthought in the style of the paper's human survey.
Use the paper's six-point Likert scale:
1 = strongly disagree / very poor
2 = disagree / poor
3 = somewhat disagree / below average
4 = somewhat agree / above average
5 = agree / good
6 = strongly agree / excellent

Rate the statement: "It makes a true/valid/logical statement."
Judge whether the Showerthought makes sense and holds up logically.
"""

_RUBRIC_LLM_JUDGE_CREATIVITY = """\
You are rating a single Showerthought in the style of the paper's human survey.
Use the paper's six-point Likert scale:
1 = strongly disagree / very poor
2 = disagree / poor
3 = somewhat disagree / below average
4 = somewhat agree / above average
5 = agree / good
6 = strongly agree / excellent

Rate the statement: "It is creative."
Judge originality, novelty, and whether the Showerthought offers an unexpected
or fresh perspective.
"""

_RUBRIC_LLM_JUDGE_HUMOR = """\
You are rating a single Showerthought in the style of the paper's human survey.
Use the paper's six-point Likert scale:
1 = strongly disagree / very poor
2 = disagree / poor
3 = somewhat disagree / below average
4 = somewhat agree / above average
5 = agree / good
6 = strongly agree / excellent

Rate the statement: "It is funny."
Judge how amusing or humorous the Showerthought is.
"""

_RUBRIC_LLM_JUDGE_CLEVERNESS = """\
You are rating a single Showerthought in the style of the paper's human survey.
Use the paper's six-point Likert scale:
1 = strongly disagree / very poor
2 = disagree / poor
3 = somewhat disagree / below average
4 = somewhat agree / above average
5 = agree / good
6 = strongly agree / excellent

Rate the statement: "It is clever."
Judge how witty, sharp, or intellectually perceptive the Showerthought is.
"""

_RUBRIC_LLM_JUDGE_REAL_PERSON = """\
You are rating a single Showerthought in the style of the paper's human survey.
Use the paper's six-point Likert scale:
1 = strongly disagree / very poor
2 = disagree / poor
3 = somewhat disagree / below average
4 = somewhat agree / above average
5 = agree / good
6 = strongly agree / excellent

Rate the statement: "I believe this Showerthought has been written by a real person."
Judge only how human-written the Showerthought appears, not whether you like it.
"""


@run_spec_function("showerthoughts")
def get_showerthoughts_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.showerthoughts_scenario.ShowerthoughtsScenario",
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
        max_tokens=64,
        temperature=0.7,
        stop_sequences=["\n"],
    )

    metric_specs = [
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_general_score"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_logical_validity"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_creativity"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_humor"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_cleverness"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_real_person_likelihood"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4", bench_var="SHOWERTHOUGHTS_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_general_score",
                "rubric": _RUBRIC_LLM_JUDGE_GENERAL_SCORE,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4", bench_var="SHOWERTHOUGHTS_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_logical_validity",
                "rubric": _RUBRIC_LLM_JUDGE_LOGICAL_VALIDITY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4", bench_var="SHOWERTHOUGHTS_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_creativity",
                "rubric": _RUBRIC_LLM_JUDGE_CREATIVITY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4", bench_var="SHOWERTHOUGHTS_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_humor",
                "rubric": _RUBRIC_LLM_JUDGE_HUMOR,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4", bench_var="SHOWERTHOUGHTS_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_cleverness",
                "rubric": _RUBRIC_LLM_JUDGE_CLEVERNESS,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4", bench_var="SHOWERTHOUGHTS_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_real_person_likelihood",
                "rubric": _RUBRIC_LLM_JUDGE_REAL_PERSON,
            },
        ),
    ]

    return RunSpec(
        name="showerthoughts",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "showerthoughts"],
        annotators=annotators,
    )
