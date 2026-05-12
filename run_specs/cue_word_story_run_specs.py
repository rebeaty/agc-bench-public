"""HELM Run Specs for cue_word_story."""

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

_RUBRIC_LLM_JUDGE_CREATIVITY = """\
Evaluate the CREATIVITY of the generated short story.
Consider the overall creative quality, including inventive use of the cue words,
novel imagery, and whether the story feels fresh rather than formulaic.

Score 1: Completely conventional or formulaic story
Score 2: Mostly conventional with little imaginative use of the cue words
Score 3: Some creative elements and acceptable use of the cue words
Score 4: Genuinely creative story with strong imaginative use of the cue words
Score 5: Highly creative, novel, and memorable story with especially inventive cue-word use
"""

_RUBRIC_LLM_JUDGE_ORIGINALITY = """\
Evaluate the ORIGINALITY of the generated short story.
Consider how unique, fresh, and unconventional the story concept is.

Score 1: Highly derivative or clichéd; the cue words are used in the most obvious way
Score 2: Mostly familiar ideas with only a small twist
Score 3: Some original elements, but the story is still largely conventional
Score 4: Notably original and uncommon use of the cue words
Score 5: Exceptionally original, fresh, and unconventional story concept
"""

_RUBRIC_LLM_JUDGE_SURPRISE = """\
Evaluate the SURPRISE of the generated short story.
Consider how unexpected the story is, including twists, reversals, and unusual
transitions between ideas.

Score 1: Completely predictable; no unexpected turns
Score 2: Mostly predictable with almost no surprise
Score 3: Some mildly unexpected ideas or transitions
Score 4: Clearly surprising with at least one strong twist or reversal
Score 5: Highly surprising and inventive with memorable unexpected turns
"""

_RUBRIC_LLM_JUDGE_EFFECTIVENESS = """\
Evaluate the EFFECTIVENESS of the generated short story.
Consider whether the story is coherent, easy to follow, and enjoyable to read.

Score 1: Hard to follow, dull, or fails to work as a story
Score 2: Weakly effective with awkward flow or low engagement
Score 3: Adequately effective but not especially engaging
Score 4: Well-structured, coherent, and enjoyable
Score 5: Highly effective story that is both coherent and engaging
"""


@run_spec_function("cue_word_story")
def get_cue_word_story_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.cue_word_story_scenario.CueWordStoryScenario",
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
        max_tokens=512,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]}),
        MetricSpec(
            class_name="metrics.cue_word_story_metric.CueWordStoryMetric",
            args={"model_name": "tfidf_char_3_5"},
        ),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_creativity"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_originality"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_surprise"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_effectiveness"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4", bench_var="CUE_WORD_STORY_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_creativity",
                "rubric": _RUBRIC_LLM_JUDGE_CREATIVITY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4", bench_var="CUE_WORD_STORY_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_originality",
                "rubric": _RUBRIC_LLM_JUDGE_ORIGINALITY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4", bench_var="CUE_WORD_STORY_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_surprise",
                "rubric": _RUBRIC_LLM_JUDGE_SURPRISE,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4", bench_var="CUE_WORD_STORY_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_effectiveness",
                "rubric": _RUBRIC_LLM_JUDGE_EFFECTIVENESS,
            },
        ),
    ]

    return RunSpec(
        name="cue_word_story",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "cue_word_story"],
        annotators=annotators,
    )
