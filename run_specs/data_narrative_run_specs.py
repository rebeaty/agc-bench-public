"""HELM Run Specs for data_narrative."""

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

_RUBRIC_DATA_NARRATIVE_RELEVANCE = """\
Evaluate the RELEVANCE of the generated data narrative.
Consider whether the response stays focused on the stated intent/theme and
selects information from the table that is actually pertinent to that theme.

Score 1: Largely off-topic or ignores the intended theme
Score 2: Weak relevance with substantial digressions or mismatched focus
Score 3: Moderately relevant but misses or blurs the main theme
Score 4: Strongly relevant with only minor omissions or drift
Score 5: Highly relevant and tightly centered on the intended theme
"""

_RUBRIC_DATA_NARRATIVE_CLARITY_COHERENCE = """\
Evaluate the CLARITY AND COHERENCE of the generated data narrative.
Consider readability, logical flow, sentence-to-sentence connection, and how
well the response organizes its ideas into a coherent paragraph.

Score 1: Confusing, disjointed, or very hard to follow
Score 2: Noticeable coherence issues that weaken comprehension
Score 3: Understandable but uneven or mechanically organized
Score 4: Clear and coherent with only minor flow issues
Score 5: Exceptionally clear, cohesive, and easy to follow
"""

_RUBRIC_DATA_NARRATIVE_INFORMATIVENESS = """\
Evaluate the INFORMATIVENESS of the generated data narrative.
Consider whether the response captures the important trends, comparisons,
patterns, or notable points in the table instead of remaining generic.

Score 1: Barely informative; misses nearly all important insights
Score 2: Limited information with many key insights omitted
Score 3: Moderately informative but incomplete or shallow
Score 4: Informative with good coverage of the main insights
Score 5: Highly informative and richly covers the key table insights
"""

_RUBRIC_DATA_NARRATIVE_NARRATIVE_QUALITY = """\
Evaluate the NARRATIVE QUALITY of the generated data narrative.
Consider whether the response turns the table observations into an insightful,
engaging paragraph instead of a flat list of facts, while still following the
stated topic and intent.

Score 1: Dry, fragmentary, or not meaningfully narrative
Score 2: Minimally narrative with weak insight or awkward phrasing
Score 3: Adequate paragraph with some narrative flow but limited insight
Score 4: Strong narrative framing with clear insight and readable prose
Score 5: Highly engaging, insightful, and well-shaped as a narrative paragraph
"""

_RUBRIC_DATA_NARRATIVE_FACTUAL_CORRECTNESS = """\
Evaluate the FACTUAL CORRECTNESS of the generated data narrative.
Consider whether numeric statements, comparisons, trends, and other claims are
supported by the provided table and avoid hallucinated or incorrect details.

Score 1: Pervasively incorrect or hallucinatory
Score 2: Multiple factual mistakes or unsupported claims
Score 3: Mostly correct but with at least one meaningful factual issue
Score 4: Factually solid with only minor imprecision
Score 5: Factually accurate and fully grounded in the table
"""


@run_spec_function("data_narrative")
def get_data_narrative_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.data_narrative_scenario.DataNarrativeScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,  # Paper framing is not a HELM-style few-shot setup
        num_outputs=1,
        max_tokens=512,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["bleu_4"]}),
        MetricSpec(class_name="metrics.bert_score_metric.BertScoreMetric", args={"model_type": "bert-base-uncased"}),
        MetricSpec(
            class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
            args={"metric_name": "data_narrative_relevance"},
        ),
        MetricSpec(
            class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
            args={"metric_name": "data_narrative_clarity_coherence"},
        ),
        MetricSpec(
            class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
            args={"metric_name": "data_narrative_informativeness"},
        ),
        MetricSpec(
            class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
            args={"metric_name": "data_narrative_narrative_quality"},
        ),
        MetricSpec(
            class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
            args={"metric_name": "data_narrative_factual_correctness"},
        ),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("google/gemini-2.5-flash-lite", bench_var="DATA_NARRATIVE_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 16,
                "metric_name": "data_narrative_relevance",
                "rubric": _RUBRIC_DATA_NARRATIVE_RELEVANCE,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("google/gemini-2.5-flash-lite", bench_var="DATA_NARRATIVE_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 16,
                "metric_name": "data_narrative_clarity_coherence",
                "rubric": _RUBRIC_DATA_NARRATIVE_CLARITY_COHERENCE,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("google/gemini-2.5-flash-lite", bench_var="DATA_NARRATIVE_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 16,
                "metric_name": "data_narrative_informativeness",
                "rubric": _RUBRIC_DATA_NARRATIVE_INFORMATIVENESS,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("google/gemini-2.5-flash-lite", bench_var="DATA_NARRATIVE_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 16,
                "metric_name": "data_narrative_narrative_quality",
                "rubric": _RUBRIC_DATA_NARRATIVE_NARRATIVE_QUALITY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": resolve_judge("google/gemini-2.5-flash-lite", bench_var="DATA_NARRATIVE_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 16,
                "metric_name": "data_narrative_factual_correctness",
                "rubric": _RUBRIC_DATA_NARRATIVE_FACTUAL_CORRECTNESS,
            },
        ),
    ]

    return RunSpec(
        name="data_narrative",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "data_narrative"],
        annotators=annotators,
    )
