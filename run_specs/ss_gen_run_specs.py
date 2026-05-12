"""HELM Run Specs for ss_gen."""

import os

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from llm_judge._judge_override import resolve_judge

_JUDGE_MODEL = resolve_judge("openai/gpt-4", bench_var="SS_GEN_JUDGE_MODEL_OVERRIDE")

# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_LLM_JUDGE_COHERENCE = """\
Evaluate the COHERENCE of the generated story segment or continuation.
Consider logical flow, narrative consistency, and how well it connects to the preceding context.

Score 1: Generated segment is completely incoherent with the story context
Score 2: Poor coherence with major narrative inconsistencies
Score 3: Adequate coherence with some narrative flow issues
Score 4: Good coherence that maintains narrative consistency
Score 5: Excellent coherence with perfect narrative flow and consistency
"""

_RUBRIC_LLM_JUDGE_DESCRIPTIVENESS = """\
Evaluate the DESCRIPTIVENESS of the generated Social Story.
Consider whether the story explains the situation clearly and emphasizes descriptive,
supportive narration rather than direct or overly imperative coaching.

Score 1: Story is mostly vague, directive, or lacks meaningful description
Score 2: Limited descriptiveness with too little situational explanation
Score 3: Moderately descriptive with some clear contextual detail
Score 4: Strong descriptiveness with clear, supportive explanation of the situation
Score 5: Exceptionally descriptive, clear, and well-contextualized while staying supportive
"""

_RUBRIC_LLM_JUDGE_EMPATHY = """\
Evaluate the EMPATHY of the generated Social Story for children and teens with autism.
Consider whether the tone is positive, patient, supportive, and respectful of the audience.

Score 1: Tone is insensitive, harsh, or poorly suited to the audience
Score 2: Limited empathy with noticeable tone or audience-fit problems
Score 3: Adequately empathetic but uneven in supportiveness or tone
Score 4: Clearly empathetic, patient, and appropriate for the audience
Score 5: Highly empathetic, reassuring, and exceptionally well-tailored to the audience
"""

_RUBRIC_LLM_JUDGE_GRAMMATICALITY = """\
Evaluate the GRAMMATICALITY of the generated Social Story.
Consider grammar, syntax, fluency, and surface-level writing correctness.

Score 1: Severe grammatical problems that make the story hard to understand
Score 2: Frequent grammatical or fluency problems
Score 3: Understandable with some noticeable grammatical issues
Score 4: Mostly grammatical and fluent with only minor issues
Score 5: Grammatically strong, fluent, and polished throughout
"""

_RUBRIC_LLM_JUDGE_RELEVANCE = """\
Evaluate the RELEVANCE of the generated Social Story to the requested title and social situation.
Consider whether the story stays on topic and addresses the intended intervention goal.

Score 1: Story is off-topic or fails to address the requested title
Score 2: Weak relevance with major gaps in topic alignment
Score 3: Moderately relevant but with some drift or missing focus
Score 4: Clearly relevant and well-aligned with the requested topic
Score 5: Highly relevant, focused, and directly responsive to the requested title and purpose
"""


@run_spec_function("ss_gen")
def get_ss_gen_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.ss_gen_scenario.SSGenScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,  # zero-shot title-to-story generation per paper
        num_outputs=1,
        max_tokens=512,
        temperature=0.0,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
            args={"names": ["bleu_4", "rouge_1", "rouge_2", "rouge_l"]},
        ),
        MetricSpec(
            class_name="metrics.bert_score_metric.BertScoreMetric",
            args={"model_type": "bert-base-uncased"},
        ),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_coherence"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_descriptiveness"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_empathy"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_grammaticality"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_relevance"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_coherence",
                "rubric": _RUBRIC_LLM_JUDGE_COHERENCE,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_descriptiveness",
                "rubric": _RUBRIC_LLM_JUDGE_DESCRIPTIVENESS,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_empathy",
                "rubric": _RUBRIC_LLM_JUDGE_EMPATHY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_grammaticality",
                "rubric": _RUBRIC_LLM_JUDGE_GRAMMATICALITY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_relevance",
                "rubric": _RUBRIC_LLM_JUDGE_RELEVANCE,
            },
        ),
    ]

    return RunSpec(
        name="ss_gen",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "ss_gen"],
        annotators=annotators,
    )
