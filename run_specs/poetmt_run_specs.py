"""HELM Run Specs for poetmt."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from llm_judge._judge_override import resolve_judge

_JUDGE_MODEL = resolve_judge("google/gemini-3-flash-preview", bench_var="POETMT_JUDGE_MODEL_OVERRIDE")

# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_LLM_JUDGE_BEAUTY_OF_SOUND = """\
Evaluate the beauty of sound in the given Chinese translation of classical poetry.
Focus on whether the translation achieves harmonious sound, adherence to strict metrical rules, and a rhythm.

1 point: Poor translation, lacks harmony and adherence to metrical rules, and fails to capture the beauty of sound.
2 point: Below average, some rhyme and meter present but with noticeable imperfections and awkwardness.
3 point: Basic translation, captures some aspects of sound beauty but with several imperfections in rhyme, meter, or rhythm.
4 point: Good translation, mostly harmonious with minor imperfections in sound quality or adherence to metrical rules.
5 point: Excellent translation, achieves harmonious sound, precise wording, strict adherence to metrical rules, and a smooth, dynamic rhythm.
"""

_RUBRIC_LLM_JUDGE_BEAUTY_OF_FORM = """\
Evaluate the translation of the given Chinese classical poem into English.
Focus on whether the translation maintains consistency with the source poem's structure, including the alignment of line numbers and balanced phrasing.

1 point: Poor translation, disregards the poem's structure, and fails to convey its aesthetic qualities.
2 point: Some attempt to maintain structure but lack alignment and aesthetic consistency.
3 point: Basic structural elements are maintained but with noticeable imperfections in alignment and phrasing.
4 point: Good translation, with most structural elements preserved and minor issues in phrasing and alignment.
5 point: Excellent translation, accurately preserving the structure, alignment, and aesthetic qualities of the original poem.
"""

_RUBRIC_LLM_JUDGE_BEAUTY_OF_MEANING = """\
Evaluate the translation of Chinese classical poetry for the beauty of meaning, focusing on whether the translation effectively conveys the themes, emotions, and messages of the original. This includes the use of concise and precise language to create vivid imagery and a rich atmosphere.

1 point: Poor translation, fails to convey the depth and richness of the original poetry.
2 point: Basic translation with significant shortcomings in capturing themes, emotions, and messages.
3 point: Satisfactory translation, conveys basic themes and emotions but lacks refinement or depth.
4 point: Good translation, effectively captures most themes, emotions, and messages with minor imperfections.
5 point: Excellent translation, accurately conveys the depth, richness, and atmosphere of the original poetry with full thematic and emotional resonance.
"""


@run_spec_function("poetmt")
def get_poetmt_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.poetmt_scenario.PoetMTScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=768,
        temperature=0.0,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.poetmt_metric.PoetMTAutomaticMetric"),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_beauty_of_sound"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_beauty_of_form"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_beauty_of_meaning"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.poetmt_annotator.PoetMTJudgeAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 32,
                "metric_name": "llm_judge_beauty_of_sound",
                "rubric": _RUBRIC_LLM_JUDGE_BEAUTY_OF_SOUND,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.poetmt_annotator.PoetMTJudgeAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 32,
                "metric_name": "llm_judge_beauty_of_form",
                "rubric": _RUBRIC_LLM_JUDGE_BEAUTY_OF_FORM,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.poetmt_annotator.PoetMTJudgeAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 32,
                "metric_name": "llm_judge_beauty_of_meaning",
                "rubric": _RUBRIC_LLM_JUDGE_BEAUTY_OF_MEANING,
            },
        ),
    ]

    return RunSpec(
        name="poetmt",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "poetmt"],
        annotators=annotators,
    )
