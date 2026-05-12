"""HELM Run Specs for tinyfabulist."""

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


_SYSTEM_PROMPT = """\
You are a world-class creative assistant that generates captivating and morally-driven fables based on structured inputs.
Each fable must be:
  - Imaginative and coherent.
  - Appropriate for a wide audience, including young readers.
  - Structured around a classic fable format (character, setting, conflict, resolution, and moral).

Age groups are defined as:
  - A: 3 years or under
  - B: 4-7 years
  - C: 8-11 years
  - D: 12-15 years
  - E: 16 years or above
"""


@run_spec_function("tinyfabulist")
def get_tinyfabulist_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.tinyfabulist_scenario.TinyFabulistScenario",
        args={"source": "benchmark"},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions=_SYSTEM_PROMPT,
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
        MetricSpec(class_name="llm_judge.tinyfabulist_metric.TinyFabulistMetric", args={}),
        MetricSpec(class_name="metrics.tinyfabulist_corpus_metric.TinyFabulistCorpusMetric", args={}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.tinyfabulist_annotator.TinyFabulistAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/o3-mini-2025-01-31", bench_var="TINYFABULIST_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 350,
            },
        ),
    ]

    return RunSpec(
        name="tinyfabulist",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "tinyfabulist"],
        annotators=annotators,
    )
