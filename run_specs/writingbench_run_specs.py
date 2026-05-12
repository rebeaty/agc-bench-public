"""HELM Run Specs for writingbench."""

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


_JUDGE_MODEL_NAME = resolve_judge("google/gemini-2.5-flash-lite", bench_var="WRITINGBENCH_JUDGE_MODEL_OVERRIDE")


@run_spec_function("writingbench")
def get_writingbench_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.writingbench_scenario.WritingBenchScenario",
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
        max_tokens=16000,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="llm_judge.writingbench_metric.WritingBenchMetric", args={}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.writingbench_annotator.WritingBenchAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL_NAME,
                "judge_temperature": 1.0,
                "judge_top_p": 0.95,
                "judge_max_new_tokens": 2048,
            },
        ),
    ]

    return RunSpec(
        name="writingbench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "writingbench"],
        annotators=annotators,
    )
