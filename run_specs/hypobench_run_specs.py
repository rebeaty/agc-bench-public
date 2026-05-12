"""HELM run specs for HypoBench."""

from __future__ import annotations

import os

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


_TASK = os.environ.get("HYPOBENCH_TASK", "all_real").strip() or "all_real"
_NUM_HYPOTHESES = int(os.environ.get("HYPOBENCH_NUM_HYPOTHESES", "10"))


@run_spec_function("hypobench")
def get_hypobench_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.hypobench_scenario.HypoBenchScenario",
        args={"task": _TASK, "num_hypotheses": _NUM_HYPOTHESES},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=1024,
        temperature=0.0,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.hypobench_inference_metric.HypoBenchInferenceMetric"),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.hypobench_inference_annotator.HypoBenchInferenceAnnotator",
            args={
                "inference_temperature": 0.0,
                "inference_max_new_tokens": 512,
            },
        ),
    ]

    return RunSpec(
        name="hypobench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "hypobench"],
        annotators=annotators,
    )
