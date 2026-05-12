"""HELM Run Specs for creatset."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from llm_judge._judge_override import resolve_judge


@run_spec_function("creatset")
def get_creatset_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.creatset_scenario.CreataSetScenario",
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
            class_name="metrics.creatset_pairwise_metric.CreatSetPairwiseMetric",
            args={},
        ),
    ]

    return RunSpec(
        name="creatset",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "creatset"],
        annotators=[
            AnnotatorSpec(
                class_name="llm_judge.creatset_pairwise_annotator.CreatSetPairwiseAnnotator",
                args={
                    "judge_model_name": resolve_judge("openai/gpt-4o", bench_var="CREATSET_JUDGE_MODEL_OVERRIDE"),
                    "judge_temperature": 0.0,
                    "judge_max_new_tokens": 512,
                },
            )
        ],
    )
