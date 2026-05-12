"""HELM Run Specs for banner_request_400."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION_MULTIMODAL,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec

@run_spec_function("banner_request_400")
def get_banner_request_400_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.banner_request_400_scenario.BannerRequest400Scenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION_MULTIMODAL,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,  # ASSUMPTION: zero-shot, no TRAIN_SPLIT seen
        num_outputs=1,
        max_tokens=1024,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="llm_judge.banner_request_400_metric.BannerRequest400Metric", args={}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.banner_request_400_annotator.BannerRequest400Annotator",
            args={
                "judge_model_name": "openai/gpt-4o",
                "judge_temperature": 0.3,
                "judge_max_new_tokens": 512,
            },
        ),
    ]

    return RunSpec(
        name="banner_request_400",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "banner_request_400"],
        annotators=annotators,
    )
