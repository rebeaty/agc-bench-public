"""HELM Run Specs for cpers."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from llm_judge._judge_override import resolve_judge


@run_spec_function("cpers")
def get_cpers_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.cpers_scenario.CPersScenario",
        args={},
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
        max_tokens=256,
        temperature=1.0,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.cpers_metric.CPersMetric", args={}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.cpers_annotator.CPersAnnotator",
            args={
                "judge_model_name": resolve_judge("anthropic/claude-3.7-sonnet", bench_var="CPERS_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 700,
            },
        ),
    ]

    return RunSpec(
        name="cpers",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "cpers"],
        annotators=annotators,
    )
