"""HELM Run Specs for fann_or_flop."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from llm_judge._judge_override import resolve_judge


@run_spec_function("fann_or_flop")
def get_fann_or_flop_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.fann_or_flop_scenario.FannOrFlopScenario",
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
        max_tokens=4096,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric",
            args={"names": ["bleu_4"]},
        ),
        MetricSpec(class_name="metrics.fann_or_flop_metric.FannOrFlopMetric", args={}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.fann_or_flop_annotator.FannOrFlopAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4o", bench_var="FANN_OR_FLOP_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
            },
        ),
    ]

    return RunSpec(
        name="fann_or_flop",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "fann_or_flop"],
        annotators=annotators,
    )
