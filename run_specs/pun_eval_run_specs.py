"""HELM Run Specs for pun_eval."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from llm_judge._judge_override import resolve_judge


@run_spec_function("pun_eval")
def get_pun_eval_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.pun_eval_scenario.PunEvalScenario",
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
        max_tokens=300,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="llm_judge.pun_eval_metric.PunEvalMetric"),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.pun_eval_annotator.PunEvalPunDetectionAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4o", bench_var="PUN_EVAL_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
            },
        ),
    ]

    return RunSpec(
        name="pun_eval",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "pun_eval"],
        annotators=annotators,
    )
