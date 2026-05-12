"""HELM Run Specs for mops."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from llm_judge._judge_override import resolve_judge


@run_spec_function("mops")
def get_mops_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.mops_scenario.MoPSPremiseScenario",
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
        max_tokens=256,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="llm_judge.mops_metric.MoPSMetric", args={}),
        MetricSpec(
            class_name="metrics.mops_diversity_metric.MoPSDiversityMetric",
            args={
                "model_name": "all-MiniLM-L6-v2",
                "tsne_random_state": 42,
                "perplexity": 50,
                "num_bins": 10,
            },
        ),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.mops_annotator.MoPSAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4-turbo", bench_var="MOPS_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "max_retries": 3,
            },
        ),
    ]

    return RunSpec(
        name="mops",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "mops"],
        annotators=annotators,
    )
