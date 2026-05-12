"""HELM Run Specs for pron_vs_prompt."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from llm_judge._judge_override import resolve_judge


@run_spec_function("pron_vs_prompt")
def get_pron_vs_prompt_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.pron_vs_prompt_scenario.PronVsPromptScenario",
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
        max_tokens=1400,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="metrics.pron_vs_prompt_metric.PronVsPromptMetric", args={}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.pron_vs_prompt_annotator.PronVsPromptAnnotator",
            args={
                "judge_model_name": resolve_judge("openai/gpt-4o", bench_var="PRON_VS_PROMPT_JUDGE_MODEL_OVERRIDE"),
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
            },
        ),
    ]

    return RunSpec(
        name="pron_vs_prompt",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "pron_vs_prompt"],
        annotators=annotators,
    )
