# Audit: future_ideas


> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/future_ideas_run_specs.py`:
>
> - **MetricSpec(s):** `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric`, `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric`, `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric`
> - **ScenarioSpec args:** `domain=domain`
> - **AnnotatorSpec(s):** `llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator`, `llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator`, `llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

**Paper:** Can LLMs Unlock Novel Scientific Research Ideas? (arXiv:2409.06185)
**Repo:** sandeep82945/Future-Idea-Generation
**Scenario:** scenarios/future_ideas_scenario.py
**Metric file:** none (metrics/future_ideas_metric.py absent)

## Prompt fidelity
Scenario reproduces the paper's Section 4.1 / code/run2.py prompt verbatim (research-scientist persona, three-step instruction, "distinct from related papers", bullet-point ending). Adaptation: paper text truncated to 2000 leading words to fit context, vs. upstream chunking of long papers. Documented and reasonable. Faithful with minor adaptation.

## Instance count
Loads all 458 papers across 5 domains (chemistry 72, computer 88, economics 61, medical 110, physics 127) from upstream xlsx files. Uses full_text_WF as input and Future_work as gold reference, correctly avoiding the leakage-prone full_text column. Faithful.

## Metric / judge fidelity
Registry specifies three GPT-4 LLM-judge metrics: novelty, relevance, feasibility. Major gaps:
- judge_prompt is null for all three; rubrics unspecified.
- No metric class implemented; in_helm: false and helm_class missing.
- Paper's signature metrics — IAScore and embedding-based Idea Distinctness Index — explicitly not implemented (only noted in docstring).
- Paper's primary evaluation is human expert ratings; the LLM-judge triple is a local substitution that only partially aligns with the paper's rating axes.

## Tier assessment
Tier 2 (partial fidelity). Prompt and instance construction are faithful; metric layer is incomplete.
Confidence: High.
Recommendation: Before running, (1) author and pin the three judge rubrics, (2) implement a metric class or wire to a generic LLM-judge metric, (3) add IAScore and Idea Distinctness Index to recover the paper's distinctive signal, or explicitly scope this port as judge-only.
