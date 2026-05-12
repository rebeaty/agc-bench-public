# hummus fidelity audit

**Tier:** 2
**Confidence:** high
**Recommendation:** patch_with_task_specific_metrics

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/hummus_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.classification_metrics.MultipleChoiceClassificationMetric`, `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric`
> - **AnnotatorSpec(s):** `llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: Humorous Multimodal Metaphor Use (arXiv:2504.02983)

## Implementation audited
- scenarios/hummus_scenario.py — 4 subsets: classification, naming, caption_highlight, explanation. Uses prompt IDs documented in repo (`CLAS02`, `NAME04`, `CAPT02`, `EXPL06` from `model_evaluation/prompts.py`).
- No `metrics/hummus_metric.py` file.
- registry_metrics.yaml: only `classification_macro_f1`, `classification_micro_f1` (HELM `MultipleChoiceClassificationMetric`) plus generic `llm_judge_quality` (gpt-4o, `judge_prompt: null`).
- 940 + 589 + 568 + 628 instances across the four subsets, consistent with paper.

## Deviations found
- [HIGH] **Modality reduction**: paper is multimodal (image + caption); scenario substitutes textual `image_description` from CapCon corpus — explicitly flagged in docstring as text-only ablation.
- [HIGH] **Task-specific metrics missing**: paper specifies macro/micro-F1 (CLAS), LaBSE cosine (NAME), Jaccard (CAPT), ROUGE (EXPL). Registry has only F1 metrics — applying F1 to NAME/CAPT/EXPL (open-ended free-text references) is misapplied.
- [MEDIUM] Visual grounding/IoU tasks intentionally dropped, consistent with text-only scope.
- [LOW] LLM judge entry has null prompt — non-functional placeholder.

## Notes
Prompts faithful (verbatim repo IDs). Instance counts and filtering (`met_class in {Yes, WIDLII}`) match paper. Tier 2 because metric fidelity is incomplete.

**Recommendation:**
1. Add task-specific metrics: LaBSE for `naming`, ROUGE-L for `explanation`, Jaccard for `caption_highlight`.
2. Restrict F1 to `classification` subset only.
3. Either author real judge prompt or drop placeholder.
4. Document text-only ablation as deliberate modality reduction.
