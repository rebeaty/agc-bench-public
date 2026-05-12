# pun_eval fidelity audit

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_caveats (or implement TPR/TNR/Kappa for full coverage)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/pun_eval_run_specs.py`:
>
> - **MetricSpec(s):** `llm_judge.pun_eval_metric.PunEvalMetric`
> - **AnnotatorSpec(s):** `llm_judge.pun_eval_annotator.PunEvalPunDetectionAnnotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: Xu et al., "A good pun is its own reword" (arXiv:2404.13599, EMNLP 2024)
- Repo: https://github.com/Zhijun-Xu/PunEval (Notebooks 2, 5, 6)

## Implementation audited
- scenarios/pun_eval_scenario.py — `generation` task uses Notebook 5 Method 1 verbatim: `<*Definition*>` + `<*Instruction*>` blocks, JSON constraint `{"Sentence": "XXX"}`, keyword + two senses. `explanation` task reuses Notebook 2 CoT recognition prompt (`{"Reason","Choice"}`) — documented adaptation since paper has no standalone explanation prompt.
- llm_judge/pun_eval_annotator.py + pun_eval_metric.py — Notebook 6 binary pun detection (Definition + Instruction → `{"Choice": ...}`); aggregated as `pun_detection_rate`. Two diagnostic rates added.
- 1,457 fully annotated entries (1,443 hom + 1,146 het = 2,589 minus SemEval-only entries lacking ExPun fields). Capped to 200.

## Deviations found
- [MEDIUM] **Coverage partial**: only generation→pun-detection ported. Recognition/explanation metrics from paper (TPR/TNR, Cohen's Kappa, human funniness ratings) not implemented. Scenario docstring mentions Ambiguity/Distinctiveness/Surprise/Unusualness but these aren't wired in.
- [LOW] 200-instance cap limits comparability with paper-scale runs.

## Notes
Faithful generation pipeline and Notebook 6 judge. Judge: `openai/gpt-4o`, T=0.0, max 256 tokens — matches paper's GPT-4-as-judge convention. References (`human_text`, `human_explanation`) attached with `CORRECT_TAG` for soft BLEU/ROUGE; not consumed by configured metric.
