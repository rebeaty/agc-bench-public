# meta4xnli fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is (optional: add IV/OOV breakdown for detection)

## Paper / repo audited
- Paper: Sanchez-Bayona & Agerri (2024), arXiv:2404.07053
- Dataset: HiTZ/meta4xnli

## Implementation audited
- scenarios/meta4xnli_scenario.py — subsets: `interpretation_en`, `interpretation_es`, `interpretation_en_cot`, `interpretation_es_cot`, `detection_en`.
- Interpretation prompts: verbatim from paper Appendix Table 29 ("Say which is the inference relationship between these two sentences. Please, answer only with one word between 'entailment', 'neutral' or 'contradiction'."). CoT prompt also verbatim with three demonstrations.
- Detection prompt: HELM-specific adaptation (paper used token-level sequence-labeling heads, not generative output). Numbered token list with binary labels.
- metrics: `MultipleChoiceClassificationMetric` for interpretation; custom `Meta4XNLIDetectionMetric` for detection (token-level F1, accuracy, `valid_label_sequence_rate`).
- 580 per language for interpretation; 3,630 for `det_en_finetune` test.

## Deviations found
- [MEDIUM] **Detection prompt is HELM adaptation**: paper used classifier head, not generative output. Documented but not paper's setup.
- [MEDIUM] **IV/OOV breakdown missing for detection**: paper reports token F1 with In-Vocab vs Out-Of-Vocab split.
- [LOW] Adapter: ADAPT_MULTIPLE_CHOICE_JOINT for interpretation (T=0.0, max_tokens=8, stop `\n`); ADAPT_GENERATION for detection (max_tokens=512, T=0.0).

## Notes
HIGH fidelity for interpretation subsets (verbatim prompts, correct metric, native splits). MEDIUM fidelity for `detection_en` (HELM-adapted prompt, missing IV/OOV breakdown). No LLM judge needed; gold references are dataset labels.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is
- Now:   Tier 2, high, keep_as_is
- Delta: confirmed (verified scenarios/meta4xnli_scenario.py lines 81-99 verbatim CoT prompt with three exemplars matching paper Appendix Table 29; lines 142-146 zero-shot prompt verbatim; detection prompt at lines 183-190 is HELM adaptation; registry exposes classification_macro_f1, classification_micro_f1 + custom Meta4XNLIDetectionMetric for token-level)
