# irfl fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is (optional: document generative-track omission)

## Paper / repo audited
- Paper: Yosef et al. 2023, "IRFL: Image Recognition of Figurative Language" (arXiv:2303.15445)
- Dataset: lampent/IRFL (HF)

## Implementation audited
- scenarios/irfl_scenario.py — phrase + optional definition (idiom) or context query (metaphor/simile), 4 labeled images A–D, with deterministic per-instance shuffling (`Random(f"{config}:{idx}:{phrase}")`) to prevent position leakage. Single-letter answer requested.
- registry: `exact_match` (BasicGenerationMetric) plus `classification_macro_f1` / `classification_micro_f1` (MultipleChoiceClassificationMetric).
- Subsets supported: idiom-detection (200), metaphor-detection (333), simile-detection (277), open-simile-detection (277). Default = idiom-detection-task.

## Deviations found
- [INFO] Paper also evaluates generative (text-to-image + human raters) track; HELM port covers only discriminative track. Acceptable scoping — generative track needs human judges, out of scope for automated benchmarking.
- [INFO] Inference: `_use_defaults: true`. Paper does not specify; low-temp MCQ defaults appropriate.

## Notes
Pure reference-based; gold image UUID from dataset `answer` field, mapped to shuffled letter. `has_reference_target: true`. Exact-letter match is functionally equivalent to accuracy; F1 metrics are reasonable supplements. Tier 1.

Optional: document generative-track omission in scenario docstring. Otherwise no changes required.
