# AnaloBench Fidelity Audit

- **Tier:** 1 (High fidelity)
- **Confidence:** High
- **Recommendation:** Keep as-is. Optionally add T2/T3/Sx variants; confirm `T1S1-Subset` matches the paper's small-bank (4-option) condition.

## Paper / Repo Audited
- Paper: Ye et al., "AnaloBench: Benchmarking the Identification of Abstract and Long-context Analogies" (arXiv:2402.12370, EMNLP 2024).
- Repo: https://github.com/jhu-clsp/AnaloBench (`code/t1.py`).
- Task audited: T1 multiple-choice analogy selection (4 options, single-sentence).

## Implementation Audited
- Scenario: `scenarios/analobench_scenario.py`
- Metric: `metrics/analobench_t1_metric.py`
- Registry: `data/registry/registry_metrics.yaml`, `registry_inference.yaml` (defaults).

## Prompt Fidelity
Verbatim port of upstream `code/t1.py` template ("Which of the following is the most analogous story...", "Note: Only generate the index without any additional text.", Target Story / Options / Answer).

## Metric Fidelity
Mirrors upstream regex evaluator: option extracted with `(?:\:\s)?([A-D])(?:\.|\s|$)`; correct=1.0, wrong=0.0, unparseable=0.25 (paper's "irrelevant" partial-credit). Adds `parsed_option_rate` and `irrelevant_rate` diagnostics. Matches paper Section 3 / Appendix D.

## Instance Count
Loads `jhu-clsp/AnaloBench` config `T1S1-Subset`, split `train` — the paper's small-scale 4-option subset. Subsampling handled downstream.

## Judge / Reference
No LLM judge; gold = dataset `Label`. References attached for all four options with `CORRECT_TAG` on gold letter.

## Deviations
- Only T1 (small-bank, single-sentence) implemented; paper also covers T2/T3 multi-sentence and Sx large-bank retrieval (MAP/P@K/MRR).
- Inference config uses HELM defaults (paper does not specify decoding params).

## Notes
Prompt and scoring faithfully reproduce upstream code. No blocking issues.
