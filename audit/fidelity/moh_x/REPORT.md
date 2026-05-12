# moh_x fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** patch_with_f1_score (add `f1_score` MetricSpec to run_spec; consider positive-class F1 to mirror Gao 2018)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/moh_x_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://aclanthology.org/S16-2003/ (Mohammad, Shutova & Turney, *SEM 2016, "Metaphor as a Medium for Emotion") - skim
- Repo: https://github.com/gao-g/metaphor-in-context (Gao et al. 2018 EMNLP, MOH-X preprocessing) - skim

## Implementation audited
- scenarios/moh_x_scenario.py - downloads MOH-X zip from Google Drive (file ID `1-v_sUlupDrKq8ERlnh5RwdI7Pyk-6cxn`), reads `MOH-X_formatted_svo_cleaned.csv`. Synthesized zero-shot binary template (lines 70-74): `Is the word "{verb}" used metaphorically in the following sentence?\n\nSentence: {sentence}\n\nAnswer (Yes or No):`. Two references with CORRECT_TAG on the gold Yes/No. 647 instances all in TEST_SPLIT.
- metrics/moh_x_metric.py - DOES NOT EXIST (`has_metric_file: false`).
- registry_metrics.yaml: lists `exact_match` + `f1_score`, both `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`.
- registry_inference.yaml: `_use_defaults: true` (T=0.7, max=512, n=1) -- T=0.7 too high for binary classification.

## Deviations found
- [MEDIUM] C. Metric scoring fidelity: registry advertises `f1_score` but the run_spec (per prior audit) only wires `exact_match`. F1 won't be reported even though it's the more comparable metric.
- [MEDIUM] C. Token-level F1 on "Yes"/"No" via `BasicGenerationMetric` does NOT match Gao 2018's positive-class (metaphor=1) F1 convention. Even if `f1_score` is added, it will be a different statistic from the paper's primary number.
- [LOW] B. Prompt synthesized (no LLM prompt in source papers). Format is reasonable but unverifiable against any paper template.
- [LOW] A. Paper uses 10-fold CV; LLM zero-shot collapses to single-pass over all 647 -- defensible.
- [LOW] D. T=0.7 default is high for a binary classification task; T=0.0-0.3 would better suit deterministic Yes/No.
- [info] A. 647 pairs match Gao 2018 MOH-X size.

## Notes
Scenario is correct for what it claims. Two patches: (1) add `f1_score` MetricSpec to `run_specs/moh_x_run_specs.py` to satisfy the registry; (2) consider implementing a positive-class (metaphor=1) F1 wrapper so the primary number is paper-comparable. Drop generation T to ~0.0-0.3 for binary task. The Google Drive download path is a reproducibility risk if the file ID changes.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, patch_with_f1_score
- Now:   Tier 2, high, patch_with_f1_score
- Delta: confirmed
