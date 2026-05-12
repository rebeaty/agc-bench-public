# sdat fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is (English-only scope; document externally-recovered calibration constants; multilingual extension would lift to Tier 1)

## Paper / repo audited
- Paper: https://arxiv.org/abs/2505.09068 (Haase, Hanel & Pokutta, S-DAT, AAAI/ACM AIES 2025) - skim
- Repo: none provided -- OSF Study 2 rescoring bundle referenced in metric file as the source for calibration constants.

## Implementation audited
- scenarios/sdat_scenario.py - 100 instances with the verbatim English DAT prompt from Olson et al. 2021 (lines 32-40). Each instance gets a zero-width-space nonce suffix to defeat HELM caching across repeated trials. No references (open-ended).
- metrics/sdat_metric.py - SDATMetric loads `ibm-granite/granite-embedding-278m-multilingual` (line 21) via `metrics.embedder_factory` (so it can route through a Gemini embedding backend), CLS-pooled + L2-normalized, pairwise (1-cosine) mean x 100. Calibration: linear scale=2.979, bias=-13.890 (lines 28-29) recovered from OSF Study 2 bundle since paper does not publish coefficients. Piecewise-linear percentile against Study 2 anchors (N=8,498). Requires >=7 parsed items, caps at 10. Emits 8 metrics including `sdat_score`, `sdat_raw_score`, `sdat_percentile_estimate`.
- registry_metrics.yaml: 5+ metrics wired to `metrics.sdat_metric.SDATMetric`, all `model_based` with `metric_model: ibm-granite/granite-embedding-278m-multilingual`. Note: registry advertises Granite specifically, but the runtime embedder factory may substitute Gemini -- a registry-vs-runtime mismatch.
- registry_inference.yaml: `_use_defaults: true` (T=0.7, max=512, n=1) -- T=0.7 reasonable for divergent generation; max=512 fine for 10-word lists.

## Deviations found
- [MEDIUM] A. English-only: paper's S-DAT covers 11+ languages; implementation only exercises the English DAT prompt. Scoped intentionally per scenario docstring.
- [MEDIUM] C. Embedder backend drift: registry pins `ibm-granite/granite-embedding-278m-multilingual` but `metrics/embedder_factory` may route to Gemini by default (see metric file lines 56-83 comments). Calibration constants were fit to Granite outputs; substituting Gemini will produce different absolute scores.
- [LOW] C. Calibration coefficients (`_CALIBRATION_SCALE`, `_CALIBRATION_BIAS`) and percentile anchors externally recovered from OSF Study 2 bundle, not published in paper. Explicitly documented in code.
- [LOW] C. CLS pooling asserted as model-card config; would benefit from verification against SDAT reference implementation.
- [info] D. 100 trials per model is a HELM-side choice for stochastic LLM estimation; paper administers once per human participant. Defensible.

## Notes
Method matches the paper end-to-end on English. The most material risk is the embedder factory possibly serving Gemini embeddings while the calibration constants were fit on Granite -- if `ABC_EMBEDDING_BACKEND` defaults to Gemini, scores will be miscalibrated even though they look reasonable. Recommend: (a) document the calibration provenance in the registry; (b) hard-pin the embedder backend to Granite for SDAT scoring (or refit constants to Gemini if Gemini is the production backend); (c) treat multilingual as a follow-up scenario; (d) keep English-only as the primary result.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is (English-only scope; multilingual extension would lift to Tier 1)
- Now:   Tier 2, high, keep_as_is
- Delta: confirmed
