# arn fidelity audit (re-audit 2026-05-04)

**Tier:** 1
**Confidence:** medium-high
**Recommendation:** keep_as_is (optional patch: register arn_metric.py in registry_metrics.yaml)

## Paper / repo audited
- Paper: https://arxiv.org/abs/2310.00996 (Sourati et al., TACL 2024) — "skim" (abstract via WebFetch; scenario docstring quotes Appendix E.2 verbatim)
- Repo: none — paper provides only a Google Drive data link

## Implementation audited
- scenarios/arn_scenario.py — Downloads xlsx via gdown from Drive folder `1itOPXtorFEgweQCd71m2bIRwWAUHcXuf`. Loads 1,095 triples; supports 5 subsets (all, near_high, near_low, far_high, far_low) matching the four partitions reported in the paper. Prompt at lines 67–76 reproduces Appendix E.2 (GPT/LLaMA format) with proper escaping of literal `{{` braces. Two references "1" and "2" with CORRECT_TAG on the gold.
- metrics/arn_metric.py — Custom parser `_extract_decision` (lines 14–26) extracts "1"/"2" via four regex fallbacks (narrative_X, answer X, leading digit, bare digit). Reports `arn_parse_rate` and `arn_accuracy`.
- registry_metrics.yaml (line 187): registers only `exact_match` (HELM `compute_reference_metrics`) — does NOT register the custom `arn_metric.py`.
- registry_inference.yaml (line 58): `_use_defaults: true`.

## Deviations found
- [LOW] C. Metric/scoring fidelity: registry registers `exact_match` not the bespoke `arn_metric.py`. Paper's prompt asks for `{narrative_x, because ...}` answer, so raw exact_match against "1"/"2" will under-score correct responses. The parser exists locally; just needs registration.
- [LOW] D. Generation configuration: registry uses benchmark defaults; paper's exact T/max_tokens not specified for ARN MCQ; defaults are reasonable.
- [info] A. Dataset/instance source: full 1,095 triples; matches paper.
- [info] B. Prompt fidelity: verbatim from Appendix E.2.

## Notes
Implementation is faithful to paper. Only real concern is that the registry registers `exact_match` rather than the local `arn_metric.py` parser. Fix is one yaml edit: add `arn_accuracy` (helm_class pointing to `metrics.arn_metric.ARNMetric`) under the arn block in registry_metrics.yaml, then keep `exact_match` as a fallback. Tier 1 since the dataset/prompt are correct and the parser already exists.

## Compared to prior audit (2026-04-25)
- Prior: Tier ?, unknown, no audit conducted
- Now:   Tier 1, medium-high, keep_as_is
- Delta: newly classified
