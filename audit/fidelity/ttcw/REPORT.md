# ttcw fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** medium
**Recommendation:** keep_as_is (fix paper pointer in registry; document N=36 open-text scope reduction)

## Paper / repo audited
- Paper (registry): https://arxiv.org/abs/2510.05135 — this is a **2025 follow-on** ("curiosity-driven" approach using the TTCW benchmark), not the originating work. The actual TTCW paper is **Chakrabarty et al. 2024, "Art or Artifice? Large Language Models and the False Promise of Creativity"** (CHI 2024 / arXiv:2309.14556). Registry pointer is wrong. (note: "skim")
- Repo: https://huggingface.co/datasets/Salesforce/ttcw_creativity_eval (registry) and https://github.com/salesforce/creativity_eval (actual fetch URL in scenario lines 14-18). Files: prompts/with_background.txt, ttcw_short_stories.json, ttcw_all_tests.json, ttcw_annotations.json. (note: "read" via scenario)

## Implementation audited
- scenarios/ttcw_scenario.py — fetches four assets from `salesforce/creativity_eval/Art_or_Artifice/`. Filters stories: drops any whose `content` starts with "http" (URL-only entries); keeps the **36 open-text stories** out of 48 total (acknowledged in docstring lines 39-43). Builds 14 TTCW tests × 36 stories = 504 (story, test) instances. Each prompt = `with_background.txt` template with [STORY], [BACKGROUND] (from test full_prompt), [QUESTION] substitutions. References empty; gold is majority of expert binary verdicts in `extra_data["majority_label"]`.
- metrics/ttcw_metric.py — `TTCWMetric.evaluate_generation` regex-extracts "Yes"/"No" from prediction. Returns `valid_binary_response`, `majority_accuracy` (1 if pred==majority of expert votes), `expert_vote_agreement` (fraction of expert votes that match prediction).
- registry_metrics.yaml (lines 2924-2937): three formula_based metrics on `TTCWMetric`. All `in_helm: false`.
- registry_inference.yaml (lines 686-688): `_use_defaults: true`. (For Yes/No tasks, low-T would be appropriate.)

## Deviations found
- [MEDIUM] Paper pointer error: `source_paper` field points to arXiv:2510.05135 (a 2025 follow-on) instead of the originating Chakrabarty et al. 2024 paper (arXiv:2309.14556 / CHI 2024 "Art or Artifice?"). Should be corrected in `registry_master.yaml`.
- [MEDIUM] A. Dataset source: 36/48 stories due to repo only releasing open text for 36; 12 stories are linked URLs. Documented in scenario header. Implies our results aren't directly comparable to the paper's full 48-story analysis.
- [LOW] B. Prompt fidelity: Prompt template loaded verbatim from upstream `prompts/with_background.txt`; tests pulled from `ttcw_all_tests.json`. As verbatim as possible.
- [LOW] C. Metric fidelity: majority_accuracy and expert_vote_agreement are paper-aligned aggregations (the paper reports per-test agreement against expert majority; this matches). Plus parse-rate diagnostic.
- [info] D. Generation config: defaults; for Yes/No, low T (e.g., 0.0) would be more appropriate.

## Notes
Implementation is structurally faithful — verbatim prompt + tests + annotations from upstream, sensible Yes/No parser, paper-aligned majority/agreement metrics. Two cleanup items: (1) fix the `source_paper` registry entry to the originating Chakrabarty et al. 2024 work, and (2) consider tightening inference T to 0 for Yes/No determinism. The 36-of-48 open-text scope reduction is acknowledged and inherent to the data release.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, medium, keep_as_is (fix paper pointer in registry; document scope reduction)
- Now:   Tier 2, medium, keep_as_is
- Delta: confirmed
