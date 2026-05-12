# artinsight fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** medium-high
**Recommendation:** patch_with_judge_rubric (populate `judge_prompt` for the six artinsight_* metrics from `description_scorer.py`; optionally add `metrics/artinsight_metric.py` wrapper)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/artinsight_run_specs.py`:
>
> - **MetricSpec(s):** `llm_judge.artinsight_metric.ArtInsightMetric`
> - **AnnotatorSpec(s):** `llm_judge.artinsight_annotator.ArtInsightAnnotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2502.19263 ("ArtInsight," IUI 2025) — abstract only (PDF too large for WebFetch); rubric details obtained from repo. (note: "skim")
- Repo: https://github.com/makeabilitylab/ArtInsight (Artwork-Description-Scoring/) — README.md is 404; description_scorer.py and folder listing accessible. (note: "read")

## Implementation audited
- scenarios/artinsight_scenario.py — pulls all 30 artwork JPEGs from repo `Artwork-Description-Scoring/images/`; uses **only the descriptive prompt** (the paper/repo describe three prompt variants: descriptive, creative, 3-questions). Open-ended generation, no gold reference. Prompt text (lines 53-85) is verbatim from the repo's descriptive prompt.
- metrics/artinsight_metric.py — **does not exist** (`has_metric_file: false`).
- registry_metrics.yaml (lines 194-237): six llm_judge metrics (`artinsight_score`, `_presumptive_score`, `_reductive_score`, `_detail_score`, `_elements_score`, `_misc_deduction`); all with `judge_model_name: openai/gpt-4o`, `judge_prompt: null`, T=0.0, max_new_tokens=1024.
- registry_inference.yaml (lines 62-64): `_use_defaults: true`; "not specified in paper or repo; using benchmark defaults."

## Deviations found
- [HIGH] C. Metric/scoring fidelity: All six judge metrics have `judge_prompt: null` and there is no `metrics/artinsight_metric.py`. Without the rubric prompt wired in, the judge cannot produce paper-comparable subscores. Maintainer needs to copy the per-criterion prompts and 0-4 anchor definitions from `Artwork-Description-Scoring/description_scorer.py`.
- [MEDIUM] B. Prompt fidelity: Only the descriptive variant is implemented. Paper evaluates three prompt families; we cover one. Defensible as a scoped subset, but should be documented.
- [LOW] A. Dataset source: All 30 images covered, matches repo. info.
- [LOW] C. Judge model: Paper uses GPT-4o; registry preserves it. Acceptable.
- [info] D. Generation config: Paper does not specify; benchmark defaults appropriate for open-ended description.

## Notes
The scenario faithfully loads the 30 artworks and the descriptive-prompt text matches the repo. The blocking issue is purely on the metric side: six `artinsight_*` judge slots are registered but unfilled. Fix is mechanical — paste the per-criterion prompt + anchor scale from `description_scorer.py` into a new `metrics/artinsight_metric.py` and set `judge_prompt` in `registry_metrics.yaml`. Until then, the bench cannot produce numbers comparable to the IUI 2025 results.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, patch_with_judge_rubric (implement `metrics/artinsight_metric.py` + populate `judge_prompt` from repo `annotator_notes.md`)
- Now:   Tier 2, medium-high, patch_with_judge_rubric (source rubric is `description_scorer.py`, not `annotator_notes.md` which is 404)
- Delta: confirmed
