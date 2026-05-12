# story_quality fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** medium
**Recommendation:** keep_as_is_with_caveat (correlation-metric bug now fixed; remaining deviations are task re-cast and reconstructed prompt — both documentable)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/story_quality_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.correlation_metric.CorrelationMetric`, `metrics.correlation_metric.CorrelationMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2206.03021 ("Plot Writing From Pre-Trained Language Models," INLG 2022) — note: skim (abstract; paper validates ScratchPlot via human pairwise win-rates and expert plot-element scoring, not point-regression)
- Repo: https://github.com/YipingNUS/scratchplot-story-generation — note: skim (`crowdsource/` Toloka annotations, 3 raters per story; TSVs fetched directly)

## Implementation audited
- scenarios/story_quality_scenario.py — Three subsets: `interestingness` (default), `coherence`, `naturalness`. Loads each dimension's `crowdsource/<dim>/full-annotation-result-new.tsv` via raw GitHub. Aggregates 3 ratings per (plan_id, system) story to a rounded mean (1–5). Builds prompt (lines 56–63): "Read the following story and rate its {dimension} on a scale of 1 to 5...Output your rating as a single number from 1 to 5." Five `Reference` objects (texts "1".."5"), only the rounded-mean one tagged `CORRECT_TAG`.
- metrics/correlation_metric.py — `CorrelationMetric` parses model completion as float `pred_value`, then iterates `references` to find the `CORRECT_TAG`-tagged one as `true_value` (lines 44–53, comment notes the 2026-04-25 bug fix). Emits per-instance `*_pred`, `*_true`, and signed-error stats; aggregate correlation must be computed downstream from collected pairs.
- registry_metrics.yaml (lines 2628–2637): two metrics — `spearman_correlation`, `pearson_correlation` (both `metrics.correlation_metric.CorrelationMetric`).
- registry_inference.yaml (lines 649–651): `_use_defaults: true`. Run-spec layer reportedly uses T=0.7, max_tokens=512 — too loose for a 1–5 rating output.

## Deviations found
- [MEDIUM] B./C. Task re-cast: Paper validates ScratchPlot via crowd-sourced pairwise win-rates and expert plot-element scoring. AGC re-casts the human-annotation TSVs as a point-rating regression task (model emits an integer 1–5, correlate with the 3-rater mean). This is a defensible LLM-as-judge calibration probe, but it is not the paper's evaluation protocol.
- [MEDIUM] B. Prompt fidelity: AGC reconstructs a generic rating prompt because the paper does not publish the verbatim Toloka annotator instructions. Reasonable but not a verbatim match.
- [LOW] D. Generation config: `_use_defaults: true` likely yields T=0.7; for an integer-rating task T=0.0 with max_tokens=4 would be more appropriate. Currently noisy.
- [info] C. Correlation metric bug fix: The previously HIGH bug — reading `references[0]` (always "1") as the true value — was fixed on 2026-04-25 to scan for the `CORRECT_TAG` reference. Confirmed in `metrics/correlation_metric.py` lines 44–53 with explicit comment.
- [info] A. ~50 unique stories per dimension after aggregation, matches paper.

## Notes
The bug that motivated the prior audit's HIGH severity has been fixed in `metrics/correlation_metric.py`; the dataset/scenario shape is otherwise reasonable. Remaining deviations are conceptual: this slice tests an LLM's ability to predict crowd ratings, not the original paper's pairwise win-rate construct. Two safe cleanups: (1) pin `temperature: 0.0`, `max_tokens: 4` in registry_inference.yaml so the rating output is deterministic; (2) document in the paper writeup that this slice is a reuse of the ScratchPlot annotations as a story-quality calibration probe rather than a reproduction of the paper's primary.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, medium, patch_with_correlation_metric_fix (real bug)
- Now:   Tier 2, medium, keep_as_is_with_caveat
- Delta: lifted (the HIGH correlation-metric bug has been fixed; remaining issues are MEDIUM construct-shift items already documented)
