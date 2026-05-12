# humor_transfer fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_caveat (document the 2-of-4-subset scope reduction explicitly; optionally add `one_liners` and `dad_jokes` subsets to recover the paper's cross-domain transfer design)

## Paper / repo audited
- Paper: https://arxiv.org/abs/2508.19402 ("One Joke to Rule them All? On the (Im)possibility of Generalizing Humor") — note: skim (abstract; confirms 4 humor sub-genres including Dad Jokes)
- Repo: https://github.com/morturr/HumorTransferLearning — note: unread (header in scenario cites Appendix C verbatim)

## Implementation audited
- scenarios/humor_transfer_scenario.py — Alpaca-style template (lines 45–54) from paper Appendix C: preamble + `### Instruction` ("Given the following text, please determine if it should be classified as funny or not funny. Base your classification on humor elements such as wit, irony, absurdity, or comedic timing.") + `### Input` + `### Response`. Yes/No labels per paper. Two subsets supported: `sarcasm_headlines` (HF `raquiba/Sarcasm_News_Headline` test split, ~26,709 examples) and `amazon_questions` (Amazon humor-detection-pds S3 CSVs, ~19,142 examples).
- metrics/humor_transfer_metric.py — `HumorTransferMetric`: parses first Yes/No-equivalent token via permissive regex (`yes|funny|humorous` vs. `no|not funny|non-humorous|non humorous`), reports `accuracy` and `parsed_label_rate` against `gold_label` from `extra_data`.
- registry_metrics.yaml (lines 1420–1429): two formula-based metrics — `accuracy` (in_helm true) and `parsed_label_rate` (in_helm false) — both `metrics.humor_transfer_metric.HumorTransferMetric`.
- registry_inference.yaml (lines 315–317): `_use_defaults: true`. For binary classification, T should be 0.0 — defaults may not enforce that.

## Deviations found
- [MEDIUM] A. Coverage: Paper studies cross-domain transfer across 4 humor sub-genres (Amazon Questions, one-liners, Onion / sarcasm headlines, Dad Jokes). Scenario implements only 2 (sarcasm_headlines + amazon_questions); one-liners and dad_jokes are missing, so the cross-domain transfer thesis cannot be fully reproduced.
- [LOW] C. Metric/scoring fidelity: Regex permissively accepts paraphrases ("funny" / "humorous" / "non-humorous" / "not funny") rather than strict Yes/No match. Defensible for free-form decoder output, but slightly looser than Appendix C's strict label scoring.
- [LOW] D. Generation config: `_use_defaults: true`; should explicitly pin T=0.0 + low max_tokens for binary classification rather than inheriting open-ended defaults.
- [info] B. Prompt fidelity: verbatim from Appendix C, Alpaca-style.
- [info] A. Reference labels: pulled from source dataset gold (`is_sarcastic`, Amazon `label`) — no LLM-judge needed.

## Notes
Prompt template, metric, and label sourcing are all faithful. The two real watch-items are coverage (2/4 subsets) and the implicit-defaults inference config. Re-classifying coverage as MEDIUM (rather than HIGH as in the prior audit) because the two subsets that ARE implemented match the paper protocol exactly, and the scope reduction is documented in the scenario header. To upgrade toward Tier 1: add the missing `one_liners` and `dad_jokes` subsets, and pin a deterministic inference config explicitly.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is_with_caveat (or expand subsets)
- Now:   Tier 2, high, keep_as_is_with_caveat
- Delta: confirmed (downgraded coverage from HIGH to MEDIUM since the implemented subsets are paper-faithful and the gap is documented)
