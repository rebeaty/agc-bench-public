# grapheval_review_advisor fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is (with documented caveat: dataset is in repo but not discussed in the GraphEval paper primary tables; class-distribution note in prompt mismatches Review_Advisor split)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/grapheval_review_advisor_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.grapheval_decision_metric.GraphEvalDecisionMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2503.12600 ("GraphEval") — abstract via arXiv. Paper's primary benchmarks differ; Review_Advisor is a repo split, not a paper-table split. (note: "skim")
- Repo: https://github.com/ulab-uiuc/GraphEval/tree/main/Data/Review_Advisor (data) and `Baselines/Prompt/basic_prompt.txt` (prompt). (note: "read" via scenario; scenario quotes verbatim)

## Implementation audited
- scenarios/grapheval_review_advisor_scenario.py — fetches `Data/Review_Advisor/ReviewAdvisor_test_set.jsonl` (1,025 test items per scenario header). System prompt (lines 51-96) is the verbatim `Baselines/Prompt/basic_prompt.txt` including 4 few-shot ICLR examples and the **ICLR class-distribution note (62/27.8/5.43/4.13)** which does not match Review_Advisor's actual distribution (56.5/40.1/1.9/1.6). Per scenario docstring (lines 26-30), this mismatch is intentional — they preserved upstream verbatim. References = decision string; gold rating mean (×10) attached as `gold_score` for Spearman.
- metrics/grapheval_decision_metric.py — `GraphEvalDecisionMetric` (EvaluateInstancesMetric, shared with `grapheval_ai_researcher`). Parses "Overall Score (0-100)= N\n{decision}" via regex with sane fallbacks. Emits decision_accuracy, decision_parsed, score_parsed, per-class precision/recall/F1 (macro), Spearman correlation between predicted and mean-rating-derived gold score, plus per-label tp/fp/fn.
- registry_metrics.yaml (lines 1336-1365): seven formula_based metrics, all routed to `GraphEvalDecisionMetric`. (note: `has_metric_file: false` in batch JSON refers to `metrics/grapheval_review_advisor_metric.py`, but the actually-used file `metrics/grapheval_decision_metric.py` does exist.)
- registry_inference.yaml (lines 299-301): `_use_defaults: true`.

## Deviations found
- [LOW] B. Prompt fidelity: Verbatim from upstream `basic_prompt.txt` — but the upstream prompt's class-distribution hint targets the ICLR_Dataset, not Review_Advisor. Documented in scenario header. Could be considered confounded as a fairness signal to the model; defensible since it preserves upstream behavior.
- [MEDIUM] A. Dataset source: Review_Advisor is an **official repo split that is not in the paper's primary tables**. Numbers will not be directly comparable to anything in the GraphEval paper. The scenario header acknowledges this.
- [LOW] C. Metric fidelity: decision_accuracy + macro F1 + Spearman are reasonable scoring choices for a 4-class classification task with auxiliary score regression. Implementation is clean.
- [info] D. Generation config: defaults; OK for short structured output.

## Notes
Implementation is technically sound — verbatim prompt, clean parser, well-thought-out metrics including parse-rate hygiene checks. The fundamental constraint is that Review_Advisor itself is a repo-only split, so any numbers are within-benchmark, not paper-primary-comparable. Worth a single-line caveat in the AGC paper. The class-distribution mismatch in the prompt is preserved upstream behavior; flagging it but no fix needed unless we want to swap to a Review_Advisor-correct distribution string.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is (with documented caveats)
- Now:   Tier 2, high, keep_as_is
- Delta: confirmed
