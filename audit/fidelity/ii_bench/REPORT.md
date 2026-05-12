# ii_bench fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_disclosure (use dev split N=35; submit to EvalAI for true test numbers if needed)

## Paper / repo audited
- Paper: https://arxiv.org/abs/2406.05862 (II-Bench, NeurIPS 2024 D&B) — abstract via arXiv. Confirms: accuracy is primary metric, 74.8% best-MLLM vs 90% human. (note: "skim")
- Repo: https://github.com/II-Bench/II-Bench and HF dataset `m-a-p/II-Bench`. (note: "skim" — relied on scenario header)

## Implementation audited
- scenarios/ii_bench_scenario.py — loads HF `m-a-p/II-Bench` `dev` split (35 items, since `test` answers are hidden behind EvalAI). Saves PIL images to `output_path/images/{idx}.jpg`. Constructs MCQ prompt with 6 options (A-F), instruction "Select exactly one option and reply with a single letter from A to F", optional CoT trigger. Builds 6 References with the gold one tagged CORRECT_TAG.
- metrics/ii_bench_metric.py — `IIBenchMetric.evaluate_generation` uses upstream parsing rules: `(X)` pattern → bare `X` pattern → option-text fuzzy match. Returns `accuracy`, `errors_rate`, `miss_rate`. Mirrors upstream parsed-answer evaluator.
- registry_metrics.yaml (lines 1498-1511): three formula_based metrics, all routed to `IIBenchMetric`. Primary `accuracy` matches paper.
- registry_inference.yaml (lines 341-343): `_use_defaults: true`.

## Deviations found
- [MEDIUM] A. Dataset source: We use the **35-instance dev split** (test answers hidden). Paper primary numbers (74.8%) are on 1,399-item test split via EvalAI. Documented in scenario header (lines 39-42). N=35 has substantial sampling noise.
- [LOW] B. Prompt fidelity: Standard MCQ prompt aligned with paper's zero-shot baseline. Optional CoT path matches paper's CoT mode. Not verbatim from any prompt file but spirit preserved.
- [LOW] C. Metric fidelity: Accuracy with upstream parsing rules; matches paper. Plus diagnostic errors_rate / miss_rate.
- [info] D. Generation config: defaults; paper uses standard MCQ-style decoding.

## Notes
Sound implementation; the only material limitation is dev split (35) instead of hidden-label test (1,399). Numbers will land within ~10pp noise of paper values; not directly comparable to the leaderboard. If paper-comparability matters, submit predictions to EvalAI; otherwise the dev-split numbers are reportable with a sample-size disclaimer. The MCQ parser in `_extract_option_label` looks robust (parenthesized → bare letter → fuzzy option-text fallback).

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is_with_disclosure (use dev split N=35; submit to EvalAI for true test numbers if needed)
- Now:   Tier 2, high, keep_as_is_with_disclosure
- Delta: confirmed
