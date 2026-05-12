# puntuguese fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_caveat (pun-location intentionally omitted; zero-shot prompted protocol is a documented shift from paper's supervised classifiers)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/puntuguese_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.markdown_normalized_classification_metric.MarkdownNormalizedMCQClassificationMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://aclanthology.org/2024.lrec-main.1167/ (Inácio et al., "Puntuguese: A Corpus of Puns in Portuguese," LREC-COLING 2024) — note: read (abstract; confirms 4,903 micro-edited pun pairs, supervised F1 = 68.9%)
- Repo / dataset: https://huggingface.co/datasets/Superar/Puntuguese — note: skim

## Implementation audited
- scenarios/puntuguese_scenario.py — Loads `Superar/Puntuguese` HF `test` split (1,140 instances). Builds Portuguese yes/no prompt (lines 58–62): `"Texto: {text}\n\nEste texto é humorístico?\nResponda apenas com Sim ou Não."`. Two references with `CORRECT_TAG` on the appropriate `Sim` (label=1) or `Não` (label=0). `tokens` and `labels` (the pun-location annotations) intentionally skipped per scenario docstring.
- metrics/puntuguese_metric.py — does not exist (`has_metric_file: false`). Scoring relies on HELM's `MultipleChoiceClassificationMetric`.
- registry_metrics.yaml (lines 2197–2206): two metrics — `classification_macro_f1` and `classification_micro_f1` (both `MultipleChoiceClassificationMetric`).
- registry_inference.yaml (lines 543–548): `temperature: 0.0`, `max_tokens: 4`, `num_outputs: 1`, `max_train_instances: 0` — explicitly pinned for the zero-shot Sim/Não classification.

## Deviations found
- [MEDIUM] A. Coverage: Paper releases two annotation layers (binary humor recognition + token-level pun-location). AGC keeps only binary humor recognition. Documented in scenario docstring as a deliberate scope reduction.
- [MEDIUM] B./D. Protocol shift: Paper reports supervised Portuguese classifier baselines (F1 = 68.9%); AGC runs zero-shot prompted classification. Documented; scores are not directly leaderboard-comparable.
- [LOW] A. Subsample: 200-instance cap (benchmark-wide policy) vs. full 1,140-instance test split.
- [info] C. Metric mapping: macro-F1 is the closest analogue to the paper's primary F1 (since the corpus is balanced by construction via micro-editing).
- [info] D. Inference config is explicitly and correctly pinned (T=0.0 for binary classification, max_tokens=4 for "Sim"/"Não").

## Notes
Clean adaptation. The scenario picks the right slice of the dataset for an LLM-as-classifier evaluation, the prompt is in Portuguese, the inference config is correctly pinned for binary classification, and the two scope reductions (pun-location omitted; zero-shot rather than supervised) are documented. No fix required to ship; just keep the caveats in the paper writeup.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is (documented scope reduction; pun-location intentionally omitted)
- Now:   Tier 2, high, keep_as_is_with_caveat
- Delta: confirmed
