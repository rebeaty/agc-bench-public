# metaphoric_analogies fidelity audit (re-audit 2026-05-04)

**Tier:** 3
**Confidence:** high
**Recommendation:** patch_with_few_shot_and_lemmatized_metric (add few-shot examples; replace token-F1 with lemmatized head-noun matching; consider LLM-judge proxy for the implicit-term subtask)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/metaphoric_analogies_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2412.15375 (Mionies et al., "Automatic Extraction of Metaphoric Analogies from Literary Texts," COLING 2025) — note: skim (abstract; methodological details from scenario header notes)
- Repo: https://github.com/Mionies/metaphoric-analogies-extraction — note: skim (preamble cited from `experiments/openai_models.py` lines 55–64 in scenario header)

## Implementation audited
- scenarios/metaphoric_analogies_scenario.py — Loads `signed-met-1.3.csv`. Preamble (lines 97–107) lifted nearly verbatim from `experiments/openai_models.py` lines 55–64 (T1/T2/S1/S2 explanation + implicit-term clause + Answer template). Each row is rotated 4× (one concept supplied at a time → model produces the other three). Reference = full 4-line "T1: / T2: / S1: / S2:" block. **Zero-shot** (no few-shot examples).
- metrics/metaphoric_analogies_metric.py — does not exist (`has_metric_file: false`).
- registry_metrics.yaml (lines 1759–1768): two basic metrics — `exact_match` and `f1_score` (both `BasicGenerationMetric`, token-level over the full multi-line reference block).
- registry_inference.yaml (lines 425–427): `_use_defaults: true`.

## Deviations found
- [HIGH] C. Metric/scoring fidelity: Paper evaluates with **lemmatized head-noun matching** for explicit terms and **0–2 human ratings** for implicit terms. Registry uses token-level F1 over a 4-line "T1:/T2:/S1:/S2:" block, which rewards label-string echoing (the model just has to emit "T1:" "T2:" "S1:" "S2:" prefixes for free F1) and entirely skips lemmatization. Numbers are not paper-comparable.
- [HIGH] B. Prompt fidelity: Scenario is zero-shot; paper uses few-shot with worked examples. This depresses absolute scores below the paper's reported numbers.
- [MEDIUM] C. Implicit-term subtask: The paper's hardest subtask requires human 0–2 ratings for implicit (inferred) terms. AGC silently scores implicit-term predictions under the same token-F1 metric as explicit extraction — wrong scoring construct.
- [LOW] B. PREAMBLE has a few missing inter-sentence whitespace joins (cosmetic; semantics intact).
- [info] A. Dataset: 203 complete rows × 4 concept rotations = 812 candidate instances. Subsampling happens at run-spec layer.
- [info] D. Generation config: `_use_defaults: true`; paper specifies decoding only loosely.

## Notes
This is the primary case where the registered metric does not match the paper construct. Token-F1 over a structured "T1:/T2:" block is well known to over-credit format echoing. The fix is two-fold: (1) write a `metaphoric_analogies_metric.py` that parses the four answer slots, lemmatizes head nouns, and matches against gold (per paper protocol); (2) add a few-shot adapter so absolute scores are comparable. Optionally add an LLM-judge or proxy human-rating annotator for the implicit-term subtask.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, patch_with_few_shot_and_lemmatized_metric
- Now:   Tier 3, high, patch_with_few_shot_and_lemmatized_metric
- Delta: regressed (two independent HIGH issues — metric mismatch and zero-shot/few-shot mismatch — both block paper-comparability; under "any HIGH → Tier 3" rubric, the prior Tier 2 was generous)
