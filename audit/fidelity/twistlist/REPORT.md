# twistlist fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** patch_with_phonetic_metrics (PO/Init-PO/iPED/oPED — paper's signature contribution)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/twistlist_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`, `metrics.bert_score_metric.BertScoreMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: Loakman et al., "TwistList: Resources and Baselines for Tongue Twister Generation" (ACL 2023, arXiv:2306.03457)
- Repo: https://github.com/tangg555/TwistList

## Implementation audited
- scenarios/twistlist_scenario.py — defaults to `use_prompts=False` (keywords-only) and wraps with `"Generate a tongue twister using these key words: {keywords}"`. Run spec passes `args={}`. Paper has 2 prompt variants (`tt-data` keywords-only; `tt-prompt-data` with prefix).
- run_specs/twistlist_run_specs.py — `BasicGenerationMetric{exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4}` + `BertScoreMetric(bert-base-uncased)`. No dedicated `metrics/twistlist_metric.py`.
- 2,125 instances total (1,912 train / 106 val / 107 test). Matches paper.

## Deviations found
- [HIGH] **Paper's signature phonetic metrics missing**: PO (Phoneme Overlap), Init-PO (Initial-Phoneme Overlap), iPED/oPED (Phonemic Edit Distance). These are the domain-specific contribution; HELM omits them entirely.
- [MEDIUM] **Human ratings not replicated**: paper's articulation difficulty, amusement Likert ratings absent.
- [LOW] **Wording divergence**: HELM's prompt is semantically equivalent to repo's prompt-data variant but not byte-identical.
- [LOW] **Metric mismatch**: exact_match/QEM/F1 are weak fits for open-ended phonetic generation against single reference.
- [INFO] Inference: T=0.7, max_tokens=512, 5-shot. Paper unspecified for LLM baselines.

## Notes
Single human-authored reference per instance with CORRECT_TAG. No LLM judge. Tier 2 because metric coverage is the gap — paper's phonetic-overlap metrics (the entire contribution!) are absent. Standard NLP metrics covered but inappropriate for phonetic generation.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, patch_with_phonetic_metrics
- Now:   Tier 2, high, patch_with_phonetic_metrics
- Delta: confirmed (verified scenarios/twistlist_scenario.py lines 75-94 loads train/val/test from Dropbox-extracted tt-data; default `use_prompts=False` triggers added "Generate a tongue twister using these key words:" wrapper at line 109; registry exposes 6 BasicGenerationMetric stats but no PO/iPED/oPED phonetic metrics)
