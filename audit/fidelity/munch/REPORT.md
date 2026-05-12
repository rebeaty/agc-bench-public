# munch fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_caveats (optionally add Metaphor-Sent + multi-prompt averaging from paper's prompts.md)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/munch_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2403.11810 (Tong et al., "Metaphor Understanding Challenge Dataset for LLMs", ACL 2024) — "skim" (relied on scenario header; paper details 1,492 judgement examples × 4 task variants)
- Repo: https://github.com/xiaoyuisrain/metaphor-understanding-challenge — "skim" (verified data file paths `tasks/word_judge.json`, `tasks/sent_judge_implicit.json`, `tasks/sent_judge_mword.json`)

## Implementation audited
- scenarios/munch_scenario.py — Downloads JSON from raw GitHub. Four subsets (word_implicit, word_mword, sent_implicit, sent_mword) using prompt templates reproduced from `prompts.md` (lines 67–97). Builds 4-way MCQ with options A (only A apt), B (only B apt), C (both apt), D (neither apt). Reference is the gold letter A/B/C/D.
- No metric file (uses HELM built-ins).
- registry_metrics.yaml (line 1860): registers `exact_match` and `quasi_exact_match`.
- registry_inference.yaml (line 449): `_use_defaults: true`.

## Deviations found
- [LOW] B. Prompt fidelity: the WOTG/SSTA/GASW templates from the paper's prompts.md are 4 of ~20 prompt variants used in the paper for multi-prompt averaging. We use only one variant per subset rather than averaging across templates. The selected variant ("WOTG" for word, "SSTA"/"GASW" for sentence) is the cleanest of the published set.
- [LOW] C. Metric/scoring fidelity: paper primary result is 4-way accuracy on the multiple-choice judgement task — `exact_match` matches. Paper additionally reports per-class precision/recall (apt vs inapt) and a Metaphor-Sent companion metric; these are not registered.
- [info] A. Dataset/instance source: pulls full upstream JSON files (1,492 items per subset).
- [info] D. Generation configuration: defaults; paper uses standard chat-completion settings.

## Notes
This is a clean, faithful implementation of the MUNCH judgement subtask. Tier 2 because it omits the multi-prompt averaging the paper uses to reduce prompt-template variance — adding 2–3 alternate templates per subset and averaging would tighten paper-comparability but is optional. The 4-way A/B/C/D mapping correctly handles "Both apt" → C and "Neither apt" → D, which is the trickiest part of the original protocol.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is_with_caveats (or add Metaphor-Sent + multi-prompt averaging)
- Now:   Tier 2, high, keep_as_is_with_caveats
- Delta: confirmed
