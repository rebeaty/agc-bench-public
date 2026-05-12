# lcc_metaphor fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is (note prompt is benchmark-authored; assert split count guard)

## Paper / repo audited
- Paper: Aghazadeh et al., ACL 2022, "Metaphors in Pre-Trained Language Models" (https://aclanthology.org/2022.acl-long.144/); dataset = LCC Metaphor Corpus (Mohler et al., LREC 2016).
- Repo: https://github.com/EhsanAghazadeh/Metaphors_in_PLMs

## Implementation audited
- scenarios/lcc_metaphor_scenario.py — self-authored binary template: `Is the word "{target_word}" used metaphorically...? Answer (Yes or No):`. Source paper is a probing study of PLM hidden states, not LLM prompting; no prompt template exists in paper. Scenario docstring transparently flags as benchmark-authored.
- registry: `exact_match`, `quasi_exact_match`, `f1_score` via HELM `BasicGenerationMetric`.
- 8,028 balanced English test examples (per scenario header); data pulled live from `data/multi_ling/lcc_en/test.csv`. Subsets en/es/ru/fa exposed.

## Deviations found
- [MEDIUM] **Prompt is benchmark-authored** — paper does not specify a prompt (it's a probing study, not LLM evaluation). HELM constructs a reasonable Yes/No template. Documented in docstring.
- [INFO] Inference: `_use_defaults: true` — acceptable; paper does not prompt an LLM.

## Notes
Tier 2 because prompt is constructed (not from paper). F1 matches paper's binary classification metrics; exact_match is sound proxy for accuracy on balanced set. No custom metric file needed.

Optional improvements: (a) note in docstring that exact_match == accuracy on balanced binary set; (b) add assertion that downloaded `en` test split == 8,028 rows to guard against upstream drift.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is
- Now:   Tier 2, high, keep_as_is
- Delta: confirmed (verified scenarios/lcc_metaphor_scenario.py lines 52-56 self-authored Yes/No template; pulls `lcc_<subset>/test.csv` live; registry exposes exact_match + quasi_exact_match + f1_score under BasicGenerationMetric)
