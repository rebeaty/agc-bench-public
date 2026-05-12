# data_narrative fidelity audit

**Tier:** 2
**Confidence:** medium-high
**Recommendation:** keep_as_is_with_judge_prompts (fix null judge prompts before scoring)

## Paper / repo audited
- Paper: Islam et al., EMNLP 2024, "DataNarrative" (arXiv:2408.05346)
- Repo: github.com/saidul-islam98/DataNarrative

## Implementation audited
- scenarios/data_narrative_scenario.py — single-turn prompt over one paragraph-table segment; reference = gold paragraph; 1,914 instances (226 train Tableau / 1,688 test across GapMinder 42, Pew 1,590, Tableau 56).
- registry_metrics.yaml: BLEU-4, BERTScore, plus five LLM-judge metrics (Gemini-2.5-flash-lite, T=0, max_new_tokens=16) named for paper's five dimensions.
- No `metrics/data_narrative_metric.py` present.

## Deviations found
- [HIGH] **Task scope reduced**: paper evaluates multi-stage agentic generation of text + visualizations. HELM evaluates single-turn paragraph generation only. "Visualization Quality" dimension is renamed to `relevance` (theme alignment) — not equivalent.
- [HIGH] **Judge model swap + protocol change**: paper uses Gemini-1.5-pro pairwise; HELM uses Gemini-2.5-flash-lite pointwise.
- [HIGH] **Judge prompts null**: all five judge entries set `judge_prompt: null`; no rubric specified. `judge_max_new_tokens=16` is tight.
- [MEDIUM] Granularity shift: 1,914 paragraph-table segments vs paper's 1,449 stories.
- [LOW] Added BLEU-4, BERTScore not used in paper; benign diagnostics.

## Notes
Instance counts and source splits match released JSON. Prompt is faithful simplification of Figure 24. Tier 2: core five dimensions and data are preserved while modality, protocol, and judge model differ. Populating five judge prompts from paper Figures 18-26 and raising max_new_tokens would elevate confidence to High.
