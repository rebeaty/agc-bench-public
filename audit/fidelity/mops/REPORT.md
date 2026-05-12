# mops fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is

## Paper / repo audited
- Paper: Ma et al., "MoPS: Modular Story Premise Synthesis" (arXiv:2406.05690)
- Dataset: ManTle/mops on HF, split `curated` (100 premises across 14 themes)

## Implementation audited
- scenarios/mops_scenario.py — reproduces upstream `SYNTHESIZE_PROMPT` from GAIR-NLP/MoPS: concatenates Theme, Background, Persona, Event, Ending, Twist with section headers and requests one "compact, concise, and coherent sentence as a story premise."
- llm_judge/mops_annotator.py + mops_metric.py — three judge dimensions (Fascination, Completeness, Originality, each 0-100) with prompts paraphrasing paper rubric verbatim including originality "0 = exactly seen, 100 = never seen" scale and validity safeguard ("must be a complete premise else 0"). `mops_quality_score` = mean (paper composite). `mops_valid_judge_rate` is benchmark-added parse-success tracker.
- metrics/mops_diversity_metric.py — replicates upstream recipe: MiniLM mean-pooled embeddings → t-SNE 2D (perplexity 50, seed 42) → ConvexHull area for **breadth**, 10x10 2D histogram with row-trim mask + std for **density**.
- 100 instances across 14 themes (Historical, Military, Romance, Martial Arts, Fantasy, Sports, Urban, Fantastic, Contemporary, Time-travel, Science Fiction, Game, Immortal Heroes, Suspense). EXACT MATCH to paper.

## Deviations found
- [LOW] Cosmetic: header levels mix `###` and `##` (likely typo).
- [LOW] **Single judge** (GPT-4-Turbo, T=0.0) — paper used GPT-4-Turbo + Claude-3-Opus dual; only first wired. Acceptable subset.
- [INFO] Adapter: zero-shot, T=0.7, max_tokens=256 (paper does not specify generation config).

## Notes
PROMPT, METRIC, INSTANCE COUNT, and JUDGE all faithfully reflect source paper and upstream repo. Embedded-prefix stripper sanitizes noisy fields. No leakage of reference premise into prompt. Tier 1.
