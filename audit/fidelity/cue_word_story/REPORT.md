# cue_word_story fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** medium
**Recommendation:** patch_with_sentence_embeddings_and_judge_prompt

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/cue_word_story_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`, `metrics.cue_word_story_metric.CueWordStoryMetric`, `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric`, `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric`, `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric`, `llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric`
> - **AnnotatorSpec(s):** `llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator`, `llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator`, `llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator`, `llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: Atmakuru et al., "Evaluating Creative Short Story Generation in Humans and LLMs" (arXiv 2411.02316, ICCC 2025).
- Repo: https://github.com/mismayil/creative-story-gen (`src/prompts.py`, `data/pilot_data.json`).

## Implementation audited
- scenarios/cue_word_story_scenario.py — system+user prompts copied verbatim from `src/prompts.py`; 4 cue-word sets with correct low/high split and boring themes; human-authored stories from HF dataset used as references.
- metrics/cue_word_story_metric.py — cue-word coverage, all-cue rate, 5-sentence compliance, boring-theme overlap, n-gram diversity, novelty/surprise proxies (char-3–5 TF-IDF cosine).
- Registry adds HELM `BasicGenerationMetric` overlap stats (exact_match, F1, BLEU-1/4, ROUGE-L) and a 4-axis gpt-4 LLM judge.

## Deviations found
- [HIGH] Instance count: only 4 prompts (one per cue-word set). Paper generates many stories per set (≈243 AI). HELM gives near-zero statistical power.
- [MEDIUM] Novelty/surprise: paper uses sentence-transformer embeddings; implementation substitutes char-n-gram TF-IDF over dominant terms. Direction-matching but not paper's formula.
- [MEDIUM] Lexical/syntactic complexity (POS, dep length, constituency depth) from paper not implemented.
- [MEDIUM] LLM judge: registry lists 4 axes with `judge_prompt: null`; paper rubric not encoded.
- [LOW] Reference-based BLEU/ROUGE/F1/EM included as HELM defaults — harmless but off-spec.
- [LOW] Boring-theme overlap is HELM-introduced diagnostic absent from paper.

## Notes
Prompt fidelity high. Patch priorities: (a) encode paper's judge rubric, (b) replace TF-IDF with sentence-transformer embeddings (paper uses sentence-transformer cosine for novelty/surprise), (c) draw multiple samples per cue set so the 4-instance count yields usable statistical power.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, medium, patch_with_sentence_embeddings_and_judge_prompt
- Now:   Tier 2, medium, patch_with_sentence_embeddings_and_judge_prompt
- Delta: confirmed (verified scenarios/cue_word_story_scenario.py lines 122-143 cue-word sets, lines 75-88 verbatim prompts; metrics/cue_word_story_metric.py uses TfidfVectorizer char_wb 3-5 not SBERT; registry exposes 6 BasicGenerationMetric overlap stats + 7 custom CueWordStoryMetric stats)
