# tinyfabulist fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is

## Paper / repo audited
- Paper: Lupascu et al., "TF1-EN-3M: Three Million Synthetic Moral Fables..." (arXiv 2504.20605)
- Repo: github.com/klusai/tinyfabulist

## Implementation audited
- scenarios/tinyfabulist_scenario.py — downloads exact upstream file `tf_prompts_c100_dt250402-085509.jsonl` from paper repo and extracts only `generator_prompt` content.
- run_specs/tinyfabulist_run_specs.py — system message matches paper's 6-slot fable scaffold (character/setting/conflict/resolution/moral; age groups A-E). T=0.7, max_tokens=512, n=1.
- llm_judge/tinyfabulist_metric.py + tinyfabulist_annotator.py — grammar, creativity, moral_clarity, adherence_to_prompt (1-10), mean_judge_score, age_group_{a..e}_rate, valid_judge_rate. Judge = `openai/o3-mini-2025-01-31`, T=0.0, 350 max tokens.
- metrics/tinyfabulist_corpus_metric.py — self_bleu (sentence-BLEU leave-one-out avg), distinct_1 (per-fable, averaged), flesch_reading_ease.
- benchmark source = 100 prompts (matches paper's curated benchmark scoring 11 models). Optional `full` mode loads 100K HF test split deduped by `prompt_hash`.

## Deviations found
- [LOW] Flesch computed via internal regex rather than upstream `textstat`; small numeric drift possible (acknowledged in notes).
- [LOW] `target_age_group` hard-coded to "B" in `extra_data`; only matters if used as ground truth for age classification downstream.
- [LOW] Judge prompt uses `{generated_response}` placeholder — verify `TinyFabulistAnnotator` substitutes correctly.

## Notes
No reference targets (`references=[]`); reference-free per paper. Judge rubric covers same 4 dimensions as paper. Prompts, count (100), judge model, and metric surface track paper closely. Tier 1.
