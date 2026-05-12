# showerthoughts fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is (linguistic/detector metrics from paper out of scope; populate the 6 judge prompts with paper's Likert rubrics)

## Paper / repo audited
- Paper: https://aclanthology.org/2024.starsem-1.23/ (Hofmann et al., *SEM 2024, "Investigating Wit, Creativity, and Detectability of Large Language Models in Domain-Specific Writing Style Adaptation of Reddit's Showerthoughts") — "skim" (relied on scenario header; paper rates standalone Showerthoughts on six 1-6 Likert axes: general_score, logical_validity, creativity, humor, cleverness, real_person)
- Repo: https://github.com/aiintelligentsystems/showerthoughts-dataset — "skim" (verified `generated/roberta_test_data_mixed.ndjson` path)

## Implementation audited
- scenarios/showerthoughts_scenario.py — Downloads `roberta_test_data_mixed.ndjson` from upstream repo; loads only `label == "genuine"` items as audit-trail metadata (NOT as references). Default `num_instances=300` standalone generation slots. Prompt at lines 88–94 adapts the paper's Section 4.1 ChatGPT prompt to one-at-a-time generation. Per-instance zero-width nonce (`_TRIAL_NONCE = "​"` × idx) appended to defeat HELM request caching across repeated trials.
- No metric file (uses 6 LLM-judge annotators).
- registry_metrics.yaml (line 2419): registers all 6 paper Likert axes as separate judges (`llm_judge_general_score`, `llm_judge_logical_validity`, `llm_judge_creativity`, `llm_judge_humor`, `llm_judge_cleverness`, `llm_judge_real_person_likelihood`), all with judge_model_name=`openai/gpt-4`, T=0.0, max_new_tokens=256, judge_prompt=null.
- registry_inference.yaml (line 607): T=0.7, max_new_tokens=64, num_return_sequences=1, stop_sequences=["\n"]. Documented source: "paper-inspired single-item Showerthought generation with newline stop for concise standalone outputs".

## Deviations found
- [MEDIUM] C. Metric/scoring fidelity: all 6 judges have `judge_prompt: null`. The paper's Likert rubrics (1-6 scale per axis) need to be populated, otherwise the judge calls produce undefined scores. The 6 axes are correctly carved out — just missing the rubric strings.
- [LOW] A. Dataset/instance source: paper used 411,189 Pushshift-collected Showerthoughts; that corpus is no longer redistributable post-Reddit policy change. The implementation uses the public 3,000-genuine slice purely to define stable evaluation slot count + audit metadata, not as references — defensible workaround.
- [LOW] B. Prompt fidelity: paper's prompt asks for 100 Showerthoughts per call; implementation adapts to one-per-call to fit HELM's per-instance protocol while preserving style guidance. The zero-width-nonce trick is a clever cache-busting fix.
- [LOW] C. Metric/scoring fidelity (coverage): paper additionally reports linguistic-style and detector metrics (RoBERTa AI-text classifier accuracy, perplexity, lexical diversity); these are explicitly out of scope per scenario header — acceptable as a HELM standardization choice.
- [info] D. Generation configuration: T=0.7, max_tokens=64, stop=["\n"] are well-tuned for "one short witty observation".

## Notes
The implementation is well-engineered (the per-instance nonce is a thoughtful fix for cache-replay). Tier 2 because the 6 judge rubrics need to be populated before reporting numbers — otherwise `llm_judge_creativity` is just calling GPT-4 with no instructions. The 6-axis carve-out faithfully matches the paper's standalone-rating panel; once rubrics are added this becomes Tier 1.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is (linguistic/detector metrics from paper out of scope)
- Now:   Tier 2, high, keep_as_is (with rubric population)
- Delta: confirmed
