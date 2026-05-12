# ss_gen fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is (human-rater criteria appropriately omitted)

## Paper / repo audited
- Paper: SS-GEN: A Social Story Generation Framework with LLMs (arXiv:2406.15695)
- Repo: https://github.com/MIMIFY/SS-GEN

## Implementation audited
- scenarios/ss_gen_scenario.py — uses paper's title-only template verbatim: "Develop a concise, clear, straightforward, positive and supportive Social Story titled '{title}' for children and teens with autism, 200-300 words..."
- run_specs: BLEU-4, ROUGE-1/2/L via HELM `BasicGenerationMetric`; BERTScore (`bert-base-uncased`); 5 GPT-4 LLM-judge rubrics (coherence, descriptiveness, empathy, grammaticality, relevance), 1-5 Likert, T=0.0, max 256 tokens.
- 5,085 stories total (train 4,068 / dev 509 / test 508), exact match to paper.

## Deviations found
- [LOW] **Human-only criteria omitted**: paper's Structural Clarity, Descriptive Orientation, Situational Safety axes require human raters; not implementable in HELM. Appropriate omission.
- [LOW] **Rubric anchors paraphrased** rather than copied verbatim from paper appendix; dimensions and scale align.

## Notes
Judge: `openai/gpt-4` (overridable via `SS_GEN_JUDGE_MODEL_OVERRIDE`); matches paper. Reference: `story_content` set as CORRECT_TAG for BLEU/ROUGE/BERTScore. Prompt, splits, instance counts, automated metric suite, judge model, and reference targets all match paper. Tier 1.
