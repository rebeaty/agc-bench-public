# story_generation_rocstories fidelity audit

**Tier:** 2
**Confidence:** high
**Recommendation:** discuss (rebrand or add five-dimension judge)

## Paper / repo audited
- Source paper: DeltaScore: Fine-Grained Story Evaluation with Perturbations (Xie et al., 2023; arXiv:2303.08991)
- Source repo: https://github.com/ZhuohanX/DeltaScore

## Implementation audited
- scenarios/story_generation_rocstories_scenario.py — builds `"Write a story based on the following prompt:\n\n{title}\n\nStory:"` from `title` field of `data/crowdsource/roc.jsonl`. Reasonable but not specified by DeltaScore.
- run_specs/story_generation_rocstories_run_specs.py — `BasicGenerationMetric` with `rouge_1, rouge_2, rouge_l, bleu_4`. Registry expects `bleu_4` and `rouge_l`.
- 20 unique ROCStories prompts.

## Deviations found
- [HIGH] **Adaptation, not reproduction**: DeltaScore is a meta-evaluation paper (perturbs stories, measures correlation with humans). Paper does NOT define a story-generation benchmark. Scenario repurposes DeltaScore prompt/reference pairs as prompt-to-story task.
- [HIGH] **Metric mismatch with paper's intent**: BLEU/ROUGE are precisely the baselines paper argues are insufficient. Reference-overlap on a single human story does not capture five fine-grained dimensions (fluency, coherence, relatedness, logicality, interestingness).
- [LOW] n=20 small but matches source.
- [INFO] Inference defaults (T=0.7, max_tokens=512); paper silent.

## Notes
Single human-written reference per prompt. No LLM judge. Tier 2 because adaptation should be flagged; current implementation uses metrics paper deprecates.

**Recommendation:** Add LLM-judge metric scoring DeltaScore's five dimensions (fluency, coherence, relatedness, logicality, interestingness) on 1-5; demote BLEU/ROUGE. OR rename to clarify this is a DeltaScore-derived prompt set, not a reproduction.
