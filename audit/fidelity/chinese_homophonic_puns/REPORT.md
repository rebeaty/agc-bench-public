# chinese_homophonic_puns fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_caveat (document prompt simplification — paper used batched line-numbered prompt; we use single-instance prompt)

## Paper / repo audited
- Paper: https://arxiv.org/abs/2405.15818 ("DuanzAI: Slang-Enhanced LLM with Prompt for Humor Understanding") — abstract only via arXiv; details from repo. (note: "skim")
- Repo: https://github.com/YesianRohn/DuanzAI (`evaluatePunchline.py`, `prompt.py`, `data/task_1.json`) (note: "read")

## Implementation audited
- scenarios/chinese_homophonic_puns_scenario.py — fetches `data/task_1.json` (currently 2,546 items per upstream; paper says "nearly 2500"). Each instance is `text` → punchline. Prompt is a per-item adaptation of the upstream batched prompt: "你现在的任务是从下面的笑话中找出其中幽默来源的那一个词语... 笑话：{text} 答案：" — original instructs line-numbered batched output; ours strips that for single-instance. Spirit preserved.
- metrics/chinese_homophonic_puns_metric.py — `ChineseHomophonicPunsMetric.evaluate_generation` returns `exact_match_accuracy` (strict equality) and `similar_match_accuracy` = `min(1, max(SequenceMatcher.ratio, fuzz.ratio/100))`. Matches `evaluatePunchline.py` formula verbatim.
- registry_metrics.yaml (lines 456-465): two formula_based metrics routed to ChineseHomophonicPunsMetric.
- registry_inference.yaml (lines 106-108): `_use_defaults: true`.

## Deviations found
- [LOW] B. Prompt fidelity: Original is batched ("each line a joke; output one word per line with index"); ours is single-instance ("找出... 答案："). Spirit and the key constraints (单一词语, 未知-fallback) preserved.
- [LOW] A. Dataset source: All 2546 items used; matches "nearly 2500" in paper. Per-prediction prompt cleaner strips "答案:" / "回答:" / "Punchline:" prefixes — reasonable.
- [info] C. Metric fidelity: Exact and fuzzy similar match formulas match upstream `evaluatePunchline.py` exactly.
- [info] D. Generation config: defaults; paper does not specify (but task is short MCQ-like extraction, would benefit from low T).

## Notes
Solid implementation. The scenario faithfully ports both data and metric. The only meaningful deviation is the prompt: upstream batched line-numbered prompt was reasonably converted to single-instance, which is the right call for HELM. One small recommendation: ensure inference uses low temperature (this is essentially extraction, not generation) — currently relying on defaults which may be too high.

## Compared to prior audit (2026-04-25)
- Prior: Tier ?, unknown, no audit conducted
- Now:   Tier 2, high, keep_as_is_with_caveat
- Delta: newly classified
