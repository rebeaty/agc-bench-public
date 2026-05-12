# newyorker_humor fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_caveats (matching/ranking are well-aligned; explanation task uses generation against gold-explanation reference, not paper's pairwise human eval)

## Paper / repo audited
- Paper: https://aclanthology.org/2023.acl-long.41/ (Hessel et al., ACL 2023, "Do Androids Laugh at Electric Sheep?") — referenced via scenario header (note: "skim" — paper not refetched but task structure matches HF dataset card)
- Repo/Dataset: HF `jmhessel/newyorker_caption_contest` and `github.com/jmhessel/caption_contest_corpus` (note: "skim")

## Implementation audited
- scenarios/newyorker_humor_scenario.py — three task modes: `matching` (5-way MCQ, label A-E), `ranking` (2-way A vs B with crowd_winner / ny_winner provenance), `explanation` (open-ended). Prefers official `from_description` packed cartoon description when available; falls back to assembled location/scene/uncanny/entities/questions. Cross-val folds 1-4 supported. Loads HF `jmhessel/newyorker_caption_contest` with task config name. Streaming with cache_dir.
- metrics/newyorker_humor_metric.py — `NewYorkerHumorMetric` (EvaluateInstancesMetric). Matching: emits `accuracy` and `parsed_label_rate` (regex looks for `answer: X` then bare `\b[ABCDE]\b`). Ranking: emits `accuracy`, `accuracy_ny`, `accuracy_crowd` (split by `winner_source`), `parsed_label_rate`. Mirrors upstream answer-letter protocol.
- registry_metrics.yaml (lines 1935-1952): four formula_based metrics on `NewYorkerHumorMetric`.
- registry_inference.yaml (lines 469-476): T=0.0, max_new_tokens=8, n=1, stop=["\n"]. Sourced from `metric_notes/newyorker_humor_eval_metrics_notes.md`. Sensible config for short letter answers.

## Deviations found
- [LOW] B. Prompt fidelity: Prompts for matching/ranking are paraphrased from the dataset card framing; the task instruction "Which of the 5 options is the caption that truly corresponds to the cartoon?" matches the paper's task setup. Not verbatim from paper's published prompt but spirit preserved.
- [MEDIUM] C. Metric fidelity (explanation only): Paper's primary result for explanation is **pairwise human evaluation** of generated explanations vs human gold. We register only matching/ranking metrics; the `explanation` task path produces generations but is not scored by any registered metric (no LLM judge, no automatic comparison). Not a HIGH because the bench primarily reports matching/ranking; explanation is an acknowledged caveat.
- [LOW] A. Dataset source: All three tasks plus 4 CV folds available; loads train/val/test. Matches dataset card exactly.
- [info] D. Generation config: T=0.0, max_new_tokens=8 — appropriate for letter-extraction tasks.

## Notes
For matching and ranking tasks the implementation is clean and paper-comparable: deterministic answer letter extraction, T=0, stop on newline, parse-rate diagnostic, and ranking even splits accuracy by `crowd_winner` vs `ny_winner` source — that mirrors upstream's analysis. The only soft spot is the explanation task: it generates but has no metric — either drop the explanation path from `subsampled_list` configurations, or wire an LLM-judge pairwise protocol against gold explanations.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is_with_caveats
- Now:   Tier 2, high, keep_as_is_with_caveats
- Delta: confirmed
