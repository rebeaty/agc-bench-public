# scimon fidelity audit (re-audit 2026-05-04)

**Tier:** 3
**Confidence:** high
**Recommendation:** patch_with_novelty_judge_or_rebrand (either add a novelty/relevance/technical-depth LLM-judge metric to recover the paper's primary construct, or explicitly rebrand the slice as "SciMON gold-subset reference-based sentence generation")

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/scimon_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`, `metrics.bert_score_metric.BertScoreMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2305.14259 (Wang et al., "SciMON: Scientific Inspiration Machines Optimized for Novelty," ACL 2024) — note: skim (abstract; confirms novelty-optimization is the primary contribution and that the paper relies on human assessment of novelty/relevance/technical depth)
- Repo: https://github.com/EagleW/Scientific-Inspiration-Machines-Optimized-for-Novelty — note: skim (gold_subset URL fetched directly into scenario)

## Implementation audited
- scenarios/scimon_scenario.py — Downloads `gold_subset.zip` from EagleW repo and reads `idea_sentence.json` (194 human-verified ACL-2022 instances, all 5 quality criteria passed). Builds a zero-shot prompt (lines 113–119): "Scientific context: {context}\n\nRelationship: {input}\n\nBased on the above context, write exactly one sentence describing the novel scientific method, dataset, or approach involved in this relationship." Reference is the gold `rel_sent` from the source paper.
- metrics/scimon_metric.py — does not exist (`has_metric_file: false`).
- registry_metrics.yaml (lines 2376–2389): three metrics — `rouge_l`, `bleu_4` (both BasicGenerationMetric), `bert_score` (BertScoreMetric, configured with SciBERT per scenario header).
- registry_inference.yaml (lines 595–601): explicitly pinned — `temperature: 0.2`, `max_tokens: 96`, `num_outputs: 1`, `stop_sequences: ["\n"]`. Tightened for single-sentence generation.

## Deviations found
- [HIGH] C. Metric/scoring fidelity: Paper's primary contribution is novelty-optimization, evaluated via human assessment of novelty / relevance / technical depth. AGC evaluates only with reference-based similarity metrics (BLEU/ROUGE/BERTScore), which measure overlap with a known prior sentence — the opposite of novelty. Reported numbers cannot be compared to the paper's main claims.
- [MEDIUM] B. Prompt fidelity: Prompt is a HELM adaptation of the repo's "released GPT zero-shot style" but not byte-identical. Defensible, but flagged.
- [LOW] A. Reference choice: Scenario uses `rel_sent` (full sentence) rather than `output` (2–3 word entity); the richer target is defensible for a generation evaluation.
- [info] A. Dataset: 194 instances, matches upstream gold subset exactly.
- [info] D. Inference config explicitly tightened (T=0.2, max_tokens=96, stop on newline) for the single-sentence target — sensible.

## Notes
The slice runs cleanly as a reference-based sentence generation evaluation, but it does not test the paper's novelty-optimization construct at all — and reference-based similarity is structurally orthogonal to novelty (high overlap with the gold paper sentence is what the paper would call "low novelty"). Two acceptable paths: (1) add an LLM-judge annotator that scores generated sentences for novelty / relevance / technical depth using the paper's human-eval rubric, or (2) explicitly rebrand the slice in the paper writeup as "SciMON gold-subset reference-based sentence generation" and stop claiming it reproduces SciMON's primary result.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, patch_with_novelty_judge (or rebrand as reference-based sentence-generation slice)
- Now:   Tier 3, high, patch_with_novelty_judge_or_rebrand
- Delta: regressed (the metric mismatch is HIGH and the construct gap is structural; under "any HIGH → Tier 3" the prior Tier 2 was generous)
