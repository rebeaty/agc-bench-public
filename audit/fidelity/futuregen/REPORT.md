# futuregen fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_caveats (optional patch: add LLM-judge metrics for novelty + feasibility to recover the paper's primary construct)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/futuregen_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`, `metrics.bert_score_metric.BertScoreMetric`, `metrics.futuregen_similarity_metric.FutureGenSimilarityMetric`
> - **ScenarioSpec args:** `prompt_style=prompt_style`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2503.16561 (FutureGen: RAG-based future work generation) — note: read (abstract; sufficient to confirm LLM-judge for novelty/feasibility/hallucination is the primary evaluation)
- Repo: https://github.com/IbrahimAlAzhar/FutureWorkGeneration — note: skim (prompts cited directly from notebook 5 & 6 in scenario header)

## Implementation audited
- scenarios/futuregen_scenario.py — Two prompt styles: `PROMPT_TOP3` (Abstract + Introduction + Conclusion, lines 63–69) and `PROMPT_SHORT` / "all_sections" (lines 71–74), both lifted verbatim from the cited notebooks with the 100-word output constraint preserved. Loads `iaadlab/FutureGen` NeurIPS CSV from HF (`df_neurips_future_work_dataset.csv`), 278 papers (2021–2022). Reference is `future_work_combined` (author future work + OpenReview reviewer suggestions, joined). Default `prompt_style="top3"`.
- metrics: `metrics/futuregen_similarity_metric.py` + `metrics/bert_score_metric.py` (referenced in registry; not separately re-read here).
- registry_metrics.yaml (lines 1150–1171): five reference-based metrics — `bleu_4`, `rouge_l` (both basic_metrics), `bert_score` (BertScoreMetric), `jaccard_similarity`, `cosine_similarity` (FutureGenSimilarityMetric). **No LLM-judge metric registered.**
- registry_inference.yaml (lines 275–277): `_use_defaults: true`.

## Deviations found
- [MEDIUM] C. Metric/scoring fidelity: Paper's primary LLM-as-judge tripartite evaluation (novelty 0–10, binary feasibility, hallucination check) is not implemented. Registered metrics are the auxiliary reference-based suite (BLEU/ROUGE/BERT/Jaccard/cosine) only.
- [MEDIUM] A. Scope: RAG retrieval pipeline (the paper's primary contribution) is intentionally omitted. Documented in scenario docstring as a scope reduction; the non-RAG top-3-sections condition is the paper's reported best non-retrieval baseline.
- [LOW] A. Subset: ACL papers excluded due to missing ground-truth columns; NeurIPS subset (278) matches a documented slice of the paper's data.
- [LOW] D. Generation config: `_use_defaults: true`; paper uses GPT-3.5/GPT-4o-mini with unspecified decoding — acceptable.
- [info] B. Prompt fidelity: prompts copied verbatim from notebooks 5 and 6, including the 100-word constraint and the explicit section list.

## Notes
The scenario itself is faithful — verbatim prompts, correct dataset subset, correct ground-truth construction (combined author + OpenReview future work). The deliberate scope reduction (no RAG, no judge) is documented and defensible. The fidelity gap relative to the paper's primary is the missing LLM-judge construct (novelty/feasibility); registering a single LLM-judge metric with a faithful rubric prompt would let AGC report numbers comparable to the paper's main table. Auxiliary reference-based metrics are appropriate to keep as secondary signals.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is_with_caveats (optional: add LLM-judge for novelty + feasibility)
- Now:   Tier 2, high, keep_as_is_with_caveats
- Delta: confirmed
