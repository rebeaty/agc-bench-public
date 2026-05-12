# mars fidelity audit (re-audit 2026-05-04)

**Tier:** 3
**Confidence:** medium-high
**Recommendation:** patch_with_quasi_em_or_ranking_protocol (replace exact_match with Hits@1/MRR-style or quasi-EM; current setup is structurally wrong for an analogy KG-completion task evaluated on free-form generation)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/mars_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`, `metrics.bert_score_metric.BertScoreMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2210.00312 ("Multimodal Analogical Reasoning over Knowledge Graphs," ICLR 2023) — abstract only via arXiv (note: "skim")
- Repo: https://github.com/zjunlp/MKG_Analogy — README does not surface the metric explicitly. The MARS task is canonically scored by Hits@1, Hits@3, Hits@10, and MRR over the entity vocabulary (standard MKGE/KG-completion convention). (note: "skim")

## Implementation audited
- scenarios/mars_scenario.py — clones `MKG_Analogy` repo. Loads `MarT/dataset/MARS/{train,dev,test}.json`. Resolves entity IDs to text via `MarKG/entity2text.txt` and relations via `relation2text.txt`. Builds an analogy prompt: "Complete the following analogy: Example A: {h_text} relates to Example B: {t_text} by the relation: {r_text}. Similarly, Question: {q_text} relates to what? Answer:". Optional image mode requires manual download from Google Drive. References = both `answer_id` and `answer_text` tagged CORRECT_TAG.
- metrics/mars_metric.py — does not exist (`has_metric_file: false`).
- registry_metrics.yaml (lines 1666-1671): one metric — `exact_match` via `BasicGenerationMetric`.
- registry_inference.yaml (lines 397-399): `_use_defaults: true`.

## Deviations found
- [HIGH] C. Metric/scoring fidelity: Paper task is analogy completion over a fixed entity vocabulary (canonical KG-completion-style metrics: Hits@1, Hits@3, Hits@10, MRR). We score with **string exact_match** against either entity ID or English description. Free-form generation will rarely produce the literal entity ID and even text matches are brittle (case, qualifiers, "the X" prefixes). This breaks paper-comparability — our numbers will floor near zero relative to paper.
- [MEDIUM] B. Prompt fidelity: Hand-written analogy prompt; paper used embedding/Transformer baselines, not LLM-prompting. The framing is reasonable but unable to constrain output to the entity vocabulary, which is the entire point of KG analogy.
- [LOW] A. Dataset source: All splits loaded; sizes match paper (10641/1020/1362).
- [info] D. Generation config: defaults; but for free-form analogy, low T would help.

## Notes
The implementation is structurally misaligned with the source task. Two possible fixes: (1) **constrained scoring** — present the entity set as candidates and compute Hits@K (rank by likelihood / multi-choice); (2) **quasi-EM with entity-text normalization** — strip articles, lowercase, optionally accept any entity whose `entity2text` matches the generation by substring + extra references for entity aliases. Option (1) is paper-faithful; option (2) is a portable fallback. As-is, MARS results are not paper-comparable and will rank-correlate poorly with reasoning ability.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, patch_with_quasi_em_or_ranking_protocol
- Now:   Tier 3, medium-high, patch_with_quasi_em_or_ranking_protocol
- Delta: regressed (severity escalated to HIGH because exact_match against entity IDs / free-text is structurally broken for this task; prior audit also flagged the fix but classified as Tier 2)
