# ocw_connections fidelity audit

**Tier:** 2
**Confidence:** high
**Recommendation:** patch_with_per_group_metric_aggregation (and T=0.0)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/ocw_connections_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`, `metrics.bert_score_metric.BertScoreMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: Naeini et al., "Large Language Models are Fixated by Red Herrings..." (NeurIPS 2023, arXiv:2306.11167)
- Repo: TaatiTeam/OCW; HF dataset: TaatiTeam/OCW_main
- Task 2: given 4 already-grouped sets of 4 clues, name each group's thematic connection.

## Implementation audited
- scenarios/ocw_connections_scenario.py — system message identifies "Round 3: Connecting Wall on the quiz show Only Connect," instructs naming connections after "Connection:", notes connections may be thematic/linguistic/factual/mathematical. Near-verbatim from upstream baseline notebook.
- run_specs: `BasicGenerationMetric{exact_match, rouge_1, rouge_l}` + `BertScoreMetric`. ROUGE-L is added beyond paper. Paper reports exact-match, ROUGE-1 F1, BERTScore F1.
- 618 walls (62 train / 62 val / 494 test). Sampling cap reports `total_eval_instances=556` (= 494+62, test+validation), randomly capped to 200. 5-shot from train.

## Deviations found
- [HIGH] **Aggregation mismatch**: paper scores connections **individually per group**; scenario emits a single concatenated reference of all 4 group lines, so EM/ROUGE/BERTScore operate on the full 4-line block. Boilerplate ("Group i: words. Connection:") inflates n-gram overlap and diverges from paper's per-group aggregation.
- [LOW] Inference: T=0.7, max_tokens=512 (paper uses T=0 for OpenAI baselines). Minor divergence.
- [LOW] ROUGE-L added beyond paper.

## Notes
Prompt and reference grounding faithful. Tier 2 due to per-group vs whole-block metric aggregation difference (boilerplate inflates scores). Consider patching to compute metrics per group then average.
