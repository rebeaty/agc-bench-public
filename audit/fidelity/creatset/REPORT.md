# creatset fidelity audit (re-audit 2026-05-04)

**Tier:** 3
**Confidence:** high
**Recommendation:** discuss / patch_with_pairwise_winrate (drop BLEU-4 + ROUGE-L, replace with judge-based pairwise win-rate against a baseline `gen_resp_*` response; populate `references`)

## Paper / repo audited
- Paper: https://arxiv.org/abs/2505.19236 (CrEval / CreataSet, ICLR 2026) — note: skim (abstract only)
- Repo: https://github.com/Aman-4-Real/CrEval — note: unread (no fetch attempted; abstract was sufficient to confirm pairwise protocol)

## Implementation audited
- scenarios/creatset_scenario.py — Loads `Aman/CreataSet` HF dataset file `CreataSet-test_with_labeling_400.jsonl` via `hf_hub_download`. Passes `instruction` field verbatim as input (no template wrapping; preserves Chinese). `references=[]`. Reference output, 4 candidate `gen_resp_*` responses, Bradley-Terry `avg_score`, and 30-annotator `labeling` matrix are stashed in `extra_data` but unused by registered metrics. 400 instances (50 × 8 domains) when `domain="all"`.
- metrics/creatset_metric.py — does not exist (`has_metric_file: false`).
- registry_metrics.yaml (lines 696–712): three metrics — `bleu_4`, `rouge_l` (both `compute_reference_metrics`), and `llm_judge_creativity` (`openai/gpt-4o`, T=0.0, max_new_tokens=512, `judge_prompt: null`).
- registry_inference.yaml (lines 170–172): `_use_defaults: true`.
- registry_master.yaml (lines 293–301): `has_reference_target: true` (but scenario sets `references=[]` — internally inconsistent).

## Deviations found
- [HIGH] C. Metric/scoring fidelity: Paper's evaluator (CrEval) is a fine-tuned pairwise judge producing win-rate against a baseline response. Registry instead encodes single-response BLEU-4 + ROUGE-L + generic creativity LLM-judge — none match CrEval's pairwise win-rate protocol.
- [HIGH] C. Reference plumbing: Scenario sets `references=[]`, so reference-based BLEU/ROUGE receive no targets and will silently produce 0 or NaN.
- [HIGH] C. Unused calibration data: Reference `output`, 4 candidate responses, and 30-annotator pairwise labels are stashed in `extra_data` but no registered metric consumes them.
- [HIGH] C. Judge mismatch: Generic `gpt-4o` judge vs. paper's fine-tuned CrEval evaluator. The `judge_prompt` is null too, so even the LLM-judge dimension is under-specified.
- [MEDIUM] A. registry_master.yaml declares `has_reference_target: true` but scenario emits empty references — internal inconsistency.
- [info] B. Prompt fidelity: instruction passed verbatim — paper has no explicit prompt template, so this is faithful.
- [info] D. Generation config: `_use_defaults: true`; paper does not specify gen-time config.

## Notes
Tier 3 stands. BLEU-4 and ROUGE-L are inappropriate for open-ended Chinese creative writing and were not used by the paper. Without rework, reported scores will not reflect the paper's construct. Recommended fix: (1) drop BLEU-4 / ROUGE-L; (2) implement pairwise win-rate metric (judge sees model response + a fixed baseline pulled from `gen_resp_*` or `output`); (3) populate `references` with the chosen baseline; (4) optionally support the released CrEval checkpoint as an alternate judge backend; (5) reconcile `has_reference_target` flag.

## Compared to prior audit (2026-04-25)
- Prior: Tier 3, high, discuss (substantial reimplementation needed)
- Now:   Tier 3, high, discuss / patch_with_pairwise_winrate
- Delta: confirmed
