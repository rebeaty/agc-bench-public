# c3_crosstalk fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is (document deviations: prompt is a synthesized instruction, not a paper template; human-eval dims absent)

## Paper / repo audited
- Paper: https://arxiv.org/abs/2207.00735 ("C3: Towards Realistic Benchmark for Chinese Comical Crosstalk Generation," ACL 2023 Findings) — abstract only via WebFetch (note: "skim")
- Repo: https://github.com/FreedomIntelligence/crosstalk-generation — README confirms 50-dialogue test set, BLEU-1..4/GLEU/ROUGE-1,2,L/distinct-1,2 metrics. (note: "read")

## Implementation audited
- scenarios/c3_crosstalk_scenario.py — fetches `eval_data/human_eval/data/meta_prompt.json` (10-utterance contexts) and `src/common_data/test_filter_50x20.txt` (50 dialogues × 20 utterances). Uses utterances 10-19 as gold continuation. 50 instances, matches paper. Prompt is an instruction template (paper used seq2seq, not instruction-tuned LLMs, so this is unavoidable).
- metrics/c3_crosstalk_metric.py — `C3CrosstalkMetric` (EvaluateInstancesMetric) computes char-level BLEU-1/2/4 (NLTK corpus_bleu, smoothing.method1), GLEU, ROUGE-1/2/L (own LCS impl), distinct-1/2. Character-level tokenization matches the upstream Chinese convention.
- registry_metrics.yaml (lines 390-427): nine formula_based metrics all routed through `C3CrosstalkMetric`. Matches the paper's automatic metric suite exactly.
- registry_inference.yaml (lines 94-96): `_use_defaults: true` ("not specified in paper or repo").

## Deviations found
- [LOW] B. Prompt fidelity: Paper used seq2seq encoder-decoder models (no LLM prompt). Scenario uses a hand-written Chinese instruction prompt ("以下是一段相声对话的开头，请续写..."). Necessary substitution; does not break comparability of generated continuations under automatic metrics.
- [MEDIUM] C. Metric fidelity: Paper's primary conclusions rely on **human evaluation** (general quality, humor, coherence, ethical-risk) since "machine metrics don't fully assess how well a generation works." We compute only the automatic suite (paper's secondary metrics). Defensible — automatic metrics are what's portable — but means our scores do not equal the paper's primary numbers.
- [LOW] A. Dataset source: 50 instances; matches paper. info.
- [info] D. Generation config: defaults (paper does not specify LLM config).

## Notes
The implementation faithfully ports the C3 automatic metric suite at the character level matching upstream. The scenario correctly uses utterances 0-9 as context and 10-19 as gold. Two acknowledged limits — (1) instruction prompt rather than seq2seq decoding, (2) no human-eval dimensions — are inherent to running this benchmark on instruction-tuned LLMs in an automated pipeline. Worth noting in the paper write-up that primary numbers in C3 were human ratings, not BLEU/ROUGE.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is (document deviations)
- Now:   Tier 2, high, keep_as_is
- Delta: confirmed
