# fann_or_flop fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** medium-high
**Recommendation:** patch_with_metric_registration (register `FannOrFlopMetric` in `registry_metrics.yaml` so chrF++/BERTScore/judge stats actually emit; current registry only points to BasicGenerationMetric)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/fann_or_flop_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`, `metrics.fann_or_flop_metric.FannOrFlopMetric`
> - **AnnotatorSpec(s):** `llm_judge.fann_or_flop_annotator.FannOrFlopAnnotator`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2505.18152 ("Fann or Flop") — abstract only via arXiv (note: "skim")
- Repo: https://github.com/mbzuai-oryx/FannOrFlop — README confirms ~7k poem-explanation pairs, BLEU/chrF++/BERTScore + GPT-4o judge (faithfulness/fluency). Primary metrics: BERTScore (0.6410 GPT-4o) and chrF++ (0.2882). (note: "read")

## Implementation audited
- scenarios/fann_or_flop_scenario.py — loads `dataset.json` from HF `omkarthawakar/FannOrFlop`. Optional era/genre filters (default "all"). Arabic instruction prompt asks for verse-by-verse explanation (literal meaning, thematic depth, cultural context, literary devices, expressive style — matches paper's human-rubric dimensions). Reference is `raw_explanation`; structured `explanation` list passed in `extra_data` for verse alignment.
- metrics/fann_or_flop_metric.py — `FannOrFlopMetric` (Metric) emits 8 verse-aware judge stats (gold_verse_count, generated_verse_count, aligned_verse_count, verse_parse_rate, judge_parse_rate, faithfulness_score, fluency_score, overall_score) plus chrF++ (sacrebleu sentence_chrf word_order=2) and AraBERT BERTScore (`aubmindlab/bert-base-arabertv02`, layer 12). Reads judge output from `request_state.annotations["fann_or_flop_poem_judge"]`.
- registry_metrics.yaml (lines 1060-1085): six entries — exact_match, quasi_exact_match, f1_score, rouge_l, bleu_1, bleu_4 — all routed to **`BasicGenerationMetric`**. The custom `FannOrFlopMetric` and its annotator `fann_or_flop_poem_judge` are **NOT registered**.
- registry_inference.yaml (lines 251-253): `_use_defaults: true`.

## Deviations found
- [HIGH] C. Metric/scoring fidelity: `FannOrFlopMetric` exists with chrF++/BERTScore/judge-stats wiring, but `registry_metrics.yaml` lists only generic BasicGenerationMetric entries (BLEU/F1/ROUGE/exact_match). The paper's primary metrics (chrF++ and AraBERT BERTScore) and the GPT-4o faithfulness/fluency judge are not actually computed. Fix: add `helm_class: metrics.fann_or_flop_metric.FannOrFlopMetric` entries for `chrfpp`, `bert_score`, `faithfulness_score`, `fluency_score`, `overall_score`, `*verse*_rate`; register the `fann_or_flop_poem_judge` annotator.
- [MEDIUM] C. Metric registry/code mismatch: registry advertises exact_match/quasi_exact_match/f1_score, none of which are reasonable for free-form Arabic verse explanations. These should be removed.
- [LOW] B. Prompt fidelity: The Arabic prompt text closely tracks the paper's evaluation rubric dimensions; not verbatim from any prompt file we could locate, but spirit preserved.
- [info] A. Dataset source: Loads full HF dataset (~7k pairs); paper-comparable.
- [info] D. Generation config: defaults; paper does not specify decoder hyperparams.

## Notes
The custom metric file is well-implemented (proper threading lock for BERTScorer, AraBERT model, sacrebleu chrF++) but it is dead code as-registered. Maintainer needs to (a) wire each emitted Stat to a registry entry pointing at FannOrFlopMetric, (b) drop the irrelevant exact/quasi/F1 entries, and (c) register the `fann_or_flop_poem_judge` annotator with a verbatim judge prompt (faithfulness + fluency + overall, 0-5 or whatever the paper uses). After that fix, this becomes a clean Tier 2.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, medium-high, patch_with_metric_registration
- Now:   Tier 2, medium-high, patch_with_metric_registration
- Delta: confirmed
