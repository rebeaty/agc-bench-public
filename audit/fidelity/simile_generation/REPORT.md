# simile_generation fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is (Novelty + human eval intentionally out of scope)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/simile_generation_run_specs.py`:
>
> - **MetricSpec(s):** `metrics.simile_generation_metric.SimileGenerationMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: Chakrabarty, Muresan & Peng (2020), "Generating similes effortlessly like a Pro: A Style Transfer Approach for Simile Generation," EMNLP 2020.
- Repo: tuhinjubcse/SimileGeneration-EMNLP2020.

## Implementation audited
- scenarios/simile_generation_scenario.py — instruction-tuned natural-language prompt with bracketed property: "Rewrite the literal sentence below as exactly one simile sentence... Replace the bracketed literal property with a simile vehicle." Paper's SCOPE seq2seq used `<MASK>` token; HELM uses prose instruction (semantically faithful, format differs).
- metrics/simile_generation_metric.py — `SimileGenerationMetric` mirrors released `autoeval.py`: corpus BLEU-1/BLEU-2 with weights (1,0,0,0) and (0,1,0,0) ×100, BERTScore F1 (`roberta-large`, `rescale_with_baseline=True`) over extracted vehicles. Adds HELM-only `vehicle_parsed_rate` health flag.
- 150 literal/simile pairs in test set (`SimileEMNLP.csv`), with two expert references each (Human1, Human2). HELM loads all 150.

## Deviations found
- [LOW] **Format**: prose instruction vs paper's `<MASK>` token (semantically faithful, appropriate for general LLMs).
- [INFO] **Novelty score not implemented**: training-set PROPERTY/VEHICLE artifact not released. Documented in metric_notes.
- [INFO] **Human eval not implemented**: paper's 900-utterance human eval (Creativity, Overall Quality, Relevance1, Relevance2; 1–5 scale; 3 MTurk raters per item) absent. Documented.

## Notes
References are gold VEHICLE phrases from `human_labels.csv` (up to 2 per item), exactly as released SCOPE scorer. No LLM judge. High fidelity for automatic evaluation. Tier 1.
