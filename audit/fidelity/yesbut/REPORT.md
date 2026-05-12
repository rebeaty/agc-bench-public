# yesbut fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_caveat (Detection + Completion tasks intentionally out of scope; Understanding slice is faithful)

> **Implementation note:** the live wiring evaluated by HELM is set in `run_specs/yesbut_run_specs.py`:
>
> - **MetricSpec(s):** `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`, `metrics.meteor_metric.MeteorMetric`, `metrics.bert_score_metric.BertScoreMetric`
>
> Where this report's deviation list describes a different metric layout (e.g. registry-only references), the run-spec wiring above is what the released runs computed.

## Paper / repo audited
- Paper: https://arxiv.org/abs/2409.13592 (Nandy et al., "YesBut: A High-Quality Annotated Multimodal Dataset for evaluating Satire Comprehension capability of Vision-Language Models," EMNLP 2024) — note: read (abstract; confirms 3 tasks Detection/Understanding/Completion and 1,084 satirical images)
- Repo / dataset: https://huggingface.co/datasets/bansalaman18/yesbut — note: skim

## Implementation audited
- scenarios/yesbut_scenario.py — Loads `bansalaman18/yesbut` HF `train` split. Filters by `difficulty_in_understanding` (default "all"). Saves PIL images as JPEGs to a temp dir, builds multimodal `Input` (image + text "Why is this image funny/satirical?", lines 117–126). Reference = `overall_description` field (the paper's gold explanation). Skips ~9 examples with missing difficulty label. Implements only the **Understanding / WhyFunny** task; Detection (binary) and Completion (image-half matching) are intentionally out of scope.
- metrics/yesbut_metric.py — does not exist (`has_metric_file: false`). Scoring relies on shared metrics.
- registry_metrics.yaml (lines 3080–3097): four reference-based metrics — `rouge_l`, `bleu_4` (BasicGenerationMetric), `meteor` (MeteorMetric), `bert_score` (BertScoreMetric). All four match the paper's automatic-evaluation suite for the Understanding task.
- registry_inference.yaml (lines 728–730): `_use_defaults: true`. Run-spec layer reportedly uses T=0.0, max_tokens=192, stop=["\n\n"] — sensible for short explanations.

## Deviations found
- [MEDIUM] A. Coverage: Only Understanding/WhyFunny implemented (1 of 3 paper tasks); Detection (binary classification) and Completion (image-half matching) are dropped. Documented in scenario docstring.
- [LOW] C. Human evaluation: Paper also reports human ratings on correctness / faithfulness / completeness; AGC has only the automatic metrics (no LLM-judge proxy).
- [LOW] C. BERTScore backbone: implementation likely uses `bert-base-uncased`; paper convention is RoBERTa-large. Documented as a BERTScore configuration choice.
- [LOW] D. Generation config: `_use_defaults: true` at registry; run-spec layer overrides — should be pinned at registry level for transparency.
- [info] B. Prompt fidelity: "Why is this image funny/satirical?" matches the paper's Understanding-task phrasing.
- [info] A. Instance count: ~1,075 satirical images after dropping the 9 unlabeled examples; matches the paper's 1,084 satirical-image inventory.

## Notes
Within the Understanding-task scope, the implementation is faithful — verbatim prompt, paper-aligned metric quartet (BLEU/ROUGE/METEOR/BERTScore), correct gold reference. The two scope reductions (Detection and Completion tasks; human eval) are documented in the scenario docstring. `yesbut_v2` is a separate sibling scenario and out of this audit's scope. Optional improvements: pin BERTScore backbone to RoBERTa-large for paper-comparability, surface the gen-time config in registry_inference.yaml, and consider adding an LLM-judge proxy for the human-eval dimensions.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is (Detection + Completion tasks intentionally out of scope)
- Now:   Tier 2, high, keep_as_is_with_caveat
- Delta: confirmed
