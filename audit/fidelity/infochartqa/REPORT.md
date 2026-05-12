# infochartqa fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is (optional: pin HF dataset revision for reproducibility)

## Paper / repo audited
- Paper: arXiv:2505.19028
- Repo: github.com/CoolDawnAnt/InfoChartQA
- Dataset: Jietson/InfoChartQA (HF); splits: text, visual_basic, visual_metaphor

## Implementation audited
- scenarios/infochartqa_scenario.py — builds prompt as optional preamble describing chart + cropped figures, dataset `question`, dataset-provided `instructions`. Main chart image plus bbox-derived crops attached as multimodal inputs, mirroring upstream README pipeline (`input_image` + `extra_input_image` from `extra_input_figure_bboxes`).
- metrics/infochartqa_metric.py (`InfoChartQAMetric`) — near line-by-line port of released qtype-aware checker: unit-aware numeric comparison (K/M/B/T/%/Cr), thousands-separator handling, fuzzy string match (SequenceMatcher >=0.8), A-D MCQ parser, ordered/unordered list matching, per-`question_type_id` dispatch covering qtypes 1/2/10-15/30/40-44/50-54/60/61/70-72/80/90/101-113/202/300/1919810-12. Emits single `infochartqa_accuracy` Stat.

## Deviations found
- [INFO] Iterates full HF dataset across all three splits unless `INFOCHARTQA_INSTANCE_LIMIT_HINT` is set. Paper reports 5,642 infographic/plain chart pairs.
- [INFO] No LLM judge. Reference is dataset gold `answer`; scoring deterministic and keyed by `question_type_id` carried in `extra_data`.

## Notes
Tier 1: prompt builds as paper specifies, metric ports upstream evaluator faithfully. Optional improvements: pin HF dataset revision for reproducibility, document `INFOCHARTQA_SUBSET` and limit-hint env vars in registry notes, verify `hhttp` URL-normalization workaround remains necessary against current HF release.
