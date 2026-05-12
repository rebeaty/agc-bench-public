# scar fidelity audit

**Tier:** 1
**Confidence:** high
**Recommendation:** keep_as_is (optional: add multi-template sweep)

## Paper / repo audited
- Paper: Yuan et al., "Beneath Surface Similarity: LLMs Make Reasonable Scientific Analogies after Structure Abduction" (EMNLP Findings 2023). https://aclanthology.org/2023.findings-emnlp.160
- Repo: https://github.com/siyuyuan/scar (data: `release/system_analogy_en.json`; prompts: `template.txt`)

## Implementation audited
- scenarios/scar_scenario.py — reproduces paper's "Instruction 1" template from `template.txt` verbatim, including bracketed list-of-pairs output spec and post-context restatement. Inputs: `system_a/b`, both `_domain`, both `_background`, plus union of items appearing in gold mappings (`items_a`, `items_b`). Zero-shot, no demonstrations.
- metrics/scar_metric.py — `SCARMetric`: balanced-bracket slicing, code-fence stripping, `ast.literal_eval`, regex fallback, with case/quote/whitespace/punctuation normalization. Reports `scar_concept_accuracy` (recall over gold pairs), `scar_system_accuracy` (exact set match), `scar_precision`, `scar_recall`, `scar_f1`, `scar_parsed_mapping_rate`.
- 400 analogies across 13 domains. Reads every line of `system_analogy_en.json` into TEST split; no subsampling.

## Deviations found
- [LOW] **Single prompt family vs paper's multi-template sweep** (paper sweeps additional templates/CoT variants).
- [LOW] **`scar_concept_accuracy` is recall proxy** rather than exact reimplementation of authors' eval script.

## Notes
No LLM judge. Reference is gold `mappings` field, stored as Reference text and `extra_data["gold_mappings"]`. Deterministic decoding (T=0, max_tokens=512, n=1). Matches paper's two primary metrics (Concept Acc., System Acc.); precision/F1/parse-rate are sensible HELM additions. Tier 1.
