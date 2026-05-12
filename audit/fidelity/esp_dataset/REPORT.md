# esp_dataset fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** medium-high
**Recommendation:** keep_as_is (optional patch: replace prompt with paper's verbatim `"{style}:"` prefix; document COCO-image dependency)

## Paper / repo audited
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Fusing_Pre-Trained_Language_Models_With_Multimodal_Prompts_Through_Reinforcement_Learning_CVPR_2023_paper.html (Yu, Chung et al., CVPR 2023, "Esper") — "skim" (relied on scenario header for prompt format; ESP dataset described as 996 COCO-val images × up to 5 styles)
- Repo: https://github.com/JiwanChung/esper — "skim" (verified raw `data/dataset_v_0_2.json` exists at the master branch URL)

## Implementation audited
- scenarios/esp_dataset_scenario.py — Downloads `dataset_v_0_2.json` from the JiwanChung/esper repo, builds image_id → coco_url map, downloads each COCO image to `output_path/images/`, then emits one Instance per (image, available_style) pair. Multimodal `Input` carries the image plus a text prompt at lines 167–169: `"\nDescribe this image in {style} style. Return only the {style}-style caption."`. Reference is the gold caption (CORRECT_TAG). Supports `ESP_DATASET_INSTANCE_LIMIT` env var for trial runs.
- No metric file (uses HELM built-ins).
- registry_metrics.yaml (line 1045): registers `bleu_4`, `meteor`, `cider` — matches paper's "BLEU-4, METEOR, CIDEr" primary metrics.
- registry_inference.yaml (line 247): `_use_defaults: true`.

## Deviations found
- [LOW] B. Prompt fidelity: paper uses bare `"{style}:"` prefix (per scenario docstring lines 20–24); implementation uses a more verbose instruction `"Describe this image in {style} style. Return only the {style}-style caption."` This is defensible for instruction-tuned VLMs that won't continue from a bare style label, but it is paraphrased.
- [LOW] D. Generation configuration: defaults used; paper does not specify decoding for downstream evaluation.
- [info] A. Dataset/instance source: pulls full annotations JSON from upstream repo — should yield ~996 base images × up to 5 styles, matching paper.
- [info] C. Metric/scoring fidelity: BLEU-4/METEOR/CIDEr match the paper primary metrics; no judge swap.

## Notes
Faithful, low-risk implementation. The only non-cosmetic note is the prompt: switching to verbatim `"{style}:"` would reproduce the paper's exact protocol but might confuse modern instruction-tuned VLMs; the current paraphrase preserves intent. CIDEr requires a per-image reference set — confirm the registered `cider` metric correctly aggregates references per image_id (not per (image, style) instance). Tier 2 (a touch above Tier 1 only because of the prompt paraphrase).

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, medium-high, keep_as_is (optional: verbatim prompt)
- Now:   Tier 2, medium-high, keep_as_is
- Delta: confirmed
