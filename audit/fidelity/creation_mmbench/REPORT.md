# Audit: creation_mmbench

**Paper:** https://arxiv.org/abs/2503.14478 (ICCV 2025)
**Repo:** https://github.com/open-compass/Creation-MMBench
**Scenario:** `scenarios/creation_mmbench_scenario.py`
**Metric:** `metrics/creation_mmbench_metric.py`

## Prompt Fidelity
The dataset's `question` field already contains the full role/background/instruction/requirement template described in paper Section 3.2; the scenario passes it through verbatim alongside 1–9 images per item. No prompt rewriting. Multimodal payload built via `MultimediaObject` with images preceding text, consistent with MLLM evaluation practice. **Match.**

## Metric Fidelity
Paper specifies dual GPT-4o judging producing Visual Factuality Score (VFS, 1–10) and pairwise Reward (-100 to +100), with position-swap to reduce bias. The metric wrapper emits `creation_mmbench_vfs`, `creation_mmbench_reward`, `creation_mmbench_dual_eval_gap`, and `judge_parse_rate` read from an annotator (`creation_mmbench_judge`). Names match the paper's two primary scores plus a dual-eval diagnostic. However, the annotator implementation/prompt is not in this repo, and `registry_metrics.yaml` declares only a generic `llm_judge_quality` with `judge_prompt: null`, understating what the wrapper consumes. **Largely faithful; registry under-specifies.**

## Instance Count
Scenario loads HF `test` split (765 cases = 15 × 51 tasks). `task_filter="all"` default reproduces paper. Subset filters are optional. **Match.**

## Judge / Reference
Judge: GPT-4o, temp 0.0 — matches paper. Per-instance `criteria` dict (subjective + groundtruth alignment) attached as a second `Reference`, preserving the rubric. Reference answer prefers `ground_truth` (356) else `reference_answer_by_gpt4o` (746), matching released dataset. **Match.**

## Verdict
- **Tier:** 1 (faithful)
- **Confidence:** Medium-High. Scenario and metric names align with paper; main gap is the dual-judge annotator code/prompt is not visible here, so VFS/Reward computation cannot be verified end-to-end from this repo alone.
- **Recommendation:** Populate `judge_prompt` in `registry_metrics.yaml` and ensure the `creation_mmbench_judge` annotator (with position-swap dual evaluation) ships with the benchmark. Otherwise ready.
