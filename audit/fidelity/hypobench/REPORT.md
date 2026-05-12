# hypobench fidelity audit (re-audit 2026-05-04)

**Tier:** 2
**Confidence:** high
**Recommendation:** keep_as_is_with_synthetic_followup (document HDR/synthetic gap; build separate `hypobench_synthetic` scenario before promoting to Tier 1)

## Paper / repo audited
- Paper: https://arxiv.org/abs/2504.11524 (Liu et al., HypoBench, 2025) - skim
- Repo: https://github.com/ChicagoHAI/hypothesis-generation - skim (config.yaml + datasets sourced live)

## Implementation audited
- scenarios/hypobench_scenario.py - downloads per-task `config.yaml` + `metadata.json` from `HypoBench-datasets/main/real`. Uses released `initialize_zero_shot` system+user templates verbatim with `num_hypotheses` substitution (default 10). 7 of paper's 12 tasks loaded: deceptive_reviews, headline_binary, gptgc_detect, llamagc_detect, dreaddit, persuasive_pairs, retweet (lines 36-44). Held-out IND/OOD data paths threaded into `extra_data` for the inference annotator.
- metrics/hypobench_inference_metric.py (HypoBenchInferenceMetric) - reports IND/OOD accuracy, F1, parsed-label rate, hypothesis count, parsed-hypothesis rate -- mirrors paper's two-stage utility pipeline (generate hypotheses -> apply in held-out classification).
- llm_judge/hypobench_inference_annotator.py - judge layer that runs hypothesis-conditioned inference on held-out IND/OOD data.
- registry_metrics.yaml: 6+ formula_based metrics all wired to `metrics.hypobench_inference_metric.HypoBenchInferenceMetric`.
- registry_inference.yaml: T=0.0, max_new_tokens=1024, n=1, do_sample=false (greedy) -- consistent with deterministic hypothesis generation followed by deterministic inference.

## Deviations found
- [MEDIUM] A. Dataset/instance source: 5 synthetic tasks omitted (HDR -- Hypothesis Discovery Rate slice explicitly out of scope per metric notes). Real-world slice only.
- [MEDIUM] C. Metric/scoring fidelity: paper's qualitative ratings (novelty, plausibility, clarity) are not reproduced. Only utility (downstream IND/OOD accuracy + F1) is captured.
- [LOW] A. One generation instance per task (7 total) plus held-out IND/OOD example counts; capped via env vars for smoke runs -- limits statistical resolution per task but matches paper's "one generation, many evaluation" design.
- [info] B. Prompts pulled live from upstream config.yaml (`initialize_zero_shot`) -- verbatim.
- [info] D. T=0, max=1024, greedy -- aligns with deterministic generation + inference.

## Notes
Real-world slice faithfully implements the paper's primary utility evaluation; synthetic HDR slice and subjective ratings are explicit, documented omissions. Tier 2 reflects scope reduction, not fidelity error. Recommendation: surface the HDR gap on the leaderboard, build a separate `hypobench_synthetic` scenario implementing semantic matching against latent rules before promoting this benchmark to Tier 1, and consider adding a judge-rated qualitative annotator (novelty/plausibility/clarity) as a complementary metric.

## Compared to prior audit (2026-04-25)
- Prior: Tier 2, high, keep_as_is_with_synthetic_followup
- Now:   Tier 2, high, keep_as_is_with_synthetic_followup
- Delta: confirmed
