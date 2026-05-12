# `analysis/` — validation and paper-result artifacts

This directory contains the frozen analysis artifacts used by the reproduction
scripts. Most readers only need the files below.

| File | Purpose |
|---|---|
| `leaderboard.csv` | Analysis-side 83-model leaderboard snapshot; `release_data/leaderboard.csv` is the canonical reader-facing leaderboard. |
| `per_domain_jrt.csv` | Per-model, per-domain composites under the released JRT-corrected scoring. |
| `leaderboard_raw_vs_jrt.csv` | Shipped 83-model raw-vs-JRT comparison used by `reproduce_paper_results.sh`. |
| `c_factor_loadings.csv` | Six-domain c-factor loadings and summary values. |
| `jrt_complete_ratings.parquet` | Item-level rating grid for the 24 LLM-judge benchmarks under the planned-missing 2-of-3 design. |
| `jrt_corrected_scores.parquet` | JRT posterior scores used to replace raw LLM-judge cells. |
| `agc_judge_*preds.csv` and `agc_judge_ft_test.parquet` | Held-out and in-distribution AGC-Judge validation artifacts. |
| `intelligence_join.csv` | Model-level intelligence indicators used for correlation checks. |
| `cap_*` | Paired human and LLM Creativity Assessment Platform artifacts. |
| `rebuilt/` | Outputs written by reproduction scripts for comparison against the frozen release artifacts. |

`leaderboard_all_models.csv` is a compatibility filename retained for the
release scripts. In this public bundle it has 83 rows and is used as the raw
leaderboard input for rebuild/comparison scripts.
