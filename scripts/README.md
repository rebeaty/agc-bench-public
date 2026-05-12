# `scripts/` — analysis and rebuild pipeline

The scripts here regenerate the released analysis artifacts and main-text
primary results from the bundled inputs in `release_data/`, `analysis/`, and
`data/registry/`. They reproduce the paper's main-text claims on the
83 strict-coverage models shipped in this bundle. Two values reproduce to within
rounding rather than exact match against the paper text — the c-factor
eigenvalue / variance and the raw-vs-JRT ρ — because the released bundle
recomputes from the frozen public release files;
the differences are documented in `release_data/SCORING_NOTES.md`. Appendix-
specific numerical claims (App D / G / H / I / J / K / L / M) are checked
end-to-end by `reproduce_appendix.sh` at the bundle root, which prints
paper-stated vs computed for each check with PASS / WITHIN_TOL / FAIL
tags (the script prints the current check count in its summary).
Run scripts from the bundle root with `python3 scripts/<name>.py`; rows below
note any external inputs or environment variables needed.

## Quick reproduce

```bash
# From the bundle root:
bash reproduce_paper_results.sh
```

This runs the three scripts that produce the primary results (c-factor, JRT
correlation, intelligence correlations) and prints the recomputed values
alongside what the paper reports.

## Script index

| Script | Reads | Writes | Paper artifact |
|---|---|---|---|
| `fit_jrt_grm.py` | `analysis/jrt_complete_ratings.parquet` (raw 3-judge ratings under planned-missing 2-of-3 design) | `analysis/rebuilt/jrt_theta.parquet`, `jrt_rater_severity.csv`, `jrt_fit_summary.csv`, `jrt_elbo_traces.parquet` | §3.5 / App. JRT — Bayesian GRM fit (NumPyro SVI, Normal(0,0.3) prior on log_alpha, 800 steps per cell). Produces the latent ability theta that becomes the JRT-corrected score. Requires `jax`, `numpyro`. |
| `build_jrt_artifacts.py` | `release_data/{leaderboard,long_model_x_dataset,lsa_per_model}.csv`, `analysis/{jrt_corrected_scores.parquet, domain_classification.csv}` | `analysis/rebuilt/*` (leaderboard_jrt, c_factor_loadings_jrt, leaderboard_with_lsa) | Fig 1, Fig 2, Fig 3, Table 1 |
| `cfactor_raw_vs_jrt.py` | `release_data/long_model_x_dataset.csv`, `analysis/{jrt_corrected_scores.parquet, domain_classification.csv}` | (prints only) | §4.3 c-factor primary result (eigenvalue 4.89, alpha 0.96, 81.5% variance under the released DQ-masked release set) |
| `jrt_vs_raw_downstream_comparison.py` | `analysis/{leaderboard_all_models, per_domain_composite_raw, domain_classification, intelligence_join}.csv`, `release_data/long_model_x_dataset.csv`, `analysis/jrt_corrected_scores.parquet` | `analysis/rebuilt/*` (leaderboard_raw_vs_jrt_full, per_domain_v3_jrt, per_domain_consistency, intel_corr_raw_vs_jrt) | §4.4 intelligence correlations on the release set. `reproduce_paper_results.sh` reads `analysis/leaderboard_raw_vs_jrt.csv` and prints the shipped 83-model JRT-vs-raw value ρ ≈ 0.94; see `release_data/SCORING_NOTES.md` for the paper-text rounding note. |
| `cap_human_vs_llm_generality.py` | `analysis/cap_human_per_task.csv` | `analysis/rebuilt/cap_generality_human_vs_llm.csv` | §4.7 LLM alpha=0.64 vs Human alpha=0.42 |
| `lookup_dataset_licenses.py` | `data/registry/registry_master.yaml`, `release_data/long_model_x_dataset.csv` | `audit/dataset_licenses.csv` | Appendix license disclosures |
| `generate_croissant.py` | `data/registry/registry_master.yaml`, `scripts/_domain_mapping.py`, `scenarios/` | `croissant/agc_bench.json`, `croissant/per_benchmark/*.json` | Croissant 1.1 manifests |
| `build_release_data.py` | `release_data/`, `analysis/` | `release_data_rebuilt_<date>/`, sibling .zip | Rebuilds the released data from the bundled inputs |
| `build_dataset_raw_distribution.py` | `analysis/agc_long_metric_z.parquet`, `analysis/jrt_corrected_scores.parquet`, `release_data/{leaderboard,long_model_x_dataset,dataset_metadata}.csv` | `release_data/dataset_raw_distribution.csv` | Builds the frozen raw-score cohort distribution used to place newly evaluated models on the released leaderboard scale |
| `build_benchmark_catalog_497.py` | `curation/catalog/source/*` | `release_data/benchmark_catalog_497.csv` | Rebuilds the 497-row benchmark catalog audit file |
| `consolidate_generations.py` | Retained HELM run outputs under `benchmark_output/runs/` | `generations/` | Builds the release-subset model-generation parquet corpus hosted on Hugging Face |
| `extract_judge_prompts.py` | `data/registry/registry_metrics.yaml`, `llm_judge/`, `run_specs/`, optional `AGC_PROMPT_SOURCE_ROOT` | `audit/judge_prompts/` | Rebuilds the LLM-judge prompt appendix. Without `AGC_PROMPT_SOURCE_ROOT`, the checked-in appendix is the complete release artifact; set `AGC_ALLOW_PARTIAL_PROMPT_REGEN=1` only for a partial local extraction. |
| `integrate_new_model.py` | `benchmark_output/runs/<suite>/*/stats.json`, `release_data/{dataset_metadata,dataset_raw_distribution,leaderboard}.csv`, `data/registry/registry_metrics.yaml` | `analysis/scored/<model>/{per_dataset_z.csv,per_domain.csv,leaderboard_line.json,missing.csv}` | Post-HELM aggregation for a live new-model run |
| `score_orwig_with_agc_judge.py` | `data/external/orwig_2024/cw_data.csv` (third-party, not redistributed), HF model `agcbench-2026/AGC-Judge` | `analysis/rebuilt/orwig_agc_judge_promptC.parquet` | Appendix Orwig OOD validation (rho=0.65 vs human, rho=0.80 vs LLM-judge gold) |
| `score_ismayilzada_with_agc_judge.py` | HuggingFace dataset `mismayil/creative_story_generation_dataset` (auto-fetched), HF model `agcbench-2026/AGC-Judge` | `analysis/rebuilt/ismayilzada_agc_judge_scores.parquet` | Independent external-validation check |

## AGC-Judge inference scripts (Orwig, Ismayilzada)

Both `score_orwig_*.py` and `score_ismayilzada_*.py` send response text to the
released AGC-Judge model (Qwen3-30B-A3B-Instruct LoRA, hosted at
[https://huggingface.co/agcbench-2026/AGC-Judge](https://huggingface.co/agcbench-2026/AGC-Judge)).
The scripts default to the Together inference API when used for live external
validation (set `TOGETHER_API_KEY`, override the model id with
`AGC_JUDGE_MODEL`). To run
locally with the released LoRA weights, swap the `call()` function for a
`transformers + peft` generate against the merged adapter. A GPU with at
least 80 GB memory is recommended for the 30B base.

For the Orwig script, the third-party Orwig 2024 corpus must be obtained
separately and placed at `data/external/orwig_2024/cw_data.csv`. The
Ismayilzada dataset is auto-fetched from HuggingFace.

## Support modules

`_domain_mapping.py` is a sibling module imported by `generate_croissant.py`.
It contains the per-dataset domain assignments used to organize the manifest
output and is not intended to be run directly.

## Output convention

Scripts write derived outputs to `analysis/rebuilt/` so paper-cited artifacts
in `analysis/` and `release_data/` stay easy to compare. Users can diff
`analysis/rebuilt/<file>` against the released equivalent to verify
reproduction.
