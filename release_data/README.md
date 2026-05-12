# AGC-Bench Release Data

These files are the frozen source for the paper's primary AGC-Bench
(Artificial General Creativity Benchmark) analyses: the **83-model** release
set evaluated on the **67-dataset** primary leaderboard. They support
leaderboard reproduction, new-model score integration, and checks of the
paper's reported numeric claims.

## Release Scope

Before data-quality masking, every release model had a canonical score on at
least 65 of 67 primary datasets. The long table contains 5,510 rows; after
excluding masked rows, per-model scored-dataset counts are: 31 models at 67,
43 at 66, 5 at 65, and 4 below 65. The four sub-65 cases reflect the mask,
not missing evaluation coverage:
`google/gemma-2-27b-it` (54), `morph/morph-v3-fast` (57),
`meta-llama/llama-3.2-3b-instruct` (63), `z-ai/glm-4.5v` (64). The matching
score table holds **5,478 unmasked scored cells out of 5,561 possible
(model, dataset) cells**. The 83-cell difference consists of 51 cells without
canonical scores, most concentrated on `irfl` (50 missing cells), and
32 rows masked by the release data-quality sweep
(`dq_masked = True` in the long table; see
[../audit/dq_sweep/REPORT.md](../audit/dq_sweep/REPORT.md)).

This release data is the frozen source for the paper's primary analyses.
A reproduction pipeline is at
[../scripts/build_release_data.py](../scripts/build_release_data.py).

v1.0.1 adds `benchmark_catalog_497.csv` and documents the Hugging Face
generation corpus. The frozen score tables above are unchanged.

The Hugging Face Dataset Viewer defaults to `model_dataset_scores.csv`, a
compact copy of the canonical long table that keeps only the columns most useful
for browsing: model, dataset, z-score, score source, and data-quality mask.

For cross-benchmark scoring conventions (the JRT three-vendor panel for the
24 LLM-judge benchmarks, the Gemini embedding-backend routing, reproducibility
scope), see [SCORING_NOTES.md](SCORING_NOTES.md). For per-benchmark
paper-vs-implementation deviations, see
[../audit/fidelity/INDEX.md](../audit/fidelity/INDEX.md).

---

## Files

| File | Description |
|---|---|
| `model_dataset_scores.csv` | Compact viewer-facing score table: model, dataset, dataset_z, score_source, dq_masked. |
| `long_model_x_dataset.csv` | Canonical long-form score table: model, dataset, dataset_z, score_source, n_score_sources, dq_masked. `n_score_sources` is aggregate-level bookkeeping: raw rows use one canonical score stream; JRT rows record how many judge identities contributed to that model-dataset aggregate. It is not the item-level planned-missing rater count. Cells with `dq_masked = True` carry NaN dataset_z and are excluded from primary aggregates. |
| `wide_model_x_dataset.csv` | Wide (model × dataset) z-score matrix; masked cells appear as empty. |
| `leaderboard.csv` | Per-model AGC composite (mean of dataset_z over included, non-masked cells), median, dataset coverage count, and rank. |
| `dataset_raw_distribution.csv` | Frozen per-dataset raw-score mean and standard deviation for the 83-model release cohort; used by `scripts/integrate_new_model.py` to place a newly evaluated model on the released leaderboard scale. |
| `dataset_metadata.csv` | Per-dataset domain (six-domain partition), release-set `n_models`, `jrt_corrected` flag, status (included / excluded). |
| `benchmark_catalog_497.csv` | Final 497-row benchmark catalog used before benchmark selection and implementation. |
| `agc_judge_per_item.csv` | Per-(model, item) AGC-Judge predictions vs JRT gold (24 LLM-judge cells). |
| `cap_human_data.csv` | Paired human and LLM CAP composite. |
| `lsa_per_model.csv` | Letter-string analogy accuracy per model. |
| `lsa_methods_note.md` | LSA methodology. |

## Data-quality mask

A release data-quality sweep cross-referenced two independent signals:
(i) heuristic flags on raw responses (empty / refusal / repetitive output)
and (ii) the on-task LLM-judge audit reported in the paper. Cells flagged by
both signals (32 of ~5,500) are masked at the score level rather than scored
as failed attempts. Cascade-impact analysis showed that the c-factor
(eigenvalue 4.89, α 0.96, 81.5 % variance) and intelligence ρ values
(AA Intelligence ρ = 0.778, MMLU-Pro ρ = 0.629, GPQA ρ = 0.770, HLE ρ = 0.638)
are unchanged to two decimals under any handling of these cells. Top-10
leaderboard is identical; the largest single per-model rank shift is
gemma-2-27b-it moving from 75 to 69 (its 12 fully-empty cells no longer
contribute extreme negative z). Full report at
[../audit/dq_sweep/cascade_impact.md](../audit/dq_sweep/cascade_impact.md).

---

## Methodology

**JRT correction (24 LLM-judge cells):**
- Three cross-vendor judges (gemini-3-flash, grok-4.1-fast, gpt-4.1-mini)
  rated item-level responses under a planned-missing 2-of-3 design (~91k units)
- Bayesian Graded Response Model with Normal(0, 0.3) prior on log alpha
- SVI fit, single seed, 800 steps per cell, all 24 cells converged
- Per-judge severity (beta mean): gemini-3-flash near zero (neutral),
  gpt-4.1-mini around -0.9 (lenient), grok-4.1-fast around -0.5
  (moderately lenient)
- Pairwise inter-judge Spearman: 0.83 to 0.89 across the three pairs

**Domain taxonomy (six-domain partition):**
- Primary domains are Brainstorming, Problem Solving, STEM, Story /
  Narrative, Figurative Language, and Humor.
- `dataset_metadata.csv` records each dataset's domain and included/excluded
  status for the 67-dataset primary leaderboard; `../analysis/domain_classification.csv`
  contains the 3-LLM consensus classification panel used for domain analyses.
- Cohen's kappa about 0.85 across the 3-LLM-rater consensus

**AGC-Judge model:**
- Qwen3-30B-A3B-Instruct base (MoE, 3B active params), LoRA r=16, alpha=32
- Trained on 48,299 JRT-corrected training rows
- Item-level Spearman vs JRT gold: 0.94 (in-distribution), 0.94 (held-out
  models), 0.83 (held-out benchmarks)
- Composite leaderboard reproduction: rho = 0.97 across all splits
- Released at https://huggingface.co/agcbench-2026/AGC-Judge
