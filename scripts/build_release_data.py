"""Rebuild the JRT-corrected release data from the bundled
inputs (JRT-corrected ratings + raw scores + domain partition).

The checked-in release_data/ directory contains the frozen v1 artifacts.
Re-running this script writes a fresh copy to release_data_rebuilt_<timestamp>/
next to the released directory, so the canonical release files stay untouched.

Reads (from bundle-relative paths):
  - release_data/long_model_x_dataset.csv             (raw per-(model, dataset) z)
  - release_data/leaderboard.csv                      (release leaderboard, release_model flag)
  - release_data/{cap_human_data, lsa_per_model, lsa_methods_note}
  - release_data/dataset_metadata.csv                 (release metadata)
  - release_data/dataset_raw_distribution.csv         (optional raw cohort distribution)
  - analysis/canonical_metrics.yaml                 (canonical metric per cell, exclusion set)
  - analysis/leaderboard_all_models.csv             (release raw leaderboard snapshot)
  - analysis/jrt_corrected_scores.parquet           (JRT per-rating posteriors)
  - analysis/jrt_complete_ratings.parquet           (raw rater grid behind JRT)
  - analysis/domain_classification.csv              (6-domain partition)
  - analysis/agc_judge_held_out_preds.csv           (AGC-Judge held-out predictions)
  - analysis/agc_judge_ft_test.parquet              (held-out test grid)

Writes (to release_data_rebuilt_<timestamp>/):
  - long_model_x_dataset.csv
  - model_dataset_scores.csv
  - wide_model_x_dataset.csv
  - leaderboard.csv
  - dataset_metadata.csv
  - dataset_raw_distribution.csv (if present in release_data/)
  - cap_human_data.csv
  - lsa_per_model.csv
  - lsa_methods_note.md
  - agc_judge_per_item.csv
  - README.md
  + a sibling .zip archive
"""
from __future__ import annotations
from pathlib import Path
import time, shutil
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / 'analysis'
SP = REPO / 'release_data'

ts = time.strftime('%Y%m%d')
OUT = REPO / f'release_data_rebuilt_{ts}'
OUT.mkdir(parents=True, exist_ok=True)

# Restrict to the 83-model release set so the downstream IRT pipeline operates
# on a consistent-coverage sample (no missing cells to patch around) and the
# model set matches the paper's primary analyses.
STRICT_COVERAGE_ONLY = True


def main():
    # --- 1. Load excluded-benches list ---
    canon = yaml.safe_load((ANALYSIS / 'canonical_metrics.yaml').read_text())
    excluded = set(canon.get('skip', []))
    jrt_cells = set(canon['canonical_metrics'].keys())
    print(f'Excluded benches (drop from sharepack): {len(excluded)}')
    print(f'JRT cells (replace raw with JRT-corrected): {len(jrt_cells)}')

    # --- 1b. Build release-model set from the canonical leaderboard ---
    src_lb = pd.read_csv(ANALYSIS / 'leaderboard_all_models.csv')
    release_models = set(src_lb[src_lb['datasets'] == src_lb['datasets'].max()]['model'].tolist())
    print(f'Release model set (full {src_lb["datasets"].max()}-bench coverage): {len(release_models)} models')

    # --- 2. Load + filter long-form (model, dataset) ---
    prev_long = pd.read_csv(SP / 'long_model_x_dataset.csv')
    print(f'\nReleased long: {len(prev_long):,} rows, {prev_long["dataset"].nunique()} datasets')

    # Drop excluded benches
    new_long = prev_long[~prev_long['dataset'].isin(excluded)].copy()
    n_dropped_b = len(prev_long) - len(new_long)
    print(f'  dropped {n_dropped_b:,} rows from {len(excluded)} excluded benches')

    # Restrict to the release model set
    if STRICT_COVERAGE_ONLY:
        before = len(new_long)
        new_long = new_long[new_long['model'].isin(release_models)].copy()
        print(f'  restricted to release models: {before:,} -> {len(new_long):,} rows ({new_long["model"].nunique()} models)')

    # Mark source
    new_long['score_source'] = 'raw'
    new_long['n_score_sources'] = 1

    # --- 3. Replace JRT-cell scores with JRT-corrected ---
    jrt = pd.read_parquet(ANALYSIS / 'jrt_corrected_scores.parquet')
    pm1 = pd.read_parquet(ANALYSIS / 'jrt_complete_ratings.parquet').dropna(subset=['score'])

    if STRICT_COVERAGE_ONLY:
        jrt = jrt[jrt['model'].isin(release_models)]
        pm1 = pm1[pm1['model'].isin(release_models)]

    jrt_per = (jrt.groupby(['benchmark', 'metric', 'model'])
               .agg(jrt_mean=('score', 'mean'))
               .reset_index())
    jrt_bench = jrt_per.groupby(['benchmark', 'model']).agg(
        jrt_mean=('jrt_mean', 'mean')).reset_index()

    rater_count = (pm1.groupby(['benchmark', 'model'])['rater'].nunique()
                   .reset_index().rename(columns={'rater': 'n_score_sources_jrt'}))
    jrt_bench = jrt_bench.merge(rater_count, left_on=['benchmark', 'model'],
                                 right_on=['benchmark', 'model'], how='left')

    def _z(s):
        return (s - s.mean()) / s.std() if s.std() > 0 else s * 0
    jrt_bench['jrt_z'] = jrt_bench.groupby('benchmark')['jrt_mean'].transform(_z)
    jrt_bench = jrt_bench.rename(columns={'benchmark': 'dataset'})

    # Capture the full data-quality mask key set BEFORE we split the released
    # long table by JRT vs non-JRT. JRT-corrected rows are about to be
    # regenerated from the GRM fit, so their dq_masked state would otherwise
    # be lost. Held in `masked_keys` and re-applied to new_long_full below.
    masked_keys = set()
    if 'dq_masked' in new_long.columns:
        masked_keys = set(
            (r.model, r.dataset)
            for r in new_long[new_long['dq_masked'] == True].itertuples(index=False)
        )
        print(f'  captured {len(masked_keys)} dq_masked keys from released long')

    new_long_no_jrt = new_long[~new_long['dataset'].isin(jrt_cells)].copy()
    # NOTE: do NOT re-z within the strict-coverage subset. The released long's
    # dataset_z values were computed across the canonical model set and
    # re-z-scoring inside the 83-model slice produces values that drift
    # from canonical (max delta ~1.8 on 30% of raw cells). Preserve them.
    jrt_rows = jrt_bench[['dataset', 'model', 'jrt_z', 'n_score_sources_jrt']].rename(
        columns={'jrt_z': 'dataset_z', 'n_score_sources_jrt': 'n_score_sources'})
    jrt_rows['score_source'] = 'jrt'
    jrt_rows['n_score_sources'] = jrt_rows['n_score_sources'].fillna(1).astype(int)
    if 'dq_masked' in new_long_no_jrt.columns:
        jrt_rows['dq_masked'] = False
    new_long_full = pd.concat([new_long_no_jrt, jrt_rows[new_long_no_jrt.columns]],
                              ignore_index=True)

    # Re-apply the full mask (covers both raw cells preserved from the released
    # long and freshly-regenerated JRT cells) so rebuilt dataset_z exactly
    # matches canonical (NaN for masked cells).
    if masked_keys and 'dq_masked' in new_long_full.columns:
        m = new_long_full.apply(
            lambda r: (r['model'], r['dataset']) in masked_keys, axis=1)
        new_long_full.loc[m, 'dataset_z'] = float('nan')
        new_long_full.loc[m, 'dq_masked'] = True
        print(f'  applied data-quality mask to {int(m.sum())} cells')

    print(f'\nNew long: {len(new_long_full):,} rows')
    print(f'  raw: {(new_long_full["score_source"]=="raw").sum():,}')
    print(f'  jrt: {(new_long_full["score_source"]=="jrt").sum():,}')
    print(f'  datasets: {new_long_full["dataset"].nunique()}')

    new_long_full.to_csv(OUT / 'long_model_x_dataset.csv', index=False)
    print(f'  wrote {OUT/"long_model_x_dataset.csv"}')
    viewer_cols = ['model', 'dataset', 'dataset_z', 'score_source', 'dq_masked']
    new_long_full[viewer_cols].to_csv(OUT / 'model_dataset_scores.csv', index=False)
    print(f'  wrote {OUT/"model_dataset_scores.csv"}')

    # --- 4. Build new wide format (for orientation) ---
    wide = new_long_full.pivot_table(index='model', columns='dataset',
                                       values='dataset_z', aggfunc='first')
    wide.to_csv(OUT / 'wide_model_x_dataset.csv')
    print(f'  wrote {OUT/"wide_model_x_dataset.csv"}')

    # --- 5. Rebuild leaderboard (per-model mean of dataset_z) ---
    # `datasets` counts datasets where the cell carries a canonical score
    # (post-mask); `n_metric_obs` counts every (model, dataset) cell that
    # ran, masked or not. This semantic matches the released leaderboard.
    grouped = new_long_full.groupby('model')
    lb = pd.DataFrame({
        'model': grouped['dataset_z'].count().index,
        'datasets': grouped['dataset_z'].count().values,
        'n_metric_obs': grouped['dataset'].nunique().values,
        'mean_z': grouped['dataset_z'].mean().values,
        'median_z': grouped['dataset_z'].median().values,
        'n_jrt_cells': grouped['score_source'].apply(lambda s: (s == 'jrt').sum()).values,
    })
    # Force int dtype on integer columns so the CSV serializes as `67` not
    # `67.0` (matches the canonical leaderboard byte layout).
    for col in ('datasets', 'n_metric_obs', 'n_jrt_cells'):
        lb[col] = lb[col].astype('int64')
    lb['rank'] = lb['mean_z'].rank(ascending=False, method='min').astype('int64')
    lb = lb.sort_values('rank').reset_index(drop=True)
    prev_lb = pd.read_csv(SP / 'leaderboard.csv')
    release_flag = dict(zip(prev_lb['model'], prev_lb['release_model']))
    lb['release_model'] = lb['model'].map(lambda m: release_flag.get(m, ''))
    lb.to_csv(OUT / 'leaderboard.csv', index=False)
    print(f'\nLeaderboard: {len(lb)} models, top: {lb.iloc[0]["model"]} (mean_z={lb.iloc[0]["mean_z"]:.3f})')

    # --- 6. Updated dataset_metadata with 6-domain labels and release counts ---
    # Apply the 6-domain remap only to INCLUDED datasets. Excluded entries
    # keep their as-released domain so the rebuilt metadata matches canonical
    # exactly (e.g. macgyver retains its STEM tag rather than picking up the
    # the 'Problem Solving' label its scenario would otherwise resolve to).
    domain_v3 = pd.read_csv(ANALYSIS / 'domain_classification.csv')
    prev_meta = pd.read_csv(SP / 'dataset_metadata.csv').copy()
    domain_map = dict(zip(domain_v3['benchmark'], domain_v3['domain']))
    is_included = prev_meta['status'] == 'included'
    prev_meta.loc[is_included, 'domain'] = prev_meta.loc[is_included, 'dataset'].map(
        lambda d: domain_map.get(d, '')
    ).where(lambda s: s != '', prev_meta.loc[is_included, 'domain'])
    release_counts = new_long_full.groupby('dataset')['model'].nunique()
    prev_meta['n_models'] = (
        prev_meta['dataset'].map(release_counts).fillna(0).astype('int64')
    )
    prev_meta['jrt_corrected'] = prev_meta['dataset'].isin(jrt_cells)
    prev_meta.to_csv(OUT / 'dataset_metadata.csv', index=False)
    print(f'  wrote {OUT/"dataset_metadata.csv"} ({len(prev_meta)} datasets)')

    # --- 7. Copy unchanged supplementary files from the released sharepack ---
    shutil.copy(SP / 'cap_human_data.csv', OUT / 'cap_human_data.csv')
    shutil.copy(SP / 'lsa_per_model.csv', OUT / 'lsa_per_model.csv')
    shutil.copy(SP / 'lsa_methods_note.md', OUT / 'lsa_methods_note.md')
    raw_dist = SP / 'dataset_raw_distribution.csv'
    if raw_dist.exists():
        shutil.copy(raw_dist, OUT / 'dataset_raw_distribution.csv')
        print(f'  copied {OUT/"dataset_raw_distribution.csv"}')

    # --- 8. AGC-Judge per-item predictions for the 24 JRT cells ---
    print(f'\nBuilding AGC-Judge per-item file...')
    eval_test = ANALYSIS / 'agc_judge_held_out_preds.csv'
    test_parq = ANALYSIS / 'agc_judge_ft_test.parquet'
    if eval_test.exists() and test_parq.exists():
        preds = pd.read_csv(eval_test)
        parq = pd.read_parquet(test_parq).reset_index(drop=True)
        # Predictions are positionally aligned with the test parquet by
        # construction (same row order from the eval pipeline). Guard against
        # silent drift if either side has been edited.
        if len(preds) != len(parq):
            raise RuntimeError(
                f'agc_judge_held_out_preds.csv ({len(preds)} rows) and '
                f'agc_judge_ft_test.parquet ({len(parq)} rows) have diverged. '
                f'They must stay positionally aligned; drop the same rows from '
                f'both whenever cleaning failed cells (see audit/dq_sweep/).'
            )
        merged = parq.copy()
        merged['agc_judge_score'] = preds['pred'].values
        merged['jrt_gold'] = preds['gold'].values
        out_cols = ['benchmark', 'model', 'item_id', 'metric',
                    'jrt_gold', 'agc_judge_score', 'family']
        merged = merged[out_cols]
        merged.to_csv(OUT / 'agc_judge_per_item.csv', index=False)
        print(f'  wrote {OUT/"agc_judge_per_item.csv"} ({len(merged):,} rows)')

    # --- 9. README ---
    readme = f"""# AGC Release Data (rebuilt)

**Date:** {ts}
**Scope:** AGC-Bench v1 **{len(release_models)}-model** strict-coverage release set
with JRT-corrected LLM-judge ratings plus a per-item AGC-Judge prediction file.

This artifact is rebuilt from the bundled inputs by `scripts/build_release_data.py`.
The released `release_data/` directory contains a frozen v1 copy that this rebuild
matches.

---

## Files

| File | Description |
|---|---|
| `model_dataset_scores.csv` | Compact viewer-facing score table |
| `long_model_x_dataset.csv` | Long-form (model, dataset, dataset_z, score_source, n_score_sources, dq_masked when present) |
| `wide_model_x_dataset.csv` | Wide (model x dataset) z-score matrix |
| `leaderboard.csv` | Per-model AGC composite (mean of cell z) + release_model flag + n_jrt_cells |
| `dataset_metadata.csv` | Per-dataset domain (6-domain), release-set n_models, jrt_corrected flag |
| `dataset_raw_distribution.csv` | Per-dataset raw-score cohort distribution for new-model leaderboard insertion, if included in the source release_data directory |
| `agc_judge_per_item.csv` | Per-(model, item) AGC-Judge predictions vs JRT gold (24 cells) |
| `cap_human_data.csv` | Paired human + LLM CAP composite |
| `lsa_per_model.csv` | Letter-string analogy accuracy per model |
| `lsa_methods_note.md` | LSA methodology |

---

## Methodology notes

**JRT correction ({len(jrt_cells)} LLM-judge cells):**
- Three judges under a planned-missing 2-of-3 item-level design
- Bayesian Graded Response Model with `Normal(0, 0.3)` prior on log_alpha
- SVI fit, single seed, 800 steps per cell, all cells converged
- Per-judge severity (β-mean): gemini-3-flash near zero (neutral),
  gpt-4.1-mini around -0.9 (lenient), grok-4.1-fast around -0.5 (moderately lenient)
- Pairwise inter-judge Spearman: 0.83 to 0.89 across the three pairs

**Domain taxonomy (6-domain):**
- Story / Narrative, STEM, Figurative Language, Problem Solving, Humor, Brainstorming
- Plus orthogonal Generation / Evaluation task-type label
- Cohen's kappa about 0.85 across 3 LLM-rater consensus

**AGC-Judge model:**
- Qwen3-30B-A3B-Instruct (MoE, 3B active params), LoRA r=16, alpha=32
- Trained on JRT-corrected training rows
- Item-level Spearman vs JRT gold: 0.94 (in-distribution), 0.94 (10 unseen models),
  0.83 (3 unseen benchmarks)
- Composite leaderboard reproduction: rho = 0.97 across all splits
- Released at https://huggingface.co/agcbench-2026/AGC-Judge
"""
    (OUT / 'README.md').write_text(readme)
    print(f'  wrote {OUT/"README.md"}')

    # --- 10. Zip the rebuilt sharepack ---
    suffix = '_strict_coverage' if STRICT_COVERAGE_ONLY else ''
    zip_path = REPO / f'release_data_rebuilt_{ts}{suffix}.zip'
    shutil.make_archive(str(zip_path).replace('.zip', ''), 'zip', OUT)
    print(f'\nWrote zip: {zip_path}')
    print(f'\nRelease data ready at: {OUT}')


if __name__ == '__main__':
    main()
