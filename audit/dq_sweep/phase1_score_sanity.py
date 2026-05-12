"""DQ sweep — Phase 1: score-side sanity checks.

Scans the score-level artifacts that drive the leaderboard for parser sentinels,
out-of-range ratings, NaN concentrations, and degenerate per-cell distributions
(all-zero, all-max, near-constant). All checks read from the AGC-Bench data
bundle and write per-cell flags to audit/dq_sweep/score_flags.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BUNDLE = Path(__file__).resolve().parents[2]
OUT = BUNDLE / 'audit/dq_sweep'
OUT.mkdir(parents=True, exist_ok=True)


def section(title):
    print(f'\n{"=" * 70}\n{title}\n{"=" * 70}')


# -------------------------------------------------------------------------
section('1. JRT raw ratings (jrt_complete_ratings.parquet)')

jrt = pd.read_parquet(BUNDLE / 'analysis/jrt_complete_ratings.parquet')
print(f'rows: {len(jrt):,}    columns: {jrt.columns.tolist()}')

rating_col = next(
    (c for c in ['rating', 'score', 'judge_score'] if c in jrt.columns), None
)
print(f'rating col: {rating_col}')
if rating_col:
    rs = jrt[rating_col]
    print(f'  total ratings: {len(rs):,}')
    print(f'  null: {rs.isna().sum():,} ({rs.isna().mean() * 100:.2f}%)')
    print(f'  min/median/max: {rs.min()} / {rs.median()} / {rs.max()}')
    print(f'  value counts (top 10):\n{rs.value_counts().head(10)}')

# Per (benchmark, judge) parser sanity
if 'judge' in jrt.columns and 'benchmark' in jrt.columns and rating_col:
    cell_stats = (
        jrt.groupby(['benchmark', 'judge'])[rating_col]
        .agg(['count', 'mean', 'std', 'min', 'max',
              lambda x: x.isna().sum(),
              lambda x: (x == 0).sum()])
        .reset_index()
    )
    cell_stats.columns = ['benchmark', 'judge', 'n', 'mean', 'std',
                          'min', 'max', 'n_null', 'n_zero']
    cell_stats['null_rate'] = cell_stats['n_null'] / cell_stats['n']
    cell_stats['zero_rate'] = cell_stats['n_zero'] / cell_stats['n']
    cell_stats['near_constant'] = cell_stats['std'] < 0.1

    flagged = cell_stats[
        (cell_stats['null_rate'] > 0.05)
        | (cell_stats['zero_rate'] > 0.10)
        | (cell_stats['near_constant'])
    ].sort_values(['null_rate', 'zero_rate'], ascending=False)
    print(f'\n  per-(benchmark, judge) cells: {len(cell_stats):,}')
    print(f'  flagged cells: {len(flagged):,}')
    if len(flagged):
        print(flagged.head(20).to_string(index=False))

    cell_stats.to_csv(OUT / 'jrt_per_cell_sanity.csv', index=False)
    flagged.to_csv(OUT / 'jrt_flagged_cells.csv', index=False)


# -------------------------------------------------------------------------
section('2. Long model x dataset z-scores (long_model_x_dataset.csv)')

lmd = pd.read_csv(BUNDLE / 'release_data/long_model_x_dataset.csv')
print(f'rows: {len(lmd):,}    columns: {lmd.columns.tolist()}')
print(f'dataset_z stats:\n{lmd["dataset_z"].describe()}')
print(f'null in dataset_z: {lmd["dataset_z"].isna().sum():,}'
      f' ({lmd["dataset_z"].isna().mean() * 100:.2f}%)')

# Find extreme z-scores worth a look
extreme = lmd[(lmd['dataset_z'].abs() > 4)].sort_values(
    'dataset_z', key=lambda x: x.abs(), ascending=False
)
print(f'|z| > 4 rows: {len(extreme):,}')
if len(extreme):
    print(extreme.head(15).to_string(index=False))
    extreme.to_csv(OUT / 'long_extreme_z.csv', index=False)


# -------------------------------------------------------------------------
section('3. Per-cell coverage shape (degenerate cells in JRT corrected)')

jc = pd.read_parquet(BUNDLE / 'analysis/jrt_corrected_scores.parquet')
print(f'rows: {len(jc):,}    columns: {jc.columns.tolist()}')

if 'theta' in jc.columns:
    print(f'theta stats: {jc["theta"].describe()}')
    print(f'  null: {jc["theta"].isna().sum():,}')

# Per-(benchmark, model) cell distribution shape
score_col = 'theta' if 'theta' in jc.columns else (
    'score' if 'score' in jc.columns else None
)
key_cols = [c for c in ['benchmark', 'model'] if c in jc.columns]
if score_col and len(key_cols) == 2:
    g = jc.groupby(key_cols)[score_col].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    g['near_constant'] = g['std'] < 0.05
    g['low_n'] = g['count'] < 20
    flagged = g[g['near_constant'] | g['low_n']].sort_values('std')
    print(f'\nper-cell shape: {len(g):,}    flagged: {len(flagged):,}')
    if len(flagged):
        print(flagged.head(15).to_string(index=False))
        flagged.to_csv(OUT / 'jrt_corrected_flagged.csv', index=False)


# -------------------------------------------------------------------------
section('4. Leaderboard sanity (release_data/leaderboard.csv)')

lb = pd.read_csv(BUNDLE / 'release_data/leaderboard.csv')
print(f'rows: {len(lb):,}')
print(f'datasets covered range: {lb["datasets"].min()} ... {lb["datasets"].max()}')
print(f'mean_z range: {lb["mean_z"].min():+.3f} ... {lb["mean_z"].max():+.3f}')
print(f'null mean_z: {lb["mean_z"].isna().sum()}')

low_coverage = lb[lb['datasets'] < 60]
print(f'\nrelease models with <60 datasets: {len(low_coverage)}')
if len(low_coverage):
    print(low_coverage[['rank', 'model', 'datasets', 'mean_z']].to_string(index=False))


# -------------------------------------------------------------------------
section('5. AGC-Judge per-item (release_data/agc_judge_per_item.csv)')

aj = pd.read_csv(BUNDLE / 'release_data/agc_judge_per_item.csv')
print(f'rows: {len(aj):,}    columns: {aj.columns.tolist()}')

for col in ['agc_judge_score', 'jrt_gold']:
    if col in aj.columns:
        print(f'  {col}: min={aj[col].min()}, max={aj[col].max()},'
              f' null={aj[col].isna().sum()},'
              f' zeros={(aj[col] == 0).sum()}')

# residual = predicted - gold
if 'agc_judge_score' in aj.columns and 'jrt_gold' in aj.columns:
    aj['residual'] = aj['agc_judge_score'] - aj['jrt_gold']
    print(f'  residual mean: {aj["residual"].mean():+.3f}')
    print(f'  |residual| > 1.5: {(aj["residual"].abs() > 1.5).sum()} '
          f'({(aj["residual"].abs() > 1.5).mean() * 100:.2f}%)')


print(f'\nWrote per-cell flags to: {OUT}')
