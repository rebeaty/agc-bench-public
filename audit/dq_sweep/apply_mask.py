"""Apply the 32-cell mask to the released long table + re-derive leaderboard
and per-domain artifacts.

For each (model, dataset) cell flagged BOTH (heuristic + audit on-task < 50%):
  - Set dataset_z to NaN
  - Add a `dq_masked` boolean column
  - Drop those rows from the wide-table aggregation and the per-domain composite
The 32 affected cells are listed in audit/dq_sweep/primary_audit_cross.csv
under status='BOTH'.

Outputs (overwrites in place):
  release_data/long_model_x_dataset.csv          long table with NaN + dq_masked column
  release_data/leaderboard.csv                   leaderboard with refreshed mean_z / median_z / datasets
  release_data/wide_model_x_dataset.csv          wide table re-pivoted from masked long
  analysis/per_domain_jrt.csv              per-(model, domain) z-scores re-aggregated
"""
from pathlib import Path

import numpy as np
import pandas as pd

BUNDLE = Path(__file__).resolve().parents[2]
OUT = BUNDLE / 'audit/dq_sweep'

DOMAINS = ['Brainstorming', 'Figurative Language', 'Humor',
           'Problem Solving', 'STEM', 'Story / Narrative']

# ---- Load BOTH cells from the cross-reference -------------------------
cross = pd.read_csv(OUT / 'primary_audit_cross.csv')
both = cross[cross['status'] == 'BOTH'][['model', 'dataset']]
print(f'BOTH cells to mask: {len(both)}')

# ---- 1. Long table ---------------------------------------------------
long_path = BUNDLE / 'release_data/long_model_x_dataset.csv'
long_bak = BUNDLE / 'release_data/long_model_x_dataset.before_dq_mask.csv'
lmd = pd.read_csv(long_path)
lmd.to_csv(long_bak, index=False)
print(f'Backup: {long_bak}')

both_keys = set(zip(both['model'], both['dataset']))
mask_idx = lmd.apply(lambda r: (r['model'], r['dataset']) in both_keys, axis=1)
print(f'rows masked in long table: {mask_idx.sum()}')

lmd['dq_masked'] = mask_idx
lmd.loc[mask_idx, 'dataset_z'] = np.nan
lmd.to_csv(long_path, index=False)
print(f'Wrote (with NaN + dq_masked column): {long_path}')

# ---- 2. Leaderboard refresh -----------------------------------------
lb_path = BUNDLE / 'release_data/leaderboard.csv'
lb_bak = BUNDLE / 'release_data/leaderboard.before_dq_mask.csv'
lb = pd.read_csv(lb_path)
lb.to_csv(lb_bak, index=False)

# Compute new mean_z / median_z / datasets covered, dropping masked NaNs
md = pd.read_csv(BUNDLE / 'release_data/dataset_metadata.csv')
included_ds = set(md[md['status'] == 'included']['dataset'])
lmd_inc = lmd[lmd['dataset'].isin(included_ds)].dropna(subset=['dataset_z'])

agg = lmd_inc.groupby('model')['dataset_z'].agg(
    mean_z='mean', median_z='median',
    datasets=lambda x: x.notna().sum(),
).reset_index()

# Preserve any other columns from the original lb
extra_cols = [c for c in lb.columns if c not in
              {'model', 'mean_z', 'median_z', 'datasets', 'rank'}]
extras = lb[['model'] + extra_cols]
new_lb = agg.merge(extras, on='model', how='left')
new_lb = new_lb.sort_values('mean_z', ascending=False).reset_index(drop=True)
new_lb['rank'] = np.arange(1, len(new_lb) + 1)
# Re-order to match historical column order roughly
front = ['model', 'datasets']
core = ['mean_z', 'median_z', 'rank']
keep = front + [c for c in lb.columns if c not in set(front + core + ['n_metric_obs', 'n_jrt_cells', 'release_model'])
                and c in new_lb.columns]
keep = [c for c in lb.columns if c in new_lb.columns]
new_lb = new_lb[keep]
new_lb.to_csv(lb_path, index=False)
print(f'Wrote refreshed leaderboard: {lb_path}')

# ---- 3. Wide table re-pivot ----------------------------------------
wide_path = BUNDLE / 'release_data/wide_model_x_dataset.csv'
wide_bak = BUNDLE / 'release_data/wide_model_x_dataset.before_dq_mask.csv'
wide_old = pd.read_csv(wide_path)
wide_old.to_csv(wide_bak, index=False)

wide_new = lmd.pivot_table(
    index='model', columns='dataset', values='dataset_z', aggfunc='first'
).reset_index()
wide_new.to_csv(wide_path, index=False)
print(f'Wrote refreshed wide table: {wide_path}')

# ---- 4. Per-domain JRT composite -----------------------------------
dc = pd.read_csv(BUNDLE / 'analysis/domain_classification.csv')
ds_to_domain = dict(zip(dc['benchmark'], dc['domain']))
lmd_inc['domain'] = lmd_inc['dataset'].map(ds_to_domain)

per_domain = (
    lmd_inc.dropna(subset=['domain'])
    .groupby(['model', 'domain'])['dataset_z'].mean()
    .unstack('domain').reindex(columns=DOMAINS).reset_index()
)
pd_path = BUNDLE / 'analysis/per_domain_jrt.csv'
pd_bak = BUNDLE / 'analysis/per_domain_jrt.before_dq_mask.csv'
pd.read_csv(pd_path).to_csv(pd_bak, index=False)
per_domain.to_csv(pd_path, index=False)
print(f'Wrote refreshed per-domain JRT composite: {pd_path}')

# ---- Primary-result diff -------------------------------------------
old = pd.read_csv(lb_bak)
diff = new_lb[['model', 'mean_z', 'rank']].merge(
    old[['model', 'mean_z', 'rank']].rename(
        columns={'mean_z': 'mean_z_old', 'rank': 'rank_old'}),
    on='model', how='left',
)
diff['Δrank'] = diff['rank'] - diff['rank_old']
moved = diff[diff['Δrank'].abs() >= 1].sort_values('Δrank', key=lambda x: x.abs(),
                                                    ascending=False)
print('\nRank movers (|Δrank| ≥ 1):')
print(moved[['model', 'rank_old', 'rank', 'Δrank',
             'mean_z_old', 'mean_z']].head(20).to_string(index=False))
print(f'\nTotal models with rank shift: {len(moved)}')
print(f'Top 3 unchanged: {(new_lb["rank"].head(3) == old["rank"].head(3)).all()}')
