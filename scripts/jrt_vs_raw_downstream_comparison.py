"""How do downstream paper findings change when we replace raw canonical
scores with JRT-corrected theta for the 24 LLM-judge cells?

Compares:
  1. Primary AGC composite leaderboard (raw vs JRT-corrected)
  2. Per-domain rankings under the 6-domain taxonomy (raw vs JRT,
     within the same partition so the two are like-for-like)
  3. Intelligence correlations (AGC composite x intelligence_index_aa,
     LSA, math, coding, MMLU-Pro, GPQA-Diamond)

For cells outside the 24-cell JRT universe, we keep the raw canonical
score. For (cell, model) pairs inside the 24-cell universe but not in the
JRT design (model not in PM1-50), we also keep the raw score so that no
model loses dataset coverage. Each (cell, model) is re-z-scored within
cell using JRT-corrected theta where available and the original z otherwise.

Reads (from bundle-relative paths):
  - analysis/leaderboard_all_models.csv      (release raw leaderboard snapshot)
  - analysis/per_domain_composite_raw.csv    (raw per-(model, domain) composite)
  - analysis/domain_classification.csv       (6-domain partition)
  - analysis/intelligence_join.csv           (per-model intelligence indicators)
  - release_data/long_model_x_dataset.csv      (raw per-(model, dataset) z)
  - analysis/jrt_corrected_scores.parquet    (JRT per-rating posteriors)

Writes (to analysis/):
  - leaderboard_raw_vs_jrt_full.csv
  - per_domain_v3_jrt.csv
  - per_domain_consistency.csv
  - intel_corr_raw_vs_jrt.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / 'analysis'
REBUILT = ANALYSIS / 'rebuilt'
REBUILT.mkdir(parents=True, exist_ok=True)
SP = REPO / 'release_data'


def _z(s):
    return (s - s.mean()) / s.std() if s.std() > 0 else s * 0


def _norm_model(s: str) -> str:
    """Underscore form back to slash form (intel file uses underscores)."""
    if not isinstance(s, str):
        return s
    return s.replace('_', '/', 1) if '_' in s and '/' not in s else s


def main():
    raw_lb = pd.read_csv(ANALYSIS / 'leaderboard_all_models.csv')
    print(f'Raw leaderboard: {len(raw_lb)} models, mean_z range '
          f'[{raw_lb["mean_z"].min():.3f}, {raw_lb["mean_z"].max():.3f}]')

    domain_v3 = pd.read_csv(ANALYSIS / 'domain_classification.csv')
    print(f'Paper-primary 6-domain taxonomy: {len(domain_v3)} benches')

    intel = pd.read_csv(ANALYSIS / 'intelligence_join.csv')
    intel['model'] = intel['model'].apply(_norm_model)
    print(f'Intelligence join: {len(intel)} models with intel index')
    print(f'  intel cols available: {[c for c in intel.columns if c not in {"model","datasets","n_metric_obs","mean_z","median_z","rank","agc_rank","aa_match_name","aa_match_conf"}][:8]}')

    # ----- Long-form raw per (model, dataset) -----
    long_path = SP / 'long_model_x_dataset.csv'
    print(f'\nLoading long-form per-(model, dataset): {long_path}')
    raw_long = pd.read_csv(long_path)
    if 'mean_z' not in raw_long.columns and 'dataset_z' in raw_long.columns:
        raw_long = raw_long.rename(columns={'dataset_z': 'mean_z'})
    print(f'  {len(raw_long):,} rows, cols: {raw_long.columns.tolist()}')
    raw_long['z_source'] = 'raw'

    # ----- JRT-corrected per (bench, model) -----
    jrt = pd.read_parquet(ANALYSIS / 'jrt_corrected_scores.parquet')
    jrt_per_model = (jrt.groupby(['benchmark', 'metric', 'model'])
                     .agg(jrt_mean=('score', 'mean'))
                     .reset_index())
    jrt_bench = (jrt_per_model.groupby(['benchmark', 'model'])
                 .agg(jrt_mean=('jrt_mean', 'mean'))
                 .reset_index()
                 .rename(columns={'benchmark': 'dataset'}))
    jrt_bench['mean_z'] = jrt_bench.groupby('dataset')['jrt_mean'].transform(_z)
    jrt_bench['z_source'] = 'jrt'
    jrt_cells = set(jrt_bench['dataset'].unique())
    print(f'  JRT per (bench, model): {len(jrt_bench):,} rows for {len(jrt_cells)} cells')

    # ----- Replace raw z with JRT z where available, else keep raw -----
    raw_long_outside = raw_long[~raw_long['dataset'].isin(jrt_cells)].copy()
    raw_long_inside = raw_long[raw_long['dataset'].isin(jrt_cells)].copy()
    inside_keys = set(zip(raw_long_inside['dataset'], raw_long_inside['model']))
    jrt_keys = set(zip(jrt_bench['dataset'], jrt_bench['model']))
    print(f'\nInside 24-cell JRT universe: {len(raw_long_inside):,} raw rows')
    print(f'  JRT keys covering raw rows : {len(inside_keys & jrt_keys):,}')
    print(f'  raw rows w/o JRT (kept raw): {len(inside_keys - jrt_keys):,}')

    # Keep raw for (model, cell) inside JRT universe but missing from JRT
    keep_raw_inside = raw_long_inside[
        ~raw_long_inside.apply(lambda r: (r['dataset'], r['model']) in jrt_keys, axis=1)
    ].copy()
    new_long = pd.concat([
        raw_long_outside[['dataset', 'model', 'mean_z', 'z_source']],
        keep_raw_inside[['dataset', 'model', 'mean_z', 'z_source']],
        jrt_bench[['dataset', 'model', 'mean_z', 'z_source']],
    ], ignore_index=True)
    n_jrt = (new_long['z_source'] == 'jrt').sum()
    n_raw = (new_long['z_source'] == 'raw').sum()
    print(f'  combined long-form: {len(new_long):,} rows ({n_jrt:,} JRT, {n_raw:,} raw)')

    # ----- New per-model composite -----
    new_lb = (new_long.groupby('model')
              .agg(n_datasets_new=('dataset', 'nunique'),
                   n_obs_new=('mean_z', 'count'),
                   mean_z_new=('mean_z', 'mean'),
                   median_z_new=('mean_z', 'median'))
              .reset_index())
    new_lb['rank_new'] = new_lb['mean_z_new'].rank(ascending=False, method='min')

    # Coverage diff vs raw.
    common_models = set(new_lb['model']) & set(raw_lb['model'])
    raw_only = set(raw_lb['model']) - set(new_lb['model'])
    print(f'\nNew leaderboard: {len(new_lb)} models | shared with raw: {len(common_models)} | raw-only dropped: {len(raw_only)}')

    # =========== ANALYSIS 1: composite leaderboard old vs new ===========
    print('\n========== PRIMARY LEADERBOARD: raw vs JRT ==========')
    cmp = raw_lb.merge(new_lb, on='model', how='inner')
    cmp['rank_delta'] = cmp['rank_new'] - cmp['rank']
    rho_lb, _ = spearmanr(cmp['mean_z'], cmp['mean_z_new'])
    rho_rank, _ = spearmanr(cmp['rank'], cmp['rank_new'])
    r_lb, _ = pearsonr(cmp['mean_z'], cmp['mean_z_new'])
    print(f'  Spearman(raw mean_z, jrt mean_z) = {rho_lb:+.4f}  (n={len(cmp)})')
    print(f'  Pearson (raw mean_z, jrt mean_z) = {r_lb:+.4f}')
    print(f'  Spearman(rank_raw, rank_new)    = {rho_rank:+.4f}')

    top5_old = set(cmp.nsmallest(5, 'rank')['model'])
    top5_new = set(cmp.nsmallest(5, 'rank_new')['model'])
    bot5_old = set(cmp.nlargest(5, 'rank')['model'])
    bot5_new = set(cmp.nlargest(5, 'rank_new')['model'])
    print(f'  top-5 overlap (old vs new): {len(top5_old & top5_new)}/5')
    print(f'  bot-5 overlap: {len(bot5_old & bot5_new)}/5')
    print(f'  top-5 raw : {sorted(top5_old)}')
    print(f'  top-5 jrt : {sorted(top5_new)}')

    big_movers = cmp[cmp['rank_delta'].abs() >= 5].sort_values('rank_delta')
    print(f'\n  Models with |rank shift| >= 5 (n={len(big_movers)}):')
    for _, r in big_movers.head(8).iterrows():
        d = int(r['rank_delta']); arrow = 'UP' if d < 0 else 'DOWN'
        print(f'    {r["model"][:42]:42s} raw#{int(r["rank"]):>3} -> jrt#{int(r["rank_new"]):>3} ({arrow}{abs(d)})')
    for _, r in big_movers.tail(8).iterrows():
        d = int(r['rank_delta']); arrow = 'UP' if d < 0 else 'DOWN'
        print(f'    {r["model"][:42]:42s} raw#{int(r["rank"]):>3} -> jrt#{int(r["rank_new"]):>3} ({arrow}{abs(d)})')

    cmp.to_csv(REBUILT / 'leaderboard_raw_vs_jrt_full.csv', index=False)
    print(f'\n  wrote {ANALYSIS/"leaderboard_raw_vs_jrt_full.csv"}')

    # =========== ANALYSIS 2: per-domain rankings under taxonomy ===========
    print('\n========== PER-DOMAIN (6-domain) RANKINGS ==========')
    # Same partition for raw and new; re-aggregate from long-form so it's apples-to-apples.
    raw_long_for_dom = raw_long.merge(domain_v3[['benchmark', 'domain']],
                                       left_on='dataset', right_on='benchmark', how='inner')
    raw_dom = (raw_long_for_dom.groupby(['model', 'domain'])
               .agg(z=('mean_z', 'mean'))
               .reset_index()
               .pivot(index='model', columns='domain', values='z'))

    new_long_for_dom = new_long.merge(domain_v3[['benchmark', 'domain']],
                                       left_on='dataset', right_on='benchmark', how='inner')
    new_dom = (new_long_for_dom.groupby(['model', 'domain'])
               .agg(z=('mean_z', 'mean'))
               .reset_index()
               .pivot(index='model', columns='domain', values='z'))

    new_dom.to_csv(REBUILT / 'per_domain_v3_jrt.csv')
    print(f'  wrote {ANALYSIS/"per_domain_v3_jrt.csv"}')

    rows_dom = []
    print(f'\n  {"domain":<25} {"n":>4} {"rho(raw,jrt)":>14} {"r(raw,jrt)":>12}')
    for d in sorted(set(raw_dom.columns) & set(new_dom.columns)):
        common = raw_dom.index.intersection(new_dom.index)
        joined = pd.DataFrame({
            'raw': raw_dom.loc[common, d],
            'jrt': new_dom.loc[common, d],
        }).dropna()
        if len(joined) < 10:
            continue
        rho_d, _ = spearmanr(joined['raw'], joined['jrt'])
        r_d, _ = pearsonr(joined['raw'], joined['jrt'])
        print(f'  {d:<25} {len(joined):>4} {rho_d:>+14.4f} {r_d:>+12.4f}')
        rows_dom.append({'domain': d, 'n': len(joined), 'rho_raw_jrt': rho_d, 'r_raw_jrt': r_d})
    pd.DataFrame(rows_dom).to_csv(REBUILT / 'per_domain_consistency.csv', index=False)

    # Top-5 per domain comparison (raw vs JRT)
    print(f'\n  Top-5 per domain (raw vs jrt):')
    for d in sorted(set(raw_dom.columns) & set(new_dom.columns)):
        top_raw = raw_dom[d].nlargest(5).index.tolist()
        top_jrt = new_dom[d].nlargest(5).index.tolist()
        overlap = len(set(top_raw) & set(top_jrt))
        print(f'    {d:<25} overlap {overlap}/5')

    # =========== ANALYSIS 3: intelligence correlations ===========
    print('\n========== AGC x INTELLIGENCE CORRELATIONS ==========')
    intel_index_cols = [
        c for c in ['intelligence_index_aa', 'math_index_aa', 'coding_index_aa',
                    'mmlu_pro_aa', 'gpqa_diamond', 'lsa_accuracy', 'aa_index']
        if c in intel.columns
    ]
    print(f'  intel cols matched: {intel_index_cols}')
    if not intel_index_cols:
        print('  WARN: no intel index columns found; dump first 30 columns:')
        print('  ', intel.columns.tolist()[:30])
    cmp_intel = cmp.merge(intel[['model'] + intel_index_cols], on='model', how='inner')
    print(f'  models with intel data: {len(cmp_intel)}')
    if len(cmp_intel) == 0:
        print(f'  raw_lb sample models: {raw_lb["model"].head(3).tolist()}')
        print(f'  intel sample models : {intel["model"].head(3).tolist()}')

    print(f'\n  {"index":<26} {"n":>4} {"raw rho":>9} {"raw r":>7} {"jrt rho":>9} {"jrt r":>7} {"d_rho":>8}')
    rows = []
    for col in intel_index_cols:
        sub = cmp_intel.dropna(subset=[col, 'mean_z', 'mean_z_new'])
        if len(sub) < 10:
            continue
        rho_raw, _ = spearmanr(sub[col], sub['mean_z'])
        r_raw, _ = pearsonr(sub[col], sub['mean_z'])
        rho_jrt, _ = spearmanr(sub[col], sub['mean_z_new'])
        r_jrt, _ = pearsonr(sub[col], sub['mean_z_new'])
        delta = rho_jrt - rho_raw
        print(f'  {col:<26} {len(sub):>4} {rho_raw:>+9.4f} {r_raw:>+7.4f} {rho_jrt:>+9.4f} {r_jrt:>+7.4f} {delta:>+8.4f}')
        rows.append({'index': col, 'n': len(sub), 'rho_raw': rho_raw, 'r_raw': r_raw,
                     'rho_jrt': rho_jrt, 'r_jrt': r_jrt, 'delta_rho': delta})
    pd.DataFrame(rows).to_csv(REBUILT / 'intel_corr_raw_vs_jrt.csv', index=False)
    print(f'\n  wrote {ANALYSIS/"intel_corr_raw_vs_jrt.csv"}')


if __name__ == '__main__':
    main()
