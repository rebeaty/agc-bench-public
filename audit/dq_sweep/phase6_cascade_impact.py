"""DQ sweep — Phase 6: cascade-impact analysis.

Recompute the primary composite, per-domain composite, c-factor (eigenvalue,
alpha, % variance), and intelligence correlations under three scenarios:
  A. as-is               — current data, current scoring
  B. mask-32-as-missing  — 32 BOTH-flagged cells set to NaN
  C. drop-3-flagged-models — also drop gemma-2-27b-it, morph-v3-fast,
                            tencent_hunyuan-a13b-instruct

Output: dq_sweep/cascade_impact.md with side-by-side comparison.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import FactorAnalysis

BUNDLE = Path(__file__).resolve().parents[2]
OUT = BUNDLE / 'audit/dq_sweep'

# -------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------
lmd = pd.read_csv(BUNDLE / 'release_data/long_model_x_dataset.csv')
dc = pd.read_csv(BUNDLE / 'analysis/domain_classification.csv').rename(
    columns={'benchmark': 'dataset'}
)
md = pd.read_csv(BUNDLE / 'release_data/dataset_metadata.csv')
included_ds = set(md[md['status'] == 'included']['dataset'])
intel = pd.read_csv(BUNDLE / 'analysis/intelligence_join.csv')
intel['model'] = intel['model'].apply(
    lambda s: s.replace('_', '/', 1) if isinstance(s, str) and '_' in s and '/' not in s else s
)

cross = pd.read_csv(OUT / 'primary_audit_cross.csv')
both = cross[cross['status'] == 'BOTH'][['model', 'dataset']].copy()
print(f'BOTH cells loaded: {len(both)}')

DROP_MODELS = {'google/gemma-2-27b-it', 'morph/morph-v3-fast',
               'tencent/hunyuan-a13b-instruct'}
DOMAINS = ['Brainstorming', 'Figurative Language', 'Humor',
           'Problem Solving', 'STEM', 'Story / Narrative']


# -------------------------------------------------------------------------
# Build per-(model, dataset) wide table; only included datasets
# -------------------------------------------------------------------------
lmd_inc = lmd[lmd['dataset'].isin(included_ds)].copy()
print(f'(model, dataset) rows in included set: {len(lmd_inc):,}')

# attach domain
ds_domain = dict(zip(dc['dataset'], dc['domain']))
lmd_inc['domain'] = lmd_inc['dataset'].map(ds_domain)


def recompute(scenario_name, df):
    """Compute leaderboard + per-domain + c-factor + intelligence ρ for df."""
    # Per-model mean z (the primary composite)
    model_mean = df.groupby('model')['dataset_z'].mean()

    # Per-(model, domain) composite — mean of dataset_z within each domain
    pdz = df.dropna(subset=['domain']).groupby(
        ['model', 'domain']
    )['dataset_z'].mean().unstack('domain')
    pdz = pdz.reindex(columns=DOMAINS)

    # c-factor: factor analysis on the model x 6 domain matrix
    M = pdz.dropna()  # drop rows with any missing domain
    n_kept = len(M)
    fa = FactorAnalysis(n_components=1, random_state=0)
    fa.fit(M.values)
    loadings = fa.components_[0]
    # Eigenvalue from correlation of M
    corr = np.corrcoef(M.values.T)
    eigvals, _ = np.linalg.eigh(corr)
    eig1 = eigvals[-1]

    # Cronbach's alpha: k * mean_offdiag / (1 + (k-1) * mean_offdiag)
    k = corr.shape[0]
    triu = corr[np.triu_indices(k, k=1)]
    mean_r = triu.mean()
    alpha = (k * mean_r) / (1 + (k - 1) * mean_r) if mean_r > 0 else float('nan')
    var_pct = eig1 / k * 100

    # Intelligence correlations (composite vs AA + LSA + GPQA)
    composite_df = pd.DataFrame({'model': model_mean.index,
                                 'composite': model_mean.values})
    j = composite_df.merge(intel, on='model', how='left')
    rhos = {}
    for col, label in [
        ('intelligence_index_aa', 'AA Intelligence'),
        ('mmlu_pro_aa', 'MMLU-Pro'),
        ('gpqa_diamond', 'GPQA-Diamond'),
        ('hle', 'HLE'),
    ]:
        if col not in j.columns:
            continue
        sub = j[['composite', col]].dropna()
        if len(sub) < 10:
            continue
        rho, _ = stats.spearmanr(sub['composite'], sub[col])
        r, _ = stats.pearsonr(sub['composite'], sub[col])
        rhos[label] = (rho, r, len(sub))

    # Top-10 leaderboard
    top10 = model_mean.sort_values(ascending=False).head(10)

    return {
        'name': scenario_name,
        'n_models': len(model_mean),
        'n_models_in_cfactor': n_kept,
        'eigenvalue_1': eig1,
        'alpha': alpha,
        'var_pct': var_pct,
        'loadings': dict(zip(DOMAINS, loadings)),
        'intel_rhos': rhos,
        'top10': top10,
        'model_mean': model_mean,
    }


# Scenario A: as-is
A = recompute('A. as-is', lmd_inc)

# Scenario B: mask 32 cells
both_keys = set(zip(both['model'], both['dataset']))
mask_idx = lmd_inc.apply(
    lambda r: (r['model'], r['dataset']) in both_keys, axis=1
)
print(f'\nMasking {mask_idx.sum()} rows for scenario B '
      f'({len(both_keys)} expected from BOTH cells; difference = naming/coverage gaps).')
B_df = lmd_inc.copy()
B_df.loc[mask_idx, 'dataset_z'] = np.nan
B_df = B_df.dropna(subset=['dataset_z'])
B = recompute('B. mask 32 BOTH cells', B_df)

# Scenario C: drop 3 flagged models
C_df = lmd_inc[~lmd_inc['model'].isin(DROP_MODELS)].copy()
print(f'Scenario C: dropping {DROP_MODELS} ({len(lmd_inc) - len(C_df)} rows)')
C = recompute('C. drop 3 flagged models', C_df)


# -------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------
def fmt_intel(rhos):
    return ', '.join(f'{k}: ρ={r[0]:+.3f} (r={r[1]:+.3f}, n={r[2]})'
                     for k, r in rhos.items())


report = ['# Cascade-impact analysis — three DQ scenarios\n',
          'Recomputed under three handlings of the 32 cells flagged by both '
          'heuristic and audit signals.\n']

for sc in (A, B, C):
    report.append(f'\n## {sc["name"]}\n')
    report.append(f'- **Models in composite**: {sc["n_models"]}')
    report.append(f'- **Models in c-factor (complete domains)**: '
                  f'{sc["n_models_in_cfactor"]}')
    report.append(f'- **First eigenvalue**: {sc["eigenvalue_1"]:.3f}')
    report.append(f'- **Cronbach α**: {sc["alpha"]:.3f}')
    report.append(f'- **% variance (eig1 / 6)**: {sc["var_pct"]:.1f}')
    loadings_str = ', '.join(f'{d}: {sc["loadings"][d]:+.2f}' for d in DOMAINS)
    report.append(f'- **Loadings**: {loadings_str}')
    report.append(f'- **Intelligence ρ**: {fmt_intel(sc["intel_rhos"])}')
    report.append(f'\n**Top 10 leaderboard:**')
    for i, (m, v) in enumerate(sc['top10'].items(), 1):
        report.append(f'  {i:>2}. {m:<40} {v:+.3f}')

# Direct deltas table
report.append('\n## Primary-number deltas\n')
report.append(
    f'| metric | A as-is | B mask-32 | Δ | C drop-3 | Δ |'
)
report.append('|---|---|---|---|---|---|')
report.append(f'| eigenvalue_1 | {A["eigenvalue_1"]:.3f} | '
              f'{B["eigenvalue_1"]:.3f} | '
              f'{B["eigenvalue_1"] - A["eigenvalue_1"]:+.3f} | '
              f'{C["eigenvalue_1"]:.3f} | '
              f'{C["eigenvalue_1"] - A["eigenvalue_1"]:+.3f} |')
report.append(f'| alpha | {A["alpha"]:.3f} | {B["alpha"]:.3f} | '
              f'{B["alpha"] - A["alpha"]:+.3f} | {C["alpha"]:.3f} | '
              f'{C["alpha"] - A["alpha"]:+.3f} |')
report.append(f'| var % | {A["var_pct"]:.1f} | {B["var_pct"]:.1f} | '
              f'{B["var_pct"] - A["var_pct"]:+.1f} | '
              f'{C["var_pct"]:.1f} | {C["var_pct"] - A["var_pct"]:+.1f} |')
for label in A['intel_rhos']:
    a_r = A['intel_rhos'].get(label, (np.nan, np.nan, 0))[0]
    b_r = B['intel_rhos'].get(label, (np.nan, np.nan, 0))[0]
    c_r = C['intel_rhos'].get(label, (np.nan, np.nan, 0))[0]
    report.append(f'| {label} ρ | {a_r:+.3f} | {b_r:+.3f} | {b_r - a_r:+.3f} | '
                  f'{c_r:+.3f} | {c_r - a_r:+.3f} |')

# Affected-model rank shifts
report.append('\n## Rank shifts under masking (B vs A) for any affected model\n')
all_affected = set(both['model']) | DROP_MODELS
rank_a = A['model_mean'].rank(ascending=False, method='min')
rank_b = B['model_mean'].rank(ascending=False, method='min')
rows = []
for m in sorted(all_affected):
    if m not in rank_a.index or m not in rank_b.index:
        continue
    rows.append(
        (m, int(rank_a[m]), A['model_mean'][m], int(rank_b[m]),
         B['model_mean'][m], int(rank_b[m] - rank_a[m]))
    )
rows.sort(key=lambda r: r[1])
report.append('| model | rank A | mean A | rank B | mean B | Δ rank |')
report.append('|---|---|---|---|---|---|')
for m, ra, ma, rb, mb, dr in rows:
    report.append(f'| {m} | {ra} | {ma:+.3f} | {rb} | {mb:+.3f} | {dr:+d} |')

(OUT / 'cascade_impact.md').write_text('\n'.join(report))
print(f'\nWrote {OUT / "cascade_impact.md"}')

# Print primary-result summary to stdout
print('\n' + '=' * 70)
print('PRIMARY RESULT SUMMARY')
print('=' * 70)
print(f'{"":<22}{"A as-is":>12}{"B mask-32":>14}{"C drop-3":>14}')
print(f'{"eigenvalue":<22}{A["eigenvalue_1"]:>12.3f}{B["eigenvalue_1"]:>14.3f}{C["eigenvalue_1"]:>14.3f}')
print(f'{"alpha":<22}{A["alpha"]:>12.3f}{B["alpha"]:>14.3f}{C["alpha"]:>14.3f}')
print(f'{"var %":<22}{A["var_pct"]:>12.1f}{B["var_pct"]:>14.1f}{C["var_pct"]:>14.1f}')
for label in A['intel_rhos']:
    a_r = A['intel_rhos'][label][0]
    b_r = B['intel_rhos'].get(label, (np.nan,))[0]
    c_r = C['intel_rhos'].get(label, (np.nan,))[0]
    print(f'{label + " ρ":<22}{a_r:>12.3f}{b_r:>14.3f}{c_r:>14.3f}')
