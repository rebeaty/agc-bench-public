"""Correlate the optional RiddleBench side audit with AGC-Bench metrics.

The primary release does not require RiddleBench. This script is a supporting
analysis aid for the side analysis reported in the audit notes. If the raw
RiddleBench per-model file is available, the script rebuilds the merged table;
otherwise it uses the bundled audit/dq_sweep/riddlebench_merged.csv snapshot.

  - AGC-Bench composite (mean_z)
  - per-domain z (Problem Solving especially, expected highest loading)
  - AA Intelligence Index, MMLU-Pro, GPQA-Diamond, HLE, LSA (Lewis-Mitchell)
  - c-factor first principal component (per model)

Outputs the correlation table and the position RiddleBench would occupy
relative to the reasoning anchors discussed in the paper.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BUNDLE = Path(__file__).resolve().parents[2]
RAW_RIDDLEBENCH = (
    BUNDLE / 'data/external/riddlebench/'
    'per_model_riddlebench_strict2_release_models81_full_20260505.csv'
)
BUNDLED_MERGED = BUNDLE / 'audit/dq_sweep/riddlebench_merged.csv'

if RAW_RIDDLEBENCH.exists():
    rb = pd.read_csv(RAW_RIDDLEBENCH)[['model', 'accuracy', 'n_items']].rename(
        columns={'accuracy': 'rb_acc'}
    )
    print(f'RiddleBench per-model rows: {len(rb)}')

    lb = pd.read_csv(BUNDLE / 'release_data/leaderboard.csv')
    print(f'AGC leaderboard rows: {len(lb)}')

    pd_dom = pd.read_csv(BUNDLE / 'analysis/per_domain_jrt.csv')

    intel = pd.read_csv(BUNDLE / 'analysis/intelligence_join.csv')
    intel['model'] = intel['model'].apply(
        lambda s: s.replace('_', '/', 1) if isinstance(s, str) and '_' in s and '/' not in s else s
    )

    lsa = pd.read_csv(BUNDLE / 'release_data/lsa_per_model.csv')
    print(f'LSA rows: {len(lsa)}')
    lsa_cols = [c for c in lsa.columns if c != 'model']
    print(f'LSA columns: {lsa.columns.tolist()}')

    df = rb.merge(lb[['model', 'mean_z']], on='model', how='inner')
    df = df.merge(pd_dom, on='model', how='left')
    df = df.merge(intel, on='model', how='left', suffixes=('', '_intel'))
    df = df.merge(lsa, on='model', how='left')
else:
    if not BUNDLED_MERGED.exists():
        raise FileNotFoundError(
            f'Need either raw RiddleBench file at {RAW_RIDDLEBENCH} '
            f'or bundled merged snapshot at {BUNDLED_MERGED}'
        )
    df = pd.read_csv(BUNDLED_MERGED)
    lsa_cols = [c for c in df.columns if c == 'lsa_acc' or 'lsa' in c.lower()]
    print(f'Loaded bundled RiddleBench merged snapshot: {len(df)} rows')

print(f'\nMerged rows: {len(df)}')


def corr(a, b, method='spearman'):
    sub = pd.DataFrame({'a': a, 'b': b}).dropna()
    if len(sub) < 10:
        return (np.nan, np.nan, len(sub))
    if method == 'spearman':
        rho, p = stats.spearmanr(sub['a'], sub['b'])
    else:
        rho, p = stats.pearsonr(sub['a'], sub['b'])
    return (rho, p, len(sub))


print('\n' + '=' * 60)
print('RiddleBench × AGC and intelligence anchors')
print('=' * 60)

probes = [
    ('AGC composite (mean_z)', 'mean_z'),
    ('AGC: Brainstorming', 'Brainstorming'),
    ('AGC: Figurative Language', 'Figurative Language'),
    ('AGC: Humor', 'Humor'),
    ('AGC: Problem Solving', 'Problem Solving'),
    ('AGC: STEM', 'STEM'),
    ('AGC: Story / Narrative', 'Story / Narrative'),
    ('AA Intelligence Index', 'intelligence_index_aa'),
    ('MMLU-Pro', 'mmlu_pro_aa'),
    ('GPQA-Diamond', 'gpqa_diamond'),
    ('HLE', 'hle'),
    ('LiveCodeBench', 'livecodebench_aa'),
    ('AIME 2025', 'aime_2025_aa'),
    ('SciCode', 'scicode_aa'),
]
# LSA column
lsa_col = next((c for c in lsa_cols if 'lsa' in c.lower()
                or 'analog' in c.lower()
                or 'counter' in c.lower()
                or 'accuracy' in c.lower()),
               None)
if lsa_col:
    probes.append((f'LSA / counterfactual analogies ({lsa_col})', lsa_col))

print(f'{"probe":<46}{"ρ":>8}{"r":>8}{"p":>10}{"n":>5}')
print('-' * 78)
results = []
for label, col in probes:
    if col not in df.columns:
        continue
    rho, p, n = corr(df['rb_acc'], df[col], 'spearman')
    r, _, _ = corr(df['rb_acc'], df[col], 'pearson')
    print(f'{label:<46}{rho:>+8.3f}{r:>+8.3f}{p:>10.2g}{n:>5}')
    results.append({'probe': label, 'spearman_rho': rho,
                    'pearson_r': r, 'p': p, 'n': n})

pd.DataFrame(results).to_csv(
    BUNDLE / 'audit/dq_sweep/riddlebench_correlations.csv', index=False
)

# Comparison with the paper's existing §4.4 anchors
print('\n' + '=' * 60)
print('§4.4 reasoning-anchor positioning')
print('=' * 60)
print('Paper currently reports:')
print('  AGC × LSA          ρ ≈ +0.55, r ≈ +0.53 (n ≈ 82)')
print('  AGC × MMLU-Pro     ρ ≈ +0.63                 (n ≈ 57)')
print('  AGC × GPQA-Diamond ρ ≈ +0.76')
print('  AGC × AA Intel Idx ρ ≈ +0.77')
print()
agc_rb = corr(df['mean_z'], df['rb_acc'])
print(f'New: AGC × RiddleBench  ρ = {agc_rb[0]:+.3f}, n = {agc_rb[2]}')

# RiddleBench × LSA itself (head-to-head reasoning anchors)
if lsa_col:
    rb_lsa = corr(df['rb_acc'], df[lsa_col])
    print(f'RiddleBench × LSA      ρ = {rb_lsa[0]:+.3f}, n = {rb_lsa[2]}')
rb_aa = corr(df['rb_acc'], df['intelligence_index_aa'])
print(f'RiddleBench × AA Intel ρ = {rb_aa[0]:+.3f}, n = {rb_aa[2]}')

# Top / bottom on RiddleBench
print('\nTop 10 on RiddleBench:')
print(df.sort_values('rb_acc', ascending=False)[
    ['model', 'rb_acc', 'mean_z', 'intelligence_index_aa']
].head(10).to_string(index=False))

print('\nBottom 5 on RiddleBench:')
print(df.sort_values('rb_acc')[
    ['model', 'rb_acc', 'mean_z', 'intelligence_index_aa']
].head(5).to_string(index=False))

# Save merged frame for downstream
df.to_csv(BUNDLE / 'audit/dq_sweep/riddlebench_merged.csv', index=False)
print(f'\nWrote: {BUNDLE / "audit/dq_sweep/riddlebench_correlations.csv"}')
print(f'Wrote: {BUNDLE / "audit/dq_sweep/riddlebench_merged.csv"}')
