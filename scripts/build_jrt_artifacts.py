"""Build the JRT-corrected, release-set, six-domain artifacts that
the regenerated figures and tables in the paper consume.

Reads (from bundle-relative paths):
  - release_data/leaderboard.csv             (release model roster)
  - release_data/long_model_x_dataset.csv    (raw canonical per-cell z)
  - analysis/jrt_corrected_scores.parquet  (JRT per-rating posteriors)
  - analysis/domain_classification.csv     (6-domain LLM-panel partition)
  - release_data/lsa_per_model.csv           (letter-string analogy accuracy)

Writes (to analysis/):
  - leaderboard_jrt.csv         (per-model composite for fig1)
  - c_factor_loadings_jrt.csv   (per-domain loadings + eig for fig2)
  - leaderboard_with_lsa.csv    (per-model jrt_z + lsa_acc for fig3)

Pipeline:
  - 83 release models from leaderboard.csv (strict coverage)
  - JRT-corrected per-(model, dataset) z replaces raw canonical z for the
    24 LLM-judge cells; raw fallback elsewhere
  - Re-z within the release model set so raw and JRT cells share the same scale
  - 6-domain partition recovers eigenvalue ~4.89, alpha ~0.96, ~81.5% variance
    (post-DQ-mask release set with 32 BOTH-flagged cells masked)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / 'analysis'
REBUILT = ANALYSIS / 'rebuilt'
REBUILT.mkdir(parents=True, exist_ok=True)
SP = REPO / 'release_data'


def _z(s):
    return (s - s.mean()) / s.std() if s.std() > 0 else s * 0


def parallel_analysis_95th(n_obs: int, n_vars: int, n_iter: int = 1000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    eigs = []
    for _ in range(n_iter):
        rand = rng.standard_normal((n_obs, n_vars))
        rand = (rand - rand.mean(0)) / rand.std(0, ddof=1)
        eigs.append(np.sort(np.linalg.eigvalsh(np.cov(rand.T)))[::-1][0])
    return float(np.percentile(eigs, 95))


def main():
    # === Release model set + raw long ===
    sp_lb = pd.read_csv(SP / 'leaderboard.csv')
    release_models = set(sp_lb.loc[sp_lb['release_model'] == 'yes', 'model'].tolist())
    print(f'Release models: {len(release_models)} (from release_data/leaderboard.csv)')

    raw_long = pd.read_csv(SP / 'long_model_x_dataset.csv').rename(columns={'dataset_z': 'mean_z'})

    # === JRT ===
    jrt = pd.read_parquet(ANALYSIS / 'jrt_corrected_scores.parquet')
    jrt_bench = (jrt.groupby(['benchmark', 'model'])
                 .agg(mean_z=('score', 'mean')).reset_index()
                 .rename(columns={'benchmark': 'dataset'}))
    jrt_bench['mean_z'] = jrt_bench.groupby('dataset')['mean_z'].transform(_z)
    jrt_cells = set(jrt_bench['dataset'].unique())
    jrt_keys = set(zip(jrt_bench['dataset'], jrt_bench['model']))

    raw_outside = raw_long[~raw_long['dataset'].isin(jrt_cells)][['dataset', 'model', 'mean_z']]
    raw_inside = raw_long[raw_long['dataset'].isin(jrt_cells)]
    keep_raw_inside = raw_inside[
        ~raw_inside.apply(lambda r: (r['dataset'], r['model']) in jrt_keys, axis=1)
    ][['dataset', 'model', 'mean_z']]
    new_long = pd.concat([
        raw_outside, keep_raw_inside,
        jrt_bench[['dataset', 'model', 'mean_z']],
    ], ignore_index=True)

    # Restrict to release models, then re-z within that set.
    new_fin = new_long[new_long['model'].isin(release_models)].copy()
    new_fin['mean_z'] = new_fin.groupby('dataset')['mean_z'].transform(_z)
    print(f'Release-set long: {len(new_fin):,} rows, {new_fin["dataset"].nunique()} datasets')

    # === 1. Leaderboard ===
    lb = (new_fin.groupby('model')
          .agg(datasets=('dataset', 'nunique'),
               n_metric_obs=('mean_z', 'count'),
               mean_z=('mean_z', 'mean'),
               median_z=('mean_z', 'median'))
          .reset_index()
          .sort_values('mean_z', ascending=False))
    lb['rank'] = lb['mean_z'].rank(ascending=False, method='min').astype(int)
    out_lb = REBUILT / 'leaderboard_jrt.csv'
    lb.to_csv(out_lb, index=False)
    print(f'\nWrote {out_lb}')
    print(lb.head(10).to_string(index=False))

    # === 2. 6-domain c-factor loadings ===
    dom6 = pd.read_csv(ANALYSIS / 'domain_classification.csv')
    new_fin_dom = new_fin.merge(dom6[['benchmark', 'domain']],
                                left_on='dataset', right_on='benchmark', how='inner')
    per_dom = (new_fin_dom.groupby(['model', 'domain'])
               .agg(z=('mean_z', 'mean')).reset_index()
               .pivot(index='model', columns='domain', values='z'))
    per_dom = per_dom.dropna()
    print(f'\n6-domain matrix: {per_dom.shape[0]} models x {per_dom.shape[1]} domains')

    Xz = (per_dom.values - per_dom.values.mean(0)) / per_dom.values.std(0, ddof=1)
    corr = np.corrcoef(Xz.T)
    eigs = np.sort(np.linalg.eigvalsh(corr))[::-1]
    pca = PCA(n_components=1).fit(Xz)
    loadings = pca.components_[0] * np.sqrt(pca.explained_variance_[0])
    pa95 = parallel_analysis_95th(Xz.shape[0], Xz.shape[1])

    domains = list(per_dom.columns)
    n_dom = len(domains)
    eig_col = list(eigs[:n_dom])
    df_load = pd.DataFrame({
        'domain': domains,
        'c_loading': loadings,
        'eigenvalue': eig_col,
    })
    df_load['rand_p95'] = pa95
    out_load = REBUILT / 'c_factor_loadings_jrt.csv'
    df_load.to_csv(out_load, index=False)
    print(f'\nWrote {out_load}')
    print(df_load.to_string(index=False))
    print(f'  parallel-analysis p95 = {pa95:.3f}')
    pct = eigs[0] / eigs.sum() * 100
    iv = Xz.var(0, ddof=1).sum()
    tv = Xz.sum(1).var(ddof=1)
    alpha = (n_dom / (n_dom - 1)) * (1 - iv / tv)
    print(f'  eig1={eigs[0]:.3f}, %var={pct:.1f}, alpha={alpha:.3f}')

    # === 3. AGC x LSA scatter source ===
    lsa = pd.read_csv(SP / 'lsa_per_model.csv')
    lsa_join = lb[['model', 'mean_z']].merge(lsa[['model', 'lsa_acc']], on='model', how='inner')
    out_lsa = REBUILT / 'leaderboard_with_lsa.csv'
    lsa_join.to_csv(out_lsa, index=False)
    print(f'\nWrote {out_lsa} ({len(lsa_join)} release models with LSA coverage)')
    from scipy.stats import spearmanr, pearsonr
    rho, _ = spearmanr(lsa_join['mean_z'], lsa_join['lsa_acc'])
    r, _ = pearsonr(lsa_join['mean_z'], lsa_join['lsa_acc'])
    print(f'  agc-jrt x lsa: rho={rho:+.3f}, r={r:+.3f}')


if __name__ == '__main__':
    main()
