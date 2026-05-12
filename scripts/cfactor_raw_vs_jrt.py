"""C-factor extraction under raw vs JRT-corrected scoring.

Re-runs the unidimensional factor extraction the paper reports
(Section sec:cfactor) on per-(model, domain) composites built (a) from raw
canonical dataset z-scores and (b) from JRT-corrected dataset z-scores
where available, raw fallback otherwise. Reports first-eigenvalue,
parallel-analysis 95th percentile, and Cronbach's alpha for both
the raw canonical and JRT-corrected six-domain paper-primary results.

Reads (from bundle-relative paths):
  - release_data/long_model_x_dataset.csv     (raw canonical per-cell z)
  - analysis/jrt_corrected_scores.parquet   (JRT per-rating posteriors)
  - analysis/domain_classification.csv      (6-domain primary-result partition)

Output: prints summary statistics; no files written.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / 'analysis'
SP = REPO / 'release_data'


def _z(s):
    return (s - s.mean()) / s.std() if s.std() > 0 else s * 0


def cronbach_alpha(matrix: np.ndarray) -> float:
    k = matrix.shape[1]
    item_var = matrix.var(axis=0, ddof=1).sum()
    total_var = matrix.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_var / total_var)


def parallel_analysis_95th(n_obs: int, n_vars: int, n_iter: int = 1000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    eigs = []
    for _ in range(n_iter):
        rand = rng.standard_normal((n_obs, n_vars))
        rand = (rand - rand.mean(0)) / rand.std(0, ddof=1)
        cov = np.cov(rand.T)
        eigs.append(np.sort(np.linalg.eigvalsh(cov))[::-1][0])
    return float(np.percentile(eigs, 95))


def cfactor_summary(per_dom: pd.DataFrame, label: str):
    domains = [c for c in per_dom.columns if c != 'model']
    X = per_dom[domains].dropna().values
    Xz = (X - X.mean(0)) / X.std(0, ddof=1)
    corr = np.corrcoef(Xz.T)
    eigs = np.sort(np.linalg.eigvalsh(corr))[::-1]
    pct = eigs[0] / eigs.sum() * 100
    pa_95 = parallel_analysis_95th(X.shape[0], len(domains))
    alpha = cronbach_alpha(Xz)
    pca = PCA(n_components=1).fit(Xz)
    loadings = pca.components_[0] * np.sqrt(pca.explained_variance_[0])
    print(f'\n  {label}: n_models={X.shape[0]}, n_domains={len(domains)}')
    print(f'    eigvals (correlation matrix) = {[f"{e:.3f}" for e in eigs]}')
    print(f'    parallel-analysis 95th       = {pa_95:.3f}')
    print(f'    first eigenvalue             = {eigs[0]:.3f}  (above PA: {eigs[0] > pa_95})')
    print(f'    %var first factor            = {pct:.1f}%')
    print(f'    Cronbach alpha               = {alpha:.3f}')
    for d, l in zip(domains, loadings):
        print(f'      {d:<28s} loading = {l:+.3f}')


def build_long_with_jrt(raw_long: pd.DataFrame, jrt_bench: pd.DataFrame) -> pd.DataFrame:
    raw_long = raw_long.rename(columns={'dataset_z': 'mean_z'}) if 'dataset_z' in raw_long.columns else raw_long
    raw_long['z_source'] = 'raw'
    jrt_cells = set(jrt_bench['dataset'].unique())
    raw_outside = raw_long[~raw_long['dataset'].isin(jrt_cells)].copy()
    raw_inside = raw_long[raw_long['dataset'].isin(jrt_cells)].copy()
    jrt_keys = set(zip(jrt_bench['dataset'], jrt_bench['model']))
    keep_raw_inside = raw_inside[
        ~raw_inside.apply(lambda r: (r['dataset'], r['model']) in jrt_keys, axis=1)
    ].copy()
    return pd.concat([
        raw_outside[['dataset', 'model', 'mean_z', 'z_source']],
        keep_raw_inside[['dataset', 'model', 'mean_z', 'z_source']],
        jrt_bench[['dataset', 'model', 'mean_z', 'z_source']],
    ], ignore_index=True)


def per_model_per_domain(long: pd.DataFrame, dom_map: pd.DataFrame, dom_col: str) -> pd.DataFrame:
    j = long.merge(dom_map[['benchmark', dom_col]],
                   left_on='dataset', right_on='benchmark', how='inner')
    return (j.groupby(['model', dom_col])
            .agg(z=('mean_z', 'mean'))
            .reset_index()
            .pivot(index='model', columns=dom_col, values='z')
            .reset_index())


def main():
    raw_long = pd.read_csv(SP / 'long_model_x_dataset.csv')
    if 'dataset_z' in raw_long.columns:
        raw_long = raw_long.rename(columns={'dataset_z': 'mean_z'})

    jrt = pd.read_parquet(ANALYSIS / 'jrt_corrected_scores.parquet')
    jrt_bench = (jrt.groupby(['benchmark', 'model'])
                 .agg(jrt_mean=('score', 'mean')).reset_index()
                 .rename(columns={'benchmark': 'dataset'}))
    jrt_bench['mean_z'] = jrt_bench.groupby('dataset')['jrt_mean'].transform(_z)
    jrt_bench['z_source'] = 'jrt'

    raw_with_zsource = raw_long.copy()
    raw_with_zsource['z_source'] = 'raw'
    new_long = build_long_with_jrt(raw_long.copy(), jrt_bench)

    dom6 = pd.read_csv(ANALYSIS / 'domain_classification.csv')

    print('==================== c-factor under 6-domain (paper primary result) ====================')
    raw_6 = per_model_per_domain(raw_with_zsource, dom6, 'domain')
    jrt_6 = per_model_per_domain(new_long, dom6, 'domain')
    cfactor_summary(raw_6, 'RAW (canonical)')
    cfactor_summary(jrt_6, 'JRT (3-judge corrected)')


if __name__ == '__main__':
    main()
