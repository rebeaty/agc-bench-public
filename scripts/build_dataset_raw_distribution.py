"""Build the frozen 83-model cohort raw distribution per dataset.

Output: `release_data/dataset_raw_distribution.csv` with columns
`dataset, canonical_metric, raw_mean, raw_sd, n, source` — for each of the
67 v1 datasets, the metric whose per-cell z values best reproduce the
published `dataset_z` column, plus that metric's cohort raw mean+sd.

Mapping discovery: for each candidate metric available in the upstream
artifact, compute the per-cell z-vector across the 83 release models and
RMSE-compare against the published per-cell `dataset_z` from
`release_data/long_model_x_dataset.csv`. The metric with lowest RMSE is
adopted as the canonical-for-reproduction per dataset.

Sources of candidate metrics:
  - `agc_long_metric_z.parquet` (per-(model, dataset, metric) raw `value`;
    predecessor build pipeline) — used for the 43 raw-source datasets.
  - `analysis/jrt_corrected_scores.parquet` (`score_raw`, the original
    3-vendor panel raw before GRM correction) — used for the 24 JRT-rated
    datasets, since AGC-Judge's raw output approximates the panel's raw,
    not the GRM theta.

Usage:
  python scripts/build_dataset_raw_distribution.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_AG = REPO / 'analysis/agc_long_metric_z.parquet'
DEFAULT_JRT = REPO / 'analysis/jrt_corrected_scores.parquet'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', default=str(DEFAULT_AG))
    p.add_argument('--jrt-source', default=str(DEFAULT_JRT))
    p.add_argument('--out', default=str(REPO / 'release_data/dataset_raw_distribution.csv'))
    args = p.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f'Upstream raw artifact missing: {src}')

    # Released cohort + canonical model set + canonical published z's
    lb = pd.read_csv(REPO / 'release_data/leaderboard.csv')
    release_models = set(lb['model'].tolist())
    md = pd.read_csv(REPO / 'release_data/dataset_metadata.csv')
    primary = set(md[md['status'] == 'included']['dataset'].tolist())
    pub_long = pd.read_csv(REPO / 'release_data/long_model_x_dataset.csv')
    if 'dq_masked' in pub_long.columns:
        pub_long = pub_long[~pub_long['dq_masked'].fillna(False)]
    pub_long = pub_long[pub_long['model'].isin(release_models)]
    pub_per_ds = {ds: g.set_index('model')['dataset_z']
                  for ds, g in pub_long.groupby('dataset')}

    # ── Source 1: agc_long_metric_z (per-cell raws for everything) ─────────
    print(f'[1/3] Reading: {src}')
    long = pd.read_parquet(src)
    print(f'  rows: {len(long):,}, models: {long["model"].nunique()}, datasets: {long["dataset"].nunique()}')
    long = long[long['model'].isin(release_models)].copy()
    ag_cell = (long.groupby(['dataset', 'metric', 'model'])
                   .agg(value=('value', 'mean'), z=('z', 'mean'))
                   .reset_index())

    # ── Source 2: jrt_corrected score_raw (for JRT cells) ──────────────────
    print(f'[2/3] Reading: {args.jrt_source}')
    jrt_path = Path(args.jrt_source)
    if jrt_path.exists():
        jrt = pd.read_parquet(jrt_path)
        jrt = jrt[jrt['model'].isin(release_models)]
        jrt_cell = (jrt.groupby(['benchmark', 'metric', 'model'])
                       .agg(value=('score_raw', 'mean'))
                       .reset_index()
                       .rename(columns={'benchmark': 'dataset'}))
        # Recompute per-(dataset, metric) cohort z so we can RMSE against
        # published dataset_z to discover the best single metric.
        def _z(s):
            sd = s.std()
            return (s - s.mean()) / sd if sd > 0 else s * 0
        jrt_cell['z'] = jrt_cell.groupby(['dataset', 'metric'])['value'].transform(_z)
    else:
        jrt_cell = pd.DataFrame(columns=['dataset', 'metric', 'model', 'value', 'z'])
        print('  (skipped — file not found)')

    # ── For each primary dataset, pick the metric whose per-cell z best
    # matches the published dataset_z. Try jrt_cell first for JRT-rated
    # benches (since AGC-Judge raw aligns with score_raw, not the agc_long
    # pre-JRT raw); otherwise agc_cell. ───────────────────────────────────
    print('[3/3] Discovering best-match canonical metric per dataset...')
    rows = []
    diagnostics = []
    score_source = pub_long.groupby('dataset')['score_source'].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else None)

    def best_metric_for(ds, candidate_df, src_name):
        if ds not in pub_per_ds:
            return None
        pub = pub_per_ds[ds]
        sub = candidate_df[candidate_df['dataset'] == ds]
        if sub.empty:
            return None
        best = None
        for m in sub['metric'].unique():
            cell = sub[sub['metric'] == m]
            sd = cell['value'].std()
            if pd.isna(sd) or sd == 0:
                continue
            zvec = cell.set_index('model')['z']
            common = pub.index.intersection(zvec.index)
            if len(common) < 5:
                continue
            rmse = float(np.sqrt(((pub.loc[common] - zvec.loc[common]) ** 2).mean()))
            cand = {
                'metric': m,
                'rmse': rmse,
                'raw_mean': float(cell['value'].mean()),
                'raw_sd': float(sd),
                'n': int(cell['model'].nunique()),
                'source': src_name,
            }
            if best is None or rmse < best['rmse']:
                best = cand
        return best

    for ds in sorted(primary):
        # Prefer agc_long_metric_z always: its per-cell raw distribution is what
        # the published single-rater LLM-judge scores were drawn from, which is
        # the scale AGC-Judge's raw output approximates. The JRT score_raw
        # cohort has a much tighter std on some benches (post-3-vendor-panel
        # aggregation), so z-norming AGC-Judge raw against it explodes the
        # composite when AGC-Judge's calibration drifts even slightly.
        chosen = best_metric_for(ds, ag_cell, 'agc_long_metric_z')
        if chosen is None:
            chosen = best_metric_for(ds, jrt_cell, 'jrt_score_raw')
        if chosen is None:
            diagnostics.append({'dataset': ds, 'reason': 'no_candidate_metric'})
            continue
        rows.append({
            'dataset': ds,
            'canonical_metric': chosen['metric'],
            'raw_mean': chosen['raw_mean'],
            'raw_sd': chosen['raw_sd'],
            'n': chosen['n'],
            'rmse_vs_published': chosen['rmse'],
            'source': chosen['source'],
        })

    out_df = pd.DataFrame(rows).sort_values('dataset').reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f'\nWrote {len(out_df)} rows to {out}')
    print(f'Coverage vs 67 v1 datasets: {len(out_df)} / {len(primary)}')
    if diagnostics:
        for d in diagnostics:
            print(f'  MISSING {d["dataset"]}: {d["reason"]}')

    # Show worst-fitting datasets so the upstream-vs-published delta is visible
    print('\nWorst RMSE-matched datasets (proxy for AGC-Judge calibration drift):')
    print(out_df.sort_values('rmse_vs_published', ascending=False)
                .head(10).to_string(index=False))


if __name__ == '__main__':
    main()
