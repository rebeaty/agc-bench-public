"""Compute a release-set-comparable z-score for a new model and slot it into
the published 83-model leaderboard.

This is the post-HELM aggregation step. Inputs:
  --new-model    : The model identifier being scored (matches HELM's model arg).
  --suite        : HELM suite name (default: first_full_trial), used to find
                   stats.json files under benchmark_output/runs/<suite>/.
  --runs-root    : Optional override for the runs root (default: benchmark_output/runs).
  --out          : Output directory.

Walks every per-(scenario, model=NEW_MODEL) stats.json under the run suite,
picks the canonical metric per dataset, and z-normalizes the new model's raw
score against the frozen 83-model cohort raw distribution in
`release_data/dataset_raw_distribution.csv`. Computes per-domain mean-z and an
overall composite, then prints the leaderboard slot vs the published 83-model
release set in `release_data/leaderboard.csv`.

Usage:
  python scripts/integrate_new_model.py \\
      --new-model openai/gpt-5.5 \\
      --suite first_full_trial \\
      --out analysis/scored/openai__gpt-5.5
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent


def release_distribution() -> pd.DataFrame:
    """Per-dataset frozen 83-model cohort raw distribution.

    Reads `release_data/dataset_raw_distribution.csv` (built via
    `scripts/build_dataset_raw_distribution.py`) which carries per-dataset
    `(raw_mean, raw_sd, n)` of the cohort's per-cell mean of the canonical
    metric, restricted to the 83 release models. A new model's raw score is
    z-normed against THIS distribution to land at a leaderboard slot
    comparable to the published 83-model cohort.

    NOTE: do not z-norm against `release_data/long_model_x_dataset.csv`'s
    `dataset_z` column — those values are already z-normalized within-dataset
    (mean=0, sd=1 by construction), and (raw - 0) / 1 collapses to the raw
    score itself, blowing up the composite for any bench whose canonical
    metric isn't already on a near-standard scale.
    """
    dist_path = REPO / 'release_data/dataset_raw_distribution.csv'
    if not dist_path.exists():
        raise SystemExit(
            f'Missing cohort raw distribution: {dist_path}. '
            f'Run `python scripts/build_dataset_raw_distribution.py` to build it '
            f'from the upstream per-cell raw artifact.'
        )
    dist = pd.read_csv(dist_path)
    return dist.rename(columns={'raw_mean': 'mean', 'raw_sd': 'std', 'n': 'count'})


def load_domain_map() -> dict:
    df = pd.read_csv(REPO / 'analysis/domain_classification.csv')
    return dict(zip(df['benchmark'], df['domain']))


def load_canonical_metrics() -> dict:
    """Load the canonical primary metric per dataset.

    `data/registry/registry_metrics.yaml` stores per-dataset metric lists. The
    first metric whose `is_canonical` flag is True (or the first listed if no
    flag is present) is the cell-defining metric for that dataset.
    """
    md_path = REPO / 'data/registry/registry_metrics.yaml'
    if not md_path.exists():
        raise SystemExit(f'Missing registry: {md_path}')
    reg = yaml.safe_load(md_path.read_text()) or {}
    # Schema: top-level `datasets:` mapping to per-bench metric lists.
    datasets_block = reg.get('datasets', reg)  # tolerate either nested or flat
    out = {}
    for dataset, spec in datasets_block.items():
        metrics = spec if isinstance(spec, list) else spec.get('metrics', [])
        if not metrics:
            continue
        canonical = None
        for m in metrics:
            if isinstance(m, dict) and (m.get('is_canonical') or m.get('canonical')):
                canonical = m.get('name') or m.get('metric_name')
                break
        if canonical is None:
            first = metrics[0]
            canonical = first.get('name') or first.get('metric_name') if isinstance(first, dict) else first
        out[dataset] = canonical
    return out


def find_stats_json(runs_root: Path, suite: str, dataset: str, model: str) -> Optional[Path]:
    """Locate the HELM-emitted stats.json for a (scenario, model) cell."""
    suite_dir = runs_root / suite
    if not suite_dir.exists():
        return None
    safe_model = model.replace('/', '_')
    candidates = [
        suite_dir / f'{dataset}:model={model}' / 'stats.json',
        suite_dir / f'{dataset}:model={safe_model}' / 'stats.json',
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: HELM sometimes uses different separators or appends sub-args
    matches = list(suite_dir.glob(f'{dataset}:model={safe_model}*/stats.json')) \
            + list(suite_dir.glob(f'{dataset}:model={model}*/stats.json'))
    return matches[0] if matches else None


def extract_canonical_score(stats_json: Path, canonical_metric: str) -> Optional[float]:
    """Extract the cell-level canonical metric value from a HELM stats.json."""
    data = json.loads(stats_json.read_text())
    # HELM stats.json is a list of {'name': {'name': metric_name, ...}, 'mean': value, ...}
    if isinstance(data, list):
        for entry in data:
            name = entry.get('name', {})
            if isinstance(name, dict):
                if name.get('name') == canonical_metric:
                    return float(entry.get('mean', np.nan))
            elif name == canonical_metric:
                return float(entry.get('mean', np.nan))
    elif isinstance(data, dict):
        if canonical_metric in data:
            entry = data[canonical_metric]
            return float(entry if isinstance(entry, (int, float)) else entry.get('mean', np.nan))
    return None


def extract_all_metric_values(stats_json: Path) -> dict:
    """Pull {metric_name: mean_value} for every per-bench metric in a stats.json.

    Returns the unaggregated metric surface; the caller intersects this with
    the cohort-distribution table to pick which metrics to z-norm.
    """
    data = json.loads(stats_json.read_text())
    out: dict = {}
    if isinstance(data, list):
        for entry in data:
            name = entry.get('name', {})
            metric_name = name.get('name') if isinstance(name, dict) else name
            if metric_name is None:
                continue
            mean = entry.get('mean')
            if mean is None:
                continue
            try:
                out.setdefault(metric_name, float(mean))
            except (TypeError, ValueError):
                continue
    elif isinstance(data, dict):
        for metric_name, entry in data.items():
            try:
                out[metric_name] = float(
                    entry if isinstance(entry, (int, float)) else entry.get('mean'))
            except (TypeError, ValueError):
                continue
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--new-model', required=True)
    p.add_argument('--suite', default='first_full_trial')
    p.add_argument('--runs-root', default=str(REPO / 'benchmark_output/runs'))
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cohort = release_distribution()  # dataset, canonical_metric, mean, std, count, ...
    cohort_idx = cohort.set_index('dataset')
    domain = load_domain_map()
    md = pd.read_csv(REPO / 'release_data/dataset_metadata.csv')
    primary_datasets = md[md['status'] == 'included']['dataset'].tolist()

    runs_root = Path(args.runs_root)
    rows = []
    missing = []
    for ds in primary_datasets:
        if ds not in cohort_idx.index:
            missing.append((ds, 'no_release_distribution'))
            continue
        sj = find_stats_json(runs_root, args.suite, ds, args.new_model)
        if not sj:
            missing.append((ds, 'no_stats_json'))
            continue
        run_metrics = extract_all_metric_values(sj)
        if not run_metrics:
            missing.append((ds, 'no_canonical_value'))
            continue

        cs = cohort_idx.loc[ds]
        canonical_metric = cs['canonical_metric']
        if canonical_metric not in run_metrics:
            missing.append((ds, f'canonical_{canonical_metric}_not_in_stats_json'))
            continue
        raw = run_metrics[canonical_metric]
        if raw is None or pd.isna(raw):
            missing.append((ds, 'no_canonical_value'))
            continue
        # AGC-Judge / annotators emit -100 as a parse-failure sentinel; treat it
        # (or anything near it) as missing rather than z-normalizing.
        if raw <= -50:
            missing.append((ds, f'parse_failure_sentinel_raw={raw:.2f}'))
            continue
        if cs['std'] == 0 or pd.isna(cs['std']):
            missing.append((ds, 'release_sd_zero'))
            continue
        z = (raw - cs['mean']) / cs['std']
        rows.append({
            'model': args.new_model, 'dataset': ds,
            'canonical_metric': canonical_metric, 'raw_score': raw, 'dataset_z': z,
            'release_mean': cs['mean'], 'release_sd': cs['std'], 'release_n': int(cs['count']),
            'cohort_source': cs.get('source', ''),
            'domain': domain.get(ds),
            'stats_json': str(sj.relative_to(REPO)),
        })

    new_long = pd.DataFrame(rows)
    new_long.to_csv(out / 'per_dataset_z.csv', index=False)
    pd.DataFrame(missing, columns=['dataset', 'reason']).to_csv(out / 'missing.csv', index=False)

    if new_long.empty:
        raise SystemExit(f'No cells scored. {len(missing)} datasets missing — see {out}/missing.csv')

    per_domain = new_long.groupby('domain')['dataset_z'].agg(['mean', 'count']).reset_index()
    per_domain.columns = ['domain', 'mean_z', 'n_datasets']
    per_domain.to_csv(out / 'per_domain.csv', index=False)
    composite = float(new_long['dataset_z'].mean())

    lb = pd.read_csv(REPO / 'release_data/leaderboard.csv')
    leaderboard_pos = (lb['mean_z'] > composite).sum() + 1
    above = lb[lb['mean_z'] > composite].sort_values('mean_z').head(1)
    below = lb[lb['mean_z'] <= composite].sort_values('mean_z', ascending=False).head(1)

    summary = {
        'model': args.new_model, 'suite': args.suite,
        'composite_mean_z': composite,
        'n_datasets_scored': int(new_long.shape[0]),
        'n_primary_datasets': int(len(primary_datasets)),
        'n_missing': int(len(missing)),
        'leaderboard_rank_if_inserted': int(leaderboard_pos),
        'release_model_count': int(len(lb)),
        'nearest_above': above.iloc[0].to_dict() if not above.empty else None,
        'nearest_below': below.iloc[0].to_dict() if not below.empty else None,
        'per_domain': per_domain.to_dict(orient='records'),
    }
    (out / 'leaderboard_line.json').write_text(json.dumps(summary, indent=2, default=str))

    print(f'\n=== {args.new_model} ===')
    print(f'  composite mean-z:   {composite:+.3f}')
    print(f'  cells scored:       {summary["n_datasets_scored"]} / {summary["n_primary_datasets"]} primary datasets')
    if missing:
        print(f'  missing:            {len(missing)} (see {out}/missing.csv)')
    print(f'  leaderboard rank:   #{summary["leaderboard_rank_if_inserted"]} of {summary["release_model_count"]}')
    if summary['nearest_above']:
        a = summary['nearest_above']; print(f'    above: {a["model"]:38s} (z = {a["mean_z"]:+.3f})')
    print(f'    you:   {args.new_model:38s} (z = {composite:+.3f})')
    if summary['nearest_below']:
        b = summary['nearest_below']; print(f'    below: {b["model"]:38s} (z = {b["mean_z"]:+.3f})')
    print(f'\n  per-domain:')
    for r in per_domain.itertuples():
        print(f'    {r.domain:30s} mean-z = {r.mean_z:+.3f}  (n = {r.n_datasets})')
    print(f'\nWrote {out}/{{per_dataset_z.csv, per_domain.csv, leaderboard_line.json, missing.csv}}')


if __name__ == '__main__':
    main()
