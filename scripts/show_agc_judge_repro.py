"""Print AGC-Judge-only reproduction of any paper model's composite vs the
JRT-corrected published composite, on the 24 LLM-judge benchmarks.

Usage:
  python scripts/show_agc_judge_repro.py claude-opus-4.5
  python scripts/show_agc_judge_repro.py anthropic/claude-opus-4.5
  python scripts/show_agc_judge_repro.py --all                # rank-correlation summary

Reproductions live in:
  analysis/agc_judge_leaderboard_repro_test.csv             (73 models, in-distribution)
  analysis/agc_judge_leaderboard_repro_holdout_models.csv   (10 models, never in AGC-Judge training)
  analysis/agc_judge_leaderboard_repro_holdout_benches.csv  (83 models, held-out 3 benchmarks)

The test split was scored on items the AGC-Judge saw during training (different
items, same models). The holdout-models split is the cleanest test: 10 models
never seen in any form during fine-tuning. The holdout-benches split shows
generalization to 3 benchmarks (tinyfabulist, liveideabench, conceptual_design)
that were entirely held out from training.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import Optional, List
import pandas as pd
from scipy.stats import spearmanr

A = Path(__file__).parent.parent / 'analysis'
SPLITS = {
    'test':            A / 'agc_judge_leaderboard_repro_test.csv',
    'holdout_models':  A / 'agc_judge_leaderboard_repro_holdout_models.csv',
    'holdout_benches': A / 'agc_judge_leaderboard_repro_holdout_benches.csv',
}


def normalize_model(name: str, candidates: List[str]) -> Optional[str]:
    """Resolve a partial model name to a full vendor/model identifier.
    Match strategy: exact > endswith > substring (case-insensitive).
    """
    n = name.lower()
    by_exact = [c for c in candidates if c.lower() == n]
    if by_exact:
        return by_exact[0]
    by_end = [c for c in candidates if c.lower().endswith('/' + n) or c.lower().endswith(n)]
    if len(by_end) == 1:
        return by_end[0]
    by_sub = [c for c in candidates if n in c.lower()]
    if len(by_sub) == 1:
        return by_sub[0]
    if len(by_sub) > 1:
        sys.stderr.write(f'Multiple matches for {name!r}:\n  ' + '\n  '.join(by_sub) + '\n')
        return None
    return None


def show_model(model_query: str) -> int:
    found_any = False
    for split_name, path in SPLITS.items():
        df = pd.read_csv(path)
        match = normalize_model(model_query, df['model'].tolist())
        if not match:
            continue
        row = df[df['model'] == match].iloc[0]
        print(f'\n=== {match} on {split_name} ({len(df)} models in this split) ===')
        print(f'  AGC-Judge composite (predicted):  z = {row["pred_composite"]:+.3f}   rank #{int(row["pred_rank"])}')
        print(f'  JRT-corrected composite (gold):    z = {row["gold_composite"]:+.3f}   rank #{int(row["gold_rank"])}')
        print(f'  Rank delta (pred - gold):         {int(row["delta"]):+d}')
        print(f'  Composite delta (pred - gold):    {row["pred_composite"] - row["gold_composite"]:+.3f}')
        print(f'  Cells contributing:               {int(row["n_cells"])}')
        found_any = True
    if not found_any:
        sys.stderr.write(f'No reproduction found for model {model_query!r} in any split.\n')
        sys.stderr.write('Try `python scripts/show_agc_judge_repro.py --all` to see the full roster.\n')
        return 1
    return 0


def show_summary() -> int:
    print('AGC-Judge-only leaderboard reproduction vs JRT-corrected published leaderboard')
    print('=' * 78)
    for split_name, path in SPLITS.items():
        df = pd.read_csv(path)
        rho_rank, _ = spearmanr(df['pred_rank'], df['gold_rank'])
        rho_z, _    = spearmanr(df['pred_composite'], df['gold_composite'])
        delta_max = df['delta'].abs().max()
        delta_mean = df['delta'].abs().mean()
        comp_mae = (df['pred_composite'] - df['gold_composite']).abs().mean()
        print(f'\n{split_name:18s}  n_models={len(df):3d}')
        print(f'  Rank ρ (pred vs gold):       {rho_rank:+.3f}')
        print(f'  Composite-z ρ (pred vs gold): {rho_z:+.3f}')
        print(f'  Mean |Δ rank|:               {delta_mean:.2f}')
        print(f'  Max |Δ rank|:                {int(delta_max)}')
        print(f'  Mean |Δ composite|:          {comp_mae:.3f}')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', nargs='?', help='Model name (full identifier or unique substring), e.g. claude-opus-4.5')
    p.add_argument('--all', action='store_true', help='Print rank-correlation summary across all 3 splits')
    args = p.parse_args()
    if args.all or not args.model:
        return show_summary()
    return show_model(args.model)


if __name__ == '__main__':
    sys.exit(main())
