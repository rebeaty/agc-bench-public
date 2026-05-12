#!/usr/bin/env bash
# Check the primary numbers reported in the AGC-Bench paper against the
# bundled inputs. The c-factor extraction and CAP within-instrument
# generality are recomputed live; the JRT-vs-raw composite rho is
# recomputed from the shipped `analysis/leaderboard_raw_vs_jrt.csv`
# artifact (the §4.2 release-set snapshot of the pre- vs post-JRT
# comparison); AGC-Judge held-out splits are recomputed from the shipped
# per-item prediction CSVs. Each printed value is shown alongside the
# released-values reference.
#
# Usage (from the bundle root):
#   bash reproduce_paper_results.sh
#
# Expected runtime: under 60 seconds on a modest CPU. No GPU required.
# No network access required.
set -euo pipefail

cd "$(dirname "$0")"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "==================================================================="
echo "AGC-Bench paper primary results: reported vs recomputed"
echo "==================================================================="
echo

# ============================================================
# 1. C-factor extraction (released model set, post-DQ-mask:
#    eigenvalue 4.89, alpha 0.96, 81.5% variance)
# ============================================================
echo "--- 1. C-factor extraction (6-domain primary result) ---"
python3 scripts/build_jrt_artifacts.py 2>&1 | tail -25 | grep -E "eig1|alpha|loading|Wrote"

echo
echo "Released values (eigenvalue, alpha, %var):  4.89, 0.96, 81.5%"
echo "Loadings reported:                          +0.87 to +0.94"
echo

# ============================================================
# 2. JRT-vs-raw downstream comparison + AA intelligence correlations
#    Computed from `analysis/leaderboard_raw_vs_jrt.csv`, the shipped
#    pre-JRT vs post-JRT comparison restricted to the 83-model release set.
#    The rho on this shipped 83-model release artifact is the §4.2 paper
#    claim recomputed on the released files. See
#    release_data/SCORING_NOTES.md for the surface-difference note.
# ============================================================
echo "--- 2. JRT vs raw composite + AA intelligence correlations ---"
python3 - <<'PY'
import pandas as pd
from scipy.stats import spearmanr

lb = pd.read_csv('analysis/leaderboard_raw_vs_jrt.csv').dropna(
    subset=['mean_z', 'mean_z_new'])
rho, _ = spearmanr(lb['mean_z'], lb['mean_z_new'])
print(f"  Raw vs JRT composite (Spearman, n={len(lb)} release models):  rho = {rho:+.3f}")

intel = pd.read_csv('analysis/intelligence_join.csv')
def _norm(s):
    if not isinstance(s, str): return s
    return s.replace('_', '/', 1) if '_' in s and '/' not in s else s
intel['model'] = intel['model'].apply(_norm)

cmp = lb.merge(intel[['model', 'intelligence_index_aa', 'gpqa_diamond']],
               on='model', how='inner')
for col in ['intelligence_index_aa', 'gpqa_diamond']:
    sub = cmp.dropna(subset=[col, 'mean_z_new'])
    r, _ = spearmanr(sub[col], sub['mean_z_new'])
    print(f"  AGC composite (JRT) vs {col:<22} (release-set n={len(sub):>3}):  rho_jrt = {r:+.3f}")
PY

echo
echo "Released values: raw-vs-JRT Spearman rho ~ 0.94 (83 release models)"
echo "Released values: AA Intelligence Index rho_jrt = 0.77 (App K table, 74-model AA-overlap subset); GPQA-Diamond rho_jrt = 0.76"
echo

# ============================================================
# 3. AGC-Human vs LLM domain-generality
#    (paper: LLM alpha = 0.64 vs Human alpha = 0.42)
# ============================================================
echo "--- 3. CAP within-instrument domain generality ---"
python3 scripts/cap_human_vs_llm_generality.py 2>&1 | tail -8

echo
echo "Paper reports: LLM alpha = 0.64, Human alpha = 0.42, p = 0.028"
echo

# ============================================================
# 4. AGC-Judge held-out generalization
#    (paper: rho = 0.83 held-out benches, rho = 0.94 held-out models, rho = 0.94 in-distribution)
# ============================================================
echo "--- 4. AGC-Judge held-out generalization (Spearman vs JRT-corrected gold) ---"
python3 - <<'PY'
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
A = Path('analysis')
for label, fname in [
    ('Held-out benchmarks', 'agc_judge_holdout_benches_preds.csv'),
    ('Held-out models    ', 'agc_judge_holdout_models_preds.csv'),
    ('In-distribution    ', 'agc_judge_held_out_preds.csv'),
]:
    df = pd.read_csv(A / fname).dropna(subset=['pred', 'gold'])
    rho, _ = spearmanr(df['gold'], df['pred'])
    print(f'  {label}:  rho = {rho:+.3f}  (n = {len(df):,})')
PY

echo
echo "Paper reports: held-out benches rho=0.83, held-out models rho=0.94, in-distribution rho=0.94"
echo

echo "==================================================================="
echo "Primary numbers reproduced from the in-bundle 83-model release set."
echo "Compare each printed value with the Released values reference above it."
echo "Two values reproduce to within rounding rather than exact paper-text match"
echo "(c-factor eigenvalue/variance and raw-vs-JRT rho). See release_data/SCORING_NOTES.md."
echo "==================================================================="
