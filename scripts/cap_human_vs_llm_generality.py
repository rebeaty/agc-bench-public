"""Within-instrument test of LLM vs human creative-domain generality.

Same five CAP tasks (AUT, SCTT, Design, Metaphor, Story), same three-component
common-scale composite (diversity, DSI, surprise), administered to both 224
humans and 125 frontier LLMs. We compute the 5x5 inter-task correlation matrix
within each population and compare four standard psychometric indicators:
mean off-diagonal r, first eigenvalue, first-factor explained variance,
and Cronbach's alpha. Bootstrap test (10,000 within-group resamples) gives
two-sided p-values for the LLM minus human difference.

Reads (from bundle-relative paths):
  - analysis/cap_human_per_task.csv   (per-entity per-task common-scale composite)

Writes (to analysis/):
  - cap_generality_human_vs_llm.csv   (bootstrap test results)

Reported in the Discussion of the AGC-Bench paper as the formal test for the
within-instrument domain-generality question (more vs. less domain-general in
LLMs vs. humans).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / 'analysis/cap_human_per_task.csv'
OUT = REPO / 'analysis/rebuilt/cap_generality_human_vs_llm.csv'

TASKS = ['AUT', 'Design', 'Metaphor', 'SCTT', 'Story']


def stats_block(X: np.ndarray) -> dict:
    Xz = (X - X.mean(0)) / X.std(0, ddof=1)
    cm = np.corrcoef(Xz.T)
    eigs = np.sort(np.linalg.eigvalsh(cm))[::-1]
    pct = eigs[0] / eigs.sum() * 100
    k = X.shape[1]
    iv = Xz.var(0, ddof=1).sum()
    tv = Xz.sum(1).var(ddof=1)
    alpha = (k / (k - 1)) * (1 - iv / tv) if tv > 0 else 0.0
    off_diag = cm[np.triu_indices_from(cm, k=1)]
    return {
        'n': len(X),
        'mean_off_diag_r': float(off_diag.mean()),
        'first_eigenvalue': float(eigs[0]),
        'pct_var_first': float(pct),
        'cronbach_alpha': float(alpha),
        'corr_matrix': cm,
    }


def bootstrap(X: np.ndarray, n_iter: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    out = {'mean_off_diag_r': [], 'first_eigenvalue': [], 'pct_var_first': [], 'cronbach_alpha': []}
    n, k = X.shape
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        Xb = X[idx]
        std = Xb.std(0, ddof=1)
        if (std == 0).any():
            continue
        Xz = (Xb - Xb.mean(0)) / std
        cm = np.corrcoef(Xz.T)
        e = np.sort(np.linalg.eigvalsh(cm))[::-1]
        out['mean_off_diag_r'].append(cm[np.triu_indices_from(cm, k=1)].mean())
        out['first_eigenvalue'].append(e[0])
        out['pct_var_first'].append(e[0] / e.sum() * 100)
        iv = Xz.var(0, ddof=1).sum()
        tv = Xz.sum(1).var(ddof=1)
        out['cronbach_alpha'].append((k / (k - 1)) * (1 - iv / tv) if tv > 0 else 0.0)
    return {k: np.array(v) for k, v in out.items()}


def main():
    df = pd.read_csv(SRC)
    df_h = df[df['entity_type'] == 'human'].dropna(subset=TASKS)
    df_l = df[df['entity_type'] != 'human'].dropna(subset=TASKS)
    Xh = df_h[TASKS].values
    Xl = df_l[TASKS].values

    h = stats_block(Xh)
    l = stats_block(Xl)

    print(f'HUMANS (n={h["n"]}):')
    print(f'  mean off-diag r = {h["mean_off_diag_r"]:+.3f}')
    print(f'  first eigenvalue = {h["first_eigenvalue"]:.3f}')
    print(f'  %var first factor = {h["pct_var_first"]:.1f}')
    print(f'  Cronbach alpha = {h["cronbach_alpha"]:.3f}')
    print(f'  correlation matrix:')
    print(pd.DataFrame(h['corr_matrix'], index=TASKS, columns=TASKS).round(3).to_string())

    print(f'\nLLMs (n={l["n"]}):')
    print(f'  mean off-diag r = {l["mean_off_diag_r"]:+.3f}')
    print(f'  first eigenvalue = {l["first_eigenvalue"]:.3f}')
    print(f'  %var first factor = {l["pct_var_first"]:.1f}')
    print(f'  Cronbach alpha = {l["cronbach_alpha"]:.3f}')
    print(f'  correlation matrix:')
    print(pd.DataFrame(l['corr_matrix'], index=TASKS, columns=TASKS).round(3).to_string())

    print('\n========== BOOTSTRAP TEST (10,000 within-group resamples) ==========')
    bh = bootstrap(Xh)
    bl = bootstrap(Xl)
    rows = []
    for stat in ['mean_off_diag_r', 'first_eigenvalue', 'pct_var_first', 'cronbach_alpha']:
        diff = bl[stat] - bh[stat]
        ci_lo, ci_hi = np.percentile(diff, [2.5, 97.5])
        p_two = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
        print(f'  {stat:<22}: LLM={l[stat]:.3f} Human={h[stat]:.3f} '
              f'diff={l[stat]-h[stat]:+.3f}  CI[{ci_lo:+.3f}, {ci_hi:+.3f}]  p={p_two:.4f}')
        rows.append({
            'statistic': stat,
            'llm_value': l[stat], 'human_value': h[stat],
            'difference': l[stat] - h[stat],
            'ci_low_95': ci_lo, 'ci_high_95': ci_hi,
            'two_sided_p': p_two,
        })

    out_df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(f'\nWrote {OUT}')


if __name__ == '__main__':
    main()
