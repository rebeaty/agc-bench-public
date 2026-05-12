"""DQ sweep — Phase 4: refine flags, filter to the primary release set.

Phase 3 raised 6,121 alarms — most were `finish_reason=None` on cells whose
text was actually fine (OpenRouter's API often doesn't populate the field).
This pass restricts to the primary release set (83 release models × 67 text-only
benchmarks), drops the noisy axis, and emits a tight per-cell report with
rationale. Output: dq_sweep/primary_flags.csv, dq_sweep/REPORT.md.
"""
from pathlib import Path

import pandas as pd

BUNDLE = Path(__file__).resolve().parents[2]
OUT = BUNDLE / 'audit/dq_sweep'

cell = pd.read_csv(OUT / 'cell_dq.csv')
lb = pd.read_csv(BUNDLE / 'release_data/leaderboard.csv')
md = pd.read_csv(BUNDLE / 'release_data/dataset_metadata.csv')
md_inc = md[md['status'] == 'included'].copy()


def encoded_model_to_name(encoded_model):
    """Map HELM/OpenRouter encoded model IDs back to vendor/model names."""
    s = encoded_model.replace('openrouter_', '', 1) if encoded_model.startswith('openrouter_') else encoded_model
    if '_' not in s:
        return s
    provider, rest = s.split('_', 1)
    rest = rest.replace('_', '-')
    return f'{provider}/{rest}'


cell['model_guess'] = cell['model_id'].apply(encoded_model_to_name)


# Map model_guess to canonical leaderboard model name (handles small naming
# differences like 3-1 vs 3.1, hyphenation drift).
def best_match(guess, candidates):
    guess_norm = guess.lower().replace('-', '').replace('.', '').replace('_', '')
    for c in candidates:
        c_norm = c.lower().replace('-', '').replace('.', '').replace('_', '')
        if c_norm == guess_norm:
            return c
    # Looser: substring (provider/ name without dots)
    for c in candidates:
        c_norm = c.lower().replace('-', '').replace('.', '').replace('_', '')
        if guess_norm in c_norm or c_norm in guess_norm:
            return c
    return None


release_models = set(lb['model'])
cell['model_lb'] = cell['model_guess'].apply(
    lambda g: best_match(g, release_models)
)

# Filter to primary release set
included_ds = set(md_inc['dataset'])
release_cells = cell[
    cell['scenario'].isin(included_ds) & cell['model_lb'].notna()
].copy()
print(f'Primary-release cells in sweep: {len(release_cells):,} '
      f'(target = 83 × 67 = {83 * 67})')

# Dedupe at release-model level (some models may appear under multiple encoded forms)
release_cells = release_cells.sort_values('n_items', ascending=False)
release_cells = release_cells.drop_duplicates(['scenario', 'model_lb'], keep='first')
print(f'After dedupe by (scenario, release model): {len(release_cells):,}')


# Real-flag definition: drop finish_none (false positives from API metadata
# inconsistency). Allow short-rate flag only on benchmarks that expect a
# long-form response (LLM-judge canonical metrics, ie jrt_corrected = True).
LONGFORM = set(md_inc[md_inc['jrt_corrected']]['dataset'])
# Known structured-output benchmarks where high repetition is expected (banners,
# RPG sheets) — exempt from repetitive_rate flag.
STRUCTURED_FORMAT = {'banner_request_400', 'rpgbench', 'speak_to_structure'}


def flag(row):
    flags = []
    if row['empty_rate'] > 0.30:
        flags.append(f'empty {row["empty_rate"] * 100:.0f}%')
    if row['scenario'] in LONGFORM:
        if row['short_rate'] > 0.80 and row['mean_chars'] < 30:
            flags.append(f'short {row["short_rate"] * 100:.0f}% mean={row["mean_chars"]:.0f}c')
    if row['refusal_rate'] > 0.30:
        flags.append(f'refusal {row["refusal_rate"] * 100:.0f}%')
    if (row['scenario'] not in STRUCTURED_FORMAT
            and row['repetitive_rate'] > 0.50):
        flags.append(f'repetitive {row["repetitive_rate"] * 100:.0f}%')
    return '; '.join(flags) if flags else ''


release_cells['issue'] = release_cells.apply(flag, axis=1)
flagged = release_cells[release_cells['issue'] != ''].sort_values(
    ['model_lb', 'scenario']
)
print(f'\nReal-flag cells in primary release set: {len(flagged):,}')

flagged_out = flagged[['model_lb', 'scenario', 'n_items', 'mean_chars',
                       'median_chars', 'empty_rate', 'short_rate',
                       'refusal_rate', 'repetitive_rate', 'issue', 'path']]
flagged_out = flagged_out.rename(columns={'model_lb': 'model', 'scenario': 'dataset'})
flagged_out.to_csv(OUT / 'primary_flags.csv', index=False)

# By model
print('\n--- by release model ---')
by_model = flagged.groupby('model_lb').size().sort_values(ascending=False)
print(by_model.head(20).to_string())

# By dataset
print('\n--- by dataset ---')
by_ds = flagged.groupby('scenario').size().sort_values(ascending=False)
print(by_ds.head(20).to_string())

# Primary-result report
report_lines = ['# AGC-Bench data-quality sweep — primary release set\n',
                f'Scope: 83 release models × 67 text-only datasets = 5,561 cells.\n',
                f'Cells located in run dirs: {len(release_cells):,}.\n',
                f'Cells flagged with a real-content issue: '
                f'**{len(flagged):,}** ({len(flagged) / max(len(release_cells), 1) * 100:.1f}%).\n',
                '\n## Flag axes\n',
                '- **empty** — > 30 % responses with zero characters\n',
                '- **short** — > 80 % responses < 20 chars on a long-form '
                '(LLM-judge) benchmark, with mean response < 30 chars\n',
                '- **refusal** — > 30 % responses match a refusal pattern\n',
                '- **repetitive** — > 50 % responses dominated by a single '
                '5-char shingle (excludes banner / rpg / structured-format benchmarks)\n',
                '\n## Flagged cells by release model (count)\n```',
                by_model.head(20).to_string(),
                '```\n\n## Flagged cells by dataset (count)\n```',
                by_ds.head(20).to_string(), '```\n']

(OUT / 'REPORT.md').write_text('\n'.join(report_lines))
print(f'\nWrote: {OUT / "primary_flags.csv"}')
print(f'Wrote: {OUT / "REPORT.md"}')

# Top 30 worst by issue count + severity preview
print('\n=== preview of primary_flags.csv (first 30 rows) ===')
print(flagged_out.head(30).to_string(index=False))
