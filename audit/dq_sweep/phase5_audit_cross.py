"""DQ sweep — Phase 5: cross-reference heuristic flags with the existing
on-task audit (data_quality_llm_judge_v4.csv).

Joins per-cell heuristic stats from cell_dq.csv with per-cell on-task /
garbled rates from the v4 audit. Tags each cell with one of:
  CLEAN          — heuristic + audit both green
  HEURISTIC_ONLY — flagged by heuristics, audit on-task ≥ 50 %
  AUDIT_ONLY     — audit on-task < 50 %, no heuristic flag (subtle off-task)
  BOTH           — flagged by both (highest priority for action)
  AUDIT_MISSING  — cell exists in heuristics but no audit row (rare)

Outputs: dq_sweep/primary_audit_cross.csv, dq_sweep/REPORT.md (updated).
"""
from pathlib import Path

import pandas as pd

BUNDLE = Path(__file__).resolve().parents[2]
OUT = BUNDLE / 'audit/dq_sweep'

# Load cell-level heuristics + primary-release filter
cell = pd.read_csv(OUT / 'cell_dq.csv')
lb = pd.read_csv(BUNDLE / 'release_data/leaderboard.csv')
md = pd.read_csv(BUNDLE / 'release_data/dataset_metadata.csv')
md_inc = md[md['status'] == 'included'].copy()


def encoded_model_to_name(encoded_model):
    s = encoded_model.replace('openrouter_', '', 1) if encoded_model.startswith('openrouter_') else encoded_model
    if '_' not in s:
        return s
    provider, rest = s.split('_', 1)
    rest = rest.replace('_', '-')
    return f'{provider}/{rest}'


cell['model_guess'] = cell['model_id'].apply(encoded_model_to_name)
release_models = set(lb['model'])


def best_match(guess, candidates):
    g = guess.lower().replace('-', '').replace('.', '').replace('_', '')
    for c in candidates:
        cn = c.lower().replace('-', '').replace('.', '').replace('_', '')
        if cn == g:
            return c
    for c in candidates:
        cn = c.lower().replace('-', '').replace('.', '').replace('_', '')
        if g in cn or cn in g:
            return c
    return None


cell['model_lb'] = cell['model_guess'].apply(
    lambda g: best_match(g, release_models)
)
cell = cell[
    cell['scenario'].isin(set(md_inc['dataset'])) & cell['model_lb'].notna()
].copy()
cell = cell.sort_values('n_items', ascending=False).drop_duplicates(
    ['scenario', 'model_lb'], keep='first'
)
print(f'primary-release cells (heuristics): {len(cell):,}')

# Audit
v4 = pd.read_csv(BUNDLE / 'audit/dq_audit/data_quality_llm_judge_v4.csv')
audit_agg = v4.groupby(['dataset', 'model']).agg(
    n_audited=('on_task', 'count'),
    on_task_rate=('on_task', 'mean'),
    garbled_rate=('garbled', 'mean'),
).reset_index()
print(f'audit cells: {len(audit_agg):,}')

# Filter audit to release models + included benchmarks only
audit_agg = audit_agg[
    audit_agg['dataset'].isin(set(md_inc['dataset']))
    & audit_agg['model'].isin(release_models)
]
print(f'audit cells in primary release set: {len(audit_agg):,}')

merged = cell.merge(
    audit_agg, left_on=['scenario', 'model_lb'], right_on=['dataset', 'model'],
    how='outer', indicator=True,
)
print(f'\nmerge state:\n{merged["_merge"].value_counts().to_string()}')


# Define flags
LONGFORM = set(md_inc[md_inc['jrt_corrected']]['dataset'])
STRUCTURED_FORMAT = {'banner_request_400', 'rpgbench', 'speak_to_structure'}


def heur_flag(row):
    if pd.isna(row['n_items']):
        return ''
    flags = []
    if row['empty_rate'] > 0.30:
        flags.append(f'empty {row["empty_rate"] * 100:.0f}%')
    if (row['scenario'] in LONGFORM and row['short_rate'] > 0.80
            and row['mean_chars'] < 30):
        flags.append(f'short {row["short_rate"] * 100:.0f}%')
    if row['refusal_rate'] > 0.30:
        flags.append(f'refusal {row["refusal_rate"] * 100:.0f}%')
    if (row['scenario'] not in STRUCTURED_FORMAT
            and row['repetitive_rate'] > 0.50):
        flags.append(f'repetitive {row["repetitive_rate"] * 100:.0f}%')
    return '; '.join(flags)


def audit_flag(row):
    if pd.isna(row['on_task_rate']):
        return ''
    flags = []
    if row['on_task_rate'] < 0.5:
        flags.append(f'on-task {row["on_task_rate"] * 100:.0f}%')
    if row['garbled_rate'] > 0.5:
        flags.append(f'garbled {row["garbled_rate"] * 100:.0f}%')
    return '; '.join(flags)


merged['heur_issue'] = merged.apply(heur_flag, axis=1)
merged['audit_issue'] = merged.apply(audit_flag, axis=1)


def classify(row):
    h = row['heur_issue'] != ''
    a = row['audit_issue'] != ''
    if pd.isna(row['n_items']) and not pd.isna(row['on_task_rate']):
        return 'AUDIT_ONLY' if a else 'CLEAN'
    if pd.isna(row['on_task_rate']):
        return 'AUDIT_MISSING'
    if h and a:
        return 'BOTH'
    if h:
        return 'HEURISTIC_ONLY'
    if a:
        return 'AUDIT_ONLY'
    return 'CLEAN'


merged['status'] = merged.apply(classify, axis=1)
print(f'\nclassification:\n{merged["status"].value_counts().to_string()}')

# Build the primary-release cross-reference table
ds_col = merged['scenario'].fillna(merged['dataset'])
mdl_col = merged['model_lb'].fillna(merged['model'])
out = pd.DataFrame({
    'model': mdl_col,
    'dataset': ds_col,
    'n_items': merged['n_items'],
    'mean_chars': merged['mean_chars'],
    'empty_rate': merged['empty_rate'],
    'short_rate': merged['short_rate'],
    'refusal_rate': merged['refusal_rate'],
    'repetitive_rate': merged['repetitive_rate'],
    'audit_n': merged['n_audited'],
    'audit_on_task': merged['on_task_rate'],
    'audit_garbled': merged['garbled_rate'],
    'heur_issue': merged['heur_issue'],
    'audit_issue': merged['audit_issue'],
    'status': merged['status'],
})

out.to_csv(OUT / 'primary_audit_cross.csv', index=False)

# Summary by status
both = out[out['status'] == 'BOTH'].sort_values(
    ['model', 'dataset']
)
heur_only = out[out['status'] == 'HEURISTIC_ONLY'].sort_values(
    ['model', 'dataset']
)
audit_only = out[out['status'] == 'AUDIT_ONLY'].sort_values('audit_on_task')

print(f'\n=== BOTH (heuristic + audit agree, highest priority): {len(both)} ===')
print(both.head(40)[['model', 'dataset', 'n_items', 'mean_chars',
                     'audit_on_task', 'heur_issue']].to_string(index=False))

print(f'\n=== AUDIT_ONLY (subtle off-task, missed by heuristics): {len(audit_only)} ===')
print(audit_only.head(30)[['model', 'dataset', 'n_items', 'mean_chars',
                           'audit_on_task', 'audit_garbled']].to_string(index=False))

print(f'\n=== HEURISTIC_ONLY (heuristic flagged, audit said on-task): {len(heur_only)} ===')
print(heur_only.head(20)[['model', 'dataset', 'n_items', 'mean_chars',
                          'heur_issue', 'audit_on_task']].to_string(index=False))

# Update report
n_clean = (out['status'] == 'CLEAN').sum()
n_total = len(out)
report = [
    '# AGC-Bench data-quality sweep — primary release set',
    '',
    'Two independent passes, cross-referenced:',
    '',
    '1. **Heuristic pass** (this sweep) — empty / short / refusal / repetitive '
    'rates from raw `scenario_state.json` files for every (model, dataset) cell.',
    '2. **On-task audit** (existing) — `data_quality_llm_judge_v4.csv`, '
    'three random items per cell rated by `x-ai/grok-4.1-fast` for on-task / garbled.',
    '',
    f'Primary release scope: 83 release models × 67 text-only datasets = 5{chr(44)}561 expected cells.',
    f'Cells in the join: {n_total:,}.',
    '',
    '## Cross-classification',
    '',
    f'| status | count | % |',
    f'|--------|-------|---|',
]
for status, cnt in out['status'].value_counts().items():
    report.append(f'| **{status}** | {cnt:,} | {cnt / n_total * 100:.1f} |')
report += [
    '',
    '- **CLEAN**: heuristic and audit both green.',
    '- **HEURISTIC_ONLY**: heuristic-flagged but audit on-task ≥ 50 %. Often '
    'short-form benchmarks where minimal output is the expected format (e.g. '
    'multiple-choice, close-ended exact-match).',
    '- **AUDIT_ONLY**: audit on-task < 50 %, normal length. Subtle off-task '
    'content the heuristics cannot see.',
    '- **BOTH**: flagged by both signals — highest-priority cells for action '
    '(re-run, exclude, or document).',
    '- **AUDIT_MISSING**: cell ran but did not appear in the audit sample (rare).',
    '',
    f'## Action priority — {len(both)} cells flagged by both signals',
    '',
    '| model | dataset | n | mean chars | audit on-task | heuristic flag |',
    '|-------|---------|---|-----------|---------------|----------------|',
]
for _, row in both.head(50).iterrows():
    report.append(
        f'| {row["model"]} | {row["dataset"]} | {row["n_items"]:.0f} | '
        f'{row["mean_chars"]:.0f} | {row["audit_on_task"] * 100:.0f}% | '
        f'{row["heur_issue"]} |'
    )

(OUT / 'REPORT.md').write_text('\n'.join(report))
print(f'\nWrote: {OUT / "primary_audit_cross.csv"}')
print(f'Wrote: {OUT / "REPORT.md"}')
