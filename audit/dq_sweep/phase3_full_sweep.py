"""DQ sweep -- Phase 3: full scan over primary-release scenario_state files.

Walks every benchmark_output/runs/<suite>/<scenario>:model=*/scenario_state.json,
extracts per-file response heuristics, dedupes by (scenario, model) keeping the
largest n_items run, and writes a per-cell DQ table to dq_sweep/cell_dq.csv.

Heuristics per cell:
  n_items                — number of request_states with completions
  mean_chars             — mean response length
  median_chars
  empty_rate             — pct with len(text) == 0
  short_rate             — pct with 0 < len(text) < 20
  refusal_rate           — pct matching common refusal patterns
  truncated_rate         — pct with finish_reason == 'length'
  finish_none_rate       — pct with finish_reason == 'None' / null (decoder fail)
  repetitive_rate        — pct with > 40% of text covered by one 5-char shingle

Reads HELM outputs from benchmark_output/runs/ in this checkout and uses
multiprocessing for speed.
"""
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BUNDLE = Path(__file__).resolve().parents[2]
RUNS_ROOT = BUNDLE / 'benchmark_output/runs'
OUT = BUNDLE / 'audit/dq_sweep'
OUT.mkdir(parents=True, exist_ok=True)
N_WORKERS = max(4, (os.cpu_count() or 8) - 2)

REFUSAL_RE = re.compile(
    r"^i (?:can(?:not|'t)|am unable|won't|will not)\b"
    r"|^i'm (?:sorry|unable|not able)\b"
    r"|^as an? (?:ai|language model|assistant)\b"
    r"|^sorry,? i "
    r"|\bi cannot (?:fulfill|comply|create|generate|provide)\b"
    r"|\bi'm not able to (?:create|generate|provide|comply)\b",
    re.I,
)


def is_repetitive(text):
    if len(text) < 50:
        return False
    shingles = Counter(text[i:i + 5] for i in range(len(text) - 4))
    if not shingles:
        return False
    _, ct = shingles.most_common(1)[0]
    return (ct * 5) / len(text) > 0.4


def extract_text(rs):
    completions = rs.get('result', {}).get('completions', [])
    if not completions:
        return ''
    return (completions[0].get('text') or '').strip()


def fr_string(rs):
    completions = rs.get('result', {}).get('completions', [])
    if not completions:
        return 'no_completion'
    fr = completions[0].get('finish_reason')
    if fr is None:
        return 'None'
    if isinstance(fr, dict):
        return str(fr.get('reason', fr.get('finish_reason', '?')))
    return str(fr)


def inspect(ss_path):
    """Return per-cell row dict, or None on parse error."""
    try:
        d = json.loads(ss_path.read_text())
    except Exception:
        return None
    rss = d.get('request_states', [])
    n = len(rss)
    if n == 0:
        return None
    counts = {'empty': 0, 'short': 0, 'refusal': 0, 'truncated': 0,
              'repetitive': 0, 'finish_none': 0}
    char_lens = []
    fr_counter = Counter()
    for rs in rss:
        text = extract_text(rs)
        char_lens.append(len(text))
        fr = fr_string(rs)
        fr_counter[fr] += 1
        if len(text) == 0:
            counts['empty'] += 1
        elif len(text) < 20:
            counts['short'] += 1
        if REFUSAL_RE.search(text):
            counts['refusal'] += 1
        if fr.lower() == 'length':
            counts['truncated'] += 1
        if fr in ('None', 'null', 'no_completion', '?'):
            counts['finish_none'] += 1
        if is_repetitive(text):
            counts['repetitive'] += 1

    parts = ss_path.parent.name  # e.g. "aidanbench:model=google_gemini-3-flash-preview"
    if ':model=' in parts:
        scenario, model = parts.split(':model=', 1)
    else:
        scenario, model = parts, 'unknown'
    char_lens_sorted = sorted(char_lens)
    median = char_lens_sorted[n // 2]

    return {
        'scenario': scenario,
        'model_id': model,
        'run_suite': ss_path.parent.parent.name,
        'n_items': n,
        'mean_chars': sum(char_lens) / n,
        'median_chars': median,
        'min_chars': char_lens_sorted[0],
        'empty_rate': counts['empty'] / n,
        'short_rate': counts['short'] / n,
        'refusal_rate': counts['refusal'] / n,
        'truncated_rate': counts['truncated'] / n,
        'finish_none_rate': counts['finish_none'] / n,
        'repetitive_rate': counts['repetitive'] / n,
        'fr_dist': dict(fr_counter),
        'path': str(ss_path.relative_to(RUNS_ROOT)),
    }


def gather_paths():
    print('scanning runs/ ...')
    paths = list(RUNS_ROOT.glob('*/*:model=*/scenario_state.json'))
    print(f'  found {len(paths):,} scenario_state.json files')
    return paths


def main():
    paths = gather_paths()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(inspect, p): p for p in paths}
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            if r:
                rows.append(r)
            if i % 2000 == 0:
                print(f'  {i:,} / {len(paths):,}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'cell_dq_raw.csv', index=False)
    print(f'\nraw rows: {len(df):,}')

    # Dedupe by (scenario, model_id) keeping the row with largest n_items
    df_sorted = df.sort_values('n_items', ascending=False)
    canonical = df_sorted.drop_duplicates(['scenario', 'model_id'], keep='first')
    canonical.to_csv(OUT / 'cell_dq.csv', index=False)
    print(f'canonical (max-n) rows: {len(canonical):,}')

    print('\n' + '=' * 70)
    print('Cells with empty_rate > 50%')
    print('=' * 70)
    bad_empty = canonical[canonical['empty_rate'] > 0.5].sort_values(
        'empty_rate', ascending=False
    )
    print(f'count: {len(bad_empty)}')
    print(bad_empty[['scenario', 'model_id', 'n_items', 'mean_chars',
                     'empty_rate', 'short_rate', 'finish_none_rate']].head(30).to_string(index=False))

    print('\n' + '=' * 70)
    print('Cells with short_rate > 80% (responses dominated by <20 char outputs)')
    print('=' * 70)
    bad_short = canonical[canonical['short_rate'] > 0.8].sort_values(
        'short_rate', ascending=False
    )
    print(f'count: {len(bad_short)}')
    print(bad_short[['scenario', 'model_id', 'n_items', 'mean_chars',
                     'short_rate', 'finish_none_rate']].head(30).to_string(index=False))

    print('\n' + '=' * 70)
    print('Cells with finish_none_rate > 50% (decoder/API failure pattern)')
    print('=' * 70)
    bad_fr = canonical[canonical['finish_none_rate'] > 0.5].sort_values(
        'finish_none_rate', ascending=False
    )
    print(f'count: {len(bad_fr)}')
    print(bad_fr[['scenario', 'model_id', 'n_items', 'mean_chars',
                  'finish_none_rate', 'short_rate']].head(30).to_string(index=False))

    print('\n' + '=' * 70)
    print('Cells with refusal_rate > 30%')
    print('=' * 70)
    bad_ref = canonical[canonical['refusal_rate'] > 0.3].sort_values(
        'refusal_rate', ascending=False
    )
    print(f'count: {len(bad_ref)}')
    print(bad_ref[['scenario', 'model_id', 'n_items', 'refusal_rate']].head(30).to_string(index=False))

    print('\n' + '=' * 70)
    print('Cells with repetitive_rate > 50%')
    print('=' * 70)
    bad_rep = canonical[canonical['repetitive_rate'] > 0.5].sort_values(
        'repetitive_rate', ascending=False
    )
    print(f'count: {len(bad_rep)}')
    print(bad_rep[['scenario', 'model_id', 'n_items', 'repetitive_rate', 'mean_chars']].head(30).to_string(index=False))

    # Aggregate flags into a single DQ status
    canonical['n_flag_axes'] = (
        (canonical['empty_rate'] > 0.5).astype(int)
        + (canonical['short_rate'] > 0.8).astype(int)
        + (canonical['finish_none_rate'] > 0.5).astype(int)
        + (canonical['refusal_rate'] > 0.3).astype(int)
        + (canonical['repetitive_rate'] > 0.5).astype(int)
    )
    canonical['flag'] = canonical['n_flag_axes'] > 0
    canonical.to_csv(OUT / 'cell_dq.csv', index=False)
    print(f'\ntotal cells flagged on >=1 axis: {canonical["flag"].sum():,} / {len(canonical):,}')


if __name__ == '__main__':
    main()
