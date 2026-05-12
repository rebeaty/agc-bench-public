"""DQ sweep -- Phase 2: response-side inspection of suspicious cells.

Given a list of (model, dataset) cells flagged in Phase 1 (extreme z, low n,
near-constant), find the corresponding scenario_state.json, sample responses,
and compute heuristic flags: empty, refusal, truncation, repetition. Outputs
a JSON report with per-cell stats and exemplar responses for review.

Reads HELM outputs from benchmark_output/runs/ in this checkout. Run
eval_scripts/00_run_all_parallel.sh first if those outputs are absent.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BUNDLE = Path(__file__).resolve().parents[2]
RUNS_ROOT = BUNDLE / 'benchmark_output/runs'
OUT = BUNDLE / 'audit/dq_sweep'
OUT.mkdir(parents=True, exist_ok=True)


REFUSAL_PATTERNS = [
    r"^i (?:can(?:not|'t)|am unable|won't|will not)\b",
    r"^i'm (?:sorry|unable|not able)\b",
    r"^as an? (?:ai|language model|assistant)\b",
    r"^sorry,? i ",
    r"\bi cannot (?:fulfill|comply|create|generate|provide)\b",
    r"\bi'm not able to (?:create|generate|provide|comply)\b",
    r"^this (?:request|content|task) (?:violates|cannot)\b",
]
REFUSAL_RE = re.compile('|'.join(REFUSAL_PATTERNS), re.I)


def model_id(model_str):
    """openrouter_mistralai_mistral_small_3_1_24b_instruct →
    mistralai/mistral-small-3.1-24b-instruct (best-effort reversal)."""
    s = model_str.replace('openrouter_', '', 1)
    if '_' not in s:
        return s
    provider, rest = s.split('_', 1)
    rest = rest.replace('_', '-').replace('--', '-')
    return f'{provider}/{rest}'


def find_runs(dataset, model_id_target):
    """Find all scenario_state.json files for a (dataset, model) cell."""
    target = model_id_target.replace('/', '_').replace('-', '_').replace('.', '_')
    target_norm = re.sub(r'_+', '_', target).strip('_').lower()
    matches = []
    for run_dir in RUNS_ROOT.iterdir():
        name = run_dir.name.lower()
        if target_norm not in name and target_norm.replace('_', '') not in name.replace('_', ''):
            continue
        for sub in run_dir.glob(f'{dataset}:model=*'):
            ss = sub / 'scenario_state.json'
            if ss.exists():
                matches.append(ss)
    return matches


def is_repetitive(text, threshold=0.4):
    """Heuristic: if any 5-char shingle covers > threshold of text length."""
    if len(text) < 50:
        return False
    shingles = Counter(text[i:i + 5] for i in range(len(text) - 4))
    if not shingles:
        return False
    top, ct = shingles.most_common(1)[0]
    return (ct * 5) / len(text) > threshold


def is_truncated(rs):
    return rs.get('result', {}).get('completions', [{}])[0].get(
        'finish_reason', ''
    ).lower() == 'length'


def extract_text(rs):
    completions = rs.get('result', {}).get('completions', [])
    if not completions:
        return ''
    return (completions[0].get('text') or '').strip()


def inspect_file(ss_path, n_examples=3):
    """Compute per-cell heuristics + sample examples from one scenario_state."""
    try:
        d = json.loads(ss_path.read_text())
    except Exception as e:
        return {'error': f'parse: {e}'}
    rss = d.get('request_states', [])
    n = len(rss)
    if n == 0:
        return {'n_items': 0, 'error': 'no request_states'}

    flags = {'empty': 0, 'short': 0, 'refusal': 0, 'truncated': 0,
             'repetitive': 0}
    char_lens = []
    finish_reasons = Counter()
    examples = {'empty': [], 'refusal': [], 'truncated': [], 'normal': []}
    for rs in rss:
        text = extract_text(rs)
        char_lens.append(len(text))
        fr_raw = rs.get('result', {}).get('completions', [{}])[0].get(
            'finish_reason', '?'
        )
        # finish_reason is sometimes a dict (e.g. {'reason': 'stop'}) instead of a string
        if isinstance(fr_raw, dict):
            fr = str(fr_raw.get('reason', fr_raw.get('finish_reason', '?')))
        else:
            fr = str(fr_raw)
        finish_reasons[fr] += 1
        if len(text) == 0:
            flags['empty'] += 1
            if len(examples['empty']) < n_examples:
                examples['empty'].append('')
        elif len(text) < 20:
            flags['short'] += 1
        if REFUSAL_RE.search(text):
            flags['refusal'] += 1
            if len(examples['refusal']) < n_examples:
                examples['refusal'].append(text[:300])
        if fr.lower() == 'length':
            flags['truncated'] += 1
            if len(examples['truncated']) < n_examples:
                examples['truncated'].append(text[-300:])
        if is_repetitive(text):
            flags['repetitive'] += 1
        if (len(text) > 50 and not REFUSAL_RE.search(text)
                and fr.lower() != 'length' and not is_repetitive(text)
                and len(examples['normal']) < n_examples):
            examples['normal'].append(text[:300])

    return {
        'n_items': n,
        'mean_chars': sum(char_lens) / n if n else 0,
        'median_chars': sorted(char_lens)[n // 2] if n else 0,
        'min_chars': min(char_lens) if char_lens else 0,
        'flags': flags,
        'rates': {k: v / n for k, v in flags.items()},
        'finish_reasons': dict(finish_reasons),
        'examples': examples,
    }


SUSPECTS = [
    # Phase-1 extreme-z cells (top 10 by |z|)
    ('arn', 'mistralai/mistral-small-3.1-24b-instruct'),
    ('hypobench', 'mistralai/mistral-small-3.1-24b-instruct'),
    ('unfun_corpus', 'google/gemma-2-27b-it'),
    ('sdat', 'mistralai/mistral-small-3.1-24b-instruct'),
    ('story_quality', 'anthropic/claude-opus-4.6'),
    ('grapheval_review_advisor', 'mistralai/mistral-small-3.1-24b-instruct'),
    ('unfun_corpus', 'nvidia/nemotron-3-nano-30b-a3b'),
    ('grapheval_iclr', 'mistralai/mistral-small-3.1-24b-instruct'),
    ('grapheval_ai_researcher', 'mistralai/mistral-small-3.1-24b-instruct'),
    ('grapheval_ai_researcher', 'deepcogito/cogito-v2.1-671b'),
    # Phase-1 low-n / near-constant cells
    ('conceptual_design', 'anthropic/claude-haiku-4.5'),
    ('arena_hard_creative', 'meta-llama/llama-3.2-3b-instruct'),
    ('cue_word_story', 'z-ai/glm-5.1'),
    ('cue_word_story', 'ibm-granite/granite-4.0-h-micro'),
    ('cue_word_story', 'z-ai/glm-5-turbo'),
    ('conceptual_design', 'morph/morph-v3-fast'),
    ('conceptual_design', 'qwen/qwen3-next-80b-a3b-instruct'),
    # Sanity: a known-good top-of-leaderboard cell
    ('aidanbench', 'anthropic/claude-opus-4.7'),
    ('cue_word_story', 'anthropic/claude-opus-4.7'),
]


def main():
    report = {}
    for ds, mdl in SUSPECTS:
        print(f'  {ds:<28} {mdl}')
        runs = find_runs(ds, mdl)
        if not runs:
            report[f'{ds} | {mdl}'] = {'error': 'no runs found'}
            continue
        # Pick the run with most items (canonical shipped)
        best = None
        best_n = -1
        for r in runs:
            try:
                head = json.loads(r.read_text())
                n = len(head.get('request_states', []))
                if n > best_n:
                    best_n = n
                    best = r
            except Exception:
                continue
        if best is None:
            report[f'{ds} | {mdl}'] = {'error': 'all runs failed parse'}
            continue
        res = inspect_file(best)
        res['_source_run'] = str(best.relative_to(RUNS_ROOT))
        res['_n_runs_found'] = len(runs)
        report[f'{ds} | {mdl}'] = res

    out_path = OUT / 'phase2_suspect_cells.json'
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f'\nWrote {out_path}')

    print('\n' + '=' * 70)
    print('Summary (cells with anomalous flag rates)')
    print('=' * 70)
    rows = []
    for key, r in report.items():
        if 'error' in r:
            print(f'  {key:<70} ERROR: {r["error"]}')
            continue
        rates = r['rates']
        rows.append({
            'cell': key,
            'n': r['n_items'],
            'mean_chars': round(r['mean_chars']),
            'empty%': f"{rates['empty'] * 100:.0f}",
            'short%': f"{rates['short'] * 100:.0f}",
            'refusal%': f"{rates['refusal'] * 100:.0f}",
            'truncated%': f"{rates['truncated'] * 100:.0f}",
            'repetitive%': f"{rates['repetitive'] * 100:.0f}",
        })
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        df.to_csv(OUT / 'phase2_suspect_summary.csv', index=False)


if __name__ == '__main__':
    main()
