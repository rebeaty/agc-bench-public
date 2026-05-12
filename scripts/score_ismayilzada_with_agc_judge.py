"""External validation: AGC-Judge on Ismayilzada et al. 2025 (mismayil/creative_story_generation_dataset).

Dataset: 479 short stories (236 human, 243 LLM from 60 different LLMs).
Each story has multi-expert and multi-non-expert ratings for creativity,
originality, surprise, value, plus author guesses (AI vs. human) at both
expert and non-expert levels.

Why this is the right external test:
  - Both human and LLM responses on the SAME task (5-sentence cue-word stories)
  - Multi-rater human gold (expert + non-expert)
  - Public, ungated, downloadable from HuggingFace
  - Independent of the AGC-Bench training distribution

Tests:
  1. Item-level Spearman against expert and non-expert mean creativity
  2. Per-source mean (human vs LLM) under each prompt -> bias direction
  3. AGC-Judge author-guess accuracy vs non-expert author guess (source detection)

Reads:
  - HuggingFace dataset mismayil/creative_story_generation_dataset (auto-fetched)

Writes (to analysis/):
  - ismayilzada_agc_judge_scores.parquet

Inference backend:
  AGC-Judge is hosted at https://huggingface.co/agcbench-2026/AGC-Judge as a
  LoRA adapter on Qwen3-30B-A3B-Instruct-2507. This script uses the Together
  inference API by default (set TOGETHER_API_KEY); to run with a local
  transformers + peft pipeline, replace the `call()` function below with a
  HuggingFace generate call against the merged adapter weights. GPU with
  >= 80 GB memory recommended for local inference on the 30B base.
"""
from __future__ import annotations
import os, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import together

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'analysis/rebuilt'
OUT.mkdir(parents=True, exist_ok=True)
FT = os.environ.get('AGC_JUDGE_MODEL', 'agcbench-2026/AGC-Judge')

PROMPT_A = """Task description: Write a five-sentence creative short story using the given cue words.

Response to rate: {r}

A creative or original response should be:
  - novel (unusual, original, unique)
  - high-quality (well-crafted and appropriate for the task)
  - surprising (unexpected, non-obvious)

Rate the creativity of the story on a scale from 10 (not at all) to 50 (extremely creative). Return only a single integer between 10 and 50, with no other text."""

PROMPT_B = """A panel of human raters evaluated short creative stories on a 10-50 scale, where 10 means "not at all creative" and 50 means "extremely creative." A creative response is novel, high-quality, and surprising.

Task: Write a five-sentence creative short story using the given cue words.

Story: {r}

Predict what the human raters' average rating would be. Return only a single integer between 10 and 50, with no other text."""

PROMPT_C = """A panel of human raters evaluated short creative stories written by human study participants on a 10-50 scale, where 10 means "not at all creative" and 50 means "extremely creative." A creative response is novel, high-quality, and surprising.

Task: Write a five-sentence creative short story using the given cue words.

Human participant's story: {r}

Predict what the human raters' average rating would be for this human-written story. Return only a single integer between 10 and 50, with no other text."""

PROMPT_D = """You will see short creative stories written by either human study participants or by AI language models. AI judges sometimes show self-preference bias, rating LLM responses higher than human responses on the same task. Apply the same scale fairly to both sources.

Task: Write a five-sentence creative short story using the given cue words.

Story: {r}

A creative response is novel, high-quality, and surprising. Predict what a panel of human raters would assign on a 10-50 scale, then identify the likely source.

Reply with exactly this format (no other text):
<integer 10-50>|<human or llm>

Example: 32|human"""


def parse_score(s):
    if not s: return None
    nums = re.findall(r'-?\d+', s.strip())
    if not nums: return None
    for n in reversed(nums):
        v = int(n)
        if 10 <= v <= 50: return v
    return int(nums[-1])


def parse_d(s):
    if not s: return None, None
    score = parse_score(s)
    low = s.lower()
    if 'human' in low and 'llm' not in low:
        label = 'human'
    elif 'llm' in low and 'human' not in low:
        label = 'llm'
    elif 'human' in low and 'llm' in low:
        h_pos = low.rfind('human'); l_pos = low.rfind('llm')
        label = 'human' if h_pos > l_pos else 'llm'
    else:
        label = None
    return score, label


def main():
    print('Loading Ismayilzada dataset from HF...')
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id='mismayil/creative_story_generation_dataset',
                        filename='data/train-00000-of-00001.parquet', repo_type='dataset')
    df = pd.read_parquet(p)
    print(f'  {len(df):,} stories ({(df["author"]=="human").sum()} human, {(df["author"]!="human").sum()} LLM)')

    # Add a clean source label
    df['source'] = df['author'].apply(lambda a: 'human' if a == 'human' else 'llm')

    client = together.Together()

    def call(prompt_text, max_tok=8):
        try:
            r = client.chat.completions.create(model=FT,
                messages=[{'role':'user','content':prompt_text}],
                max_tokens=max_tok, temperature=0)
            return (r.choices[0].message.content or '').strip()
        except Exception as e:
            return f'ERR:{str(e)[:50]}'

    def make_one(template, max_tok=8, parser=parse_score):
        def _one(row):
            raw = call(template.format(r=row['story']), max_tok=max_tok)
            return parser(raw), raw
        return _one

    # Run all 4 prompts
    for label, template, max_tok, parser in [
        ('A', PROMPT_A, 8, parse_score),
        ('B', PROMPT_B, 8, parse_score),
        ('C', PROMPT_C, 8, parse_score),
        ('D', PROMPT_D, 16, parse_d),
    ]:
        fn = make_one(template, max_tok, parser)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(fn, [r for _, r in df.iterrows()]))
        dt = time.time()-t0
        if label == 'D':
            df[f'pred_{label}'] = [r[0][0] if isinstance(r[0], tuple) else r[0] for r in results]
            df[f'label_{label}'] = [r[0][1] if isinstance(r[0], tuple) else None for r in results]
            df[f'raw_{label}'] = [r[1] for r in results]
        else:
            df[f'pred_{label}'] = [r[0] for r in results]
            df[f'raw_{label}'] = [r[1] for r in results]
        ok = df[f'pred_{label}'].notna().sum()
        print(f'  prompt {label}: {dt:.0f}s, ok={ok}/{len(df)}')

    out = OUT / 'ismayilzada_agc_judge_scores.parquet'
    df.to_parquet(out, index=False)
    print(f'\nWrote {out}')

    # === ALIGNMENT vs human gold (mean of expert creativity ratings) ===
    print(f'\n========== ALIGNMENT vs EXPERT CREATIVITY (mean) ==========')
    df['gold_expert'] = df['expert_creativity'].apply(lambda a: float(np.mean(a)) if a is not None and len(a) else np.nan)
    df['gold_nonexpert'] = df['non_expert_creativity'].apply(lambda a: float(np.mean(a)) if a is not None and len(a) else np.nan)

    from scipy.stats import spearmanr, pearsonr
    for ptag in ['A','B','C','D']:
        for gold_col, glabel in [('gold_expert','expert'), ('gold_nonexpert','non-expert')]:
            sub = df.dropna(subset=[f'pred_{ptag}', gold_col])
            if len(sub) < 30: continue
            rho, _ = spearmanr(sub[gold_col], sub[f'pred_{ptag}'])
            r, _ = pearsonr(sub[gold_col], sub[f'pred_{ptag}'])
            print(f'  prompt {ptag} vs {glabel}: rho={rho:+.3f}, r={r:+.3f}, n={len(sub)}')

    # === HUMAN vs LLM BIAS GAP ==========
    print(f'\n========== HUMAN vs LLM BIAS GAP (per prompt) ==========')
    print(f'{"prompt":<6} {"H mean":>7} {"L mean":>7} {"gap":>6}')
    for ptag in ['A','B','C','D']:
        h = df[df['source']=='human'][f'pred_{ptag}'].dropna()
        l = df[df['source']=='llm'][f'pred_{ptag}'].dropna()
        if len(h) and len(l):
            gap = l.mean() - h.mean()
            print(f'{ptag:<6} {h.mean():>7.2f} {l.mean():>7.2f} {gap:>+6.2f}')

    # === SOURCE-DETECTION (prompt D) ==========
    print(f'\n========== SOURCE-DETECTION ACCURACY (prompt D) ==========')
    valid = df.dropna(subset=['label_D']).copy()
    print(f'parseable D-labels: {len(valid)}/{len(df)}')
    valid['correct'] = valid['label_D'] == valid['source']
    print(f'overall accuracy: {valid["correct"].mean():.1%}  (chance = 50%)')
    for src in ['human','llm']:
        sub = valid[valid['source']==src]
        print(f'  {src}: {sub["correct"].mean():.1%}  (n={len(sub):,})')
    print(f'\nConfusion (rows = true, cols = predicted):')
    print(pd.crosstab(valid['source'], valid['label_D'], margins=True).to_string())


if __name__ == '__main__':
    main()
