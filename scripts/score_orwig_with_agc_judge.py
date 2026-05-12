"""External validation: AGC-Judge on the Orwig creative writing dataset.

Dataset: 718 short stories on six 3-word cue triplets, with conditions
human (300), GPT-3 (298), GPT-4 (120). Each story has:
  - creativity (human-rater mean)
  - dsi (divergent semantic integration)
  - GPT_score (an LLM-judge rating from the original Orwig pipeline)
  - word_count, perceptual_details

Why this matters: in the Orwig data, human raters do NOT show a strong
source bias (humans 2.93 vs. GPT-4 2.94 mean creativity), so this is a
fairer external test of AGC-Judge alignment with humans than Ismayilzada
(where experts strongly preferred humans). The existing GPT_score column
gives us an LLM-judge baseline to compare against.

Tests:
  1. AGC-Judge prompt C correlation with human creativity (Spearman + Pearson)
  2. AGC-Judge prompt C correlation with the existing GPT_score
  3. Steiger Z (Williams' modification) for whether the two correlations differ
  4. Per-condition mean (AGC-Judge bias vs human/GPT-3/GPT-4)
  5. Within-condition correlations (controlling for source)

Reads (third-party data, not redistributed in this bundle):
  - data/external/orwig_2024/cw_data.csv

The Orwig 2024 creative-writing corpus is not redistributed under this bundle.
Obtain from Orwig et al. 2024 (citation in paper) and place at the path above
before running this script. Columns expected: prompt, response, condition,
creativity, GPT_score, dsi, word_count, perceptual_details.

Writes (to analysis/):
  - orwig_agc_judge_promptC.parquet

Inference backend:
  AGC-Judge is hosted at https://huggingface.co/agcbench-2026/AGC-Judge as a
  LoRA adapter on Qwen3-30B-A3B-Instruct-2507. This script uses the Together
  inference API by default (set TOGETHER_API_KEY); to run with a local
  transformers + peft pipeline, replace the `call()` function below with a
  HuggingFace generate call against the merged adapter weights. GPU with
  >= 80 GB memory recommended for local inference on the 30B base.
"""
from __future__ import annotations
import os, re, time, math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, norm
import together

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'analysis/rebuilt'
OUT.mkdir(parents=True, exist_ok=True)
ORWIG_DATA = REPO / 'data/external/orwig_2024/cw_data.csv'
FT = os.environ.get('AGC_JUDGE_MODEL', 'agcbench-2026/AGC-Judge')

PROMPT_C = """A panel of human raters evaluated short creative stories written by human study participants on a 1-5 scale (1 = not at all creative, 5 = extremely creative). A creative response is novel, high-quality, and surprising.

Task: Write a short creative story using the given three cue words.

Cue words: {p}

Human participant's story: {r}

Predict what the human raters' average rating would be for this human-written story. Return only a single number between 1 and 5 (one decimal allowed), with no other text."""


def parse_score(s: str) -> float | None:
    if not s:
        return None
    # accept floats too: "3.5", "3", "Score: 3.5"
    m = re.findall(r'-?\d+(?:\.\d+)?', s.strip())
    if not m:
        return None
    for n in reversed(m):
        v = float(n)
        if 1.0 <= v <= 5.0:
            return v
    return float(m[-1])


def steiger_z(r12: float, r13: float, r23: float, n: int) -> tuple[float, float]:
    """Williams' modification of Hotelling's t-test for the difference between
    two dependent correlations sharing variable 1 (here: agc-judge).

    Returns (t, two-sided p-value).
    """
    R = (1 - r12**2 - r13**2 - r23**2) + 2 * r12 * r13 * r23
    avg = (r12 + r13) / 2
    num = (r12 - r13) * math.sqrt((n - 1) * (1 + r23))
    den = math.sqrt(2 * (n - 1) / (n - 3) * R + (avg**2) * (1 - r23)**3)
    t = num / den if den > 0 else float('nan')
    # Williams' t is approximately Student's t with n-3 df; large n -> normal
    p = 2 * (1 - norm.cdf(abs(t)))
    return t, p


def main():
    if not ORWIG_DATA.exists():
        raise FileNotFoundError(
            f'Orwig 2024 corpus not found at {ORWIG_DATA}. The Orwig data is not '
            f'redistributed under this bundle. See module docstring for instructions.'
        )
    df = pd.read_csv(ORWIG_DATA)
    print(f'Loaded Orwig: n={len(df)}, conditions={df["condition"].value_counts().to_dict()}')

    client = together.Together()

    def call(prompt_text: str, max_tok: int = 6) -> str:
        try:
            r = client.chat.completions.create(
                model=FT,
                messages=[{'role': 'user', 'content': prompt_text}],
                max_tokens=max_tok, temperature=0)
            return (r.choices[0].message.content or '').strip()
        except Exception as e:
            return f'ERR:{str(e)[:60]}'

    def score_one(row):
        raw = call(PROMPT_C.format(p=row['prompt'], r=row['response']))
        return parse_score(raw), raw

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(score_one, [r for _, r in df.iterrows()]))
    df['pred_C'] = [x[0] for x in results]
    df['raw_C'] = [x[1] for x in results]
    print(f'Scoring: {time.time()-t0:.0f}s, parseable: {df["pred_C"].notna().sum()}/{len(df)}')

    out = OUT / 'orwig_agc_judge_promptC.parquet'
    df.to_parquet(out, index=False)
    print(f'Wrote {out}')

    # ===== Analysis =====
    print('\n========== Item-level correlations (n=718) ==========')
    sub = df.dropna(subset=['pred_C', 'creativity', 'GPT_score'])
    n = len(sub)
    r12 = sub[['pred_C', 'creativity']].corr(method='spearman').iloc[0, 1]
    r13 = sub[['pred_C', 'GPT_score']].corr(method='spearman').iloc[0, 1]
    r23 = sub[['creativity', 'GPT_score']].corr(method='spearman').iloc[0, 1]
    p12 = sub[['pred_C', 'creativity']].corr(method='pearson').iloc[0, 1]
    p13 = sub[['pred_C', 'GPT_score']].corr(method='pearson').iloc[0, 1]
    p23 = sub[['creativity', 'GPT_score']].corr(method='pearson').iloc[0, 1]
    print(f'  agc-judge x human-creativity:  rho={r12:+.3f}  r={p12:+.3f}')
    print(f'  agc-judge x GPT_score (LLM):   rho={r13:+.3f}  r={p13:+.3f}')
    print(f'  human-creativity x GPT_score:  rho={r23:+.3f}  r={p23:+.3f}  (shared var)')

    # Steiger Z (Williams' t) on Pearson r (more standard for this test)
    t, p = steiger_z(p12, p13, p23, n)
    print(f"\n  Steiger / Williams' t (Pearson): t={t:+.3f}, p={p:.4f}, n={n}")
    print(f'  Interpretation: agc-judge {"more aligned with LLM-judge gold" if p13 > p12 else "more aligned with human gold"}'
          f' ({"sig" if p < 0.05 else "n.s."} at alpha=0.05)')

    # Also on Spearman (z-transform-based)
    t_s, p_s = steiger_z(r12, r13, r23, n)
    print(f"  Steiger / Williams' t (Spearman): t={t_s:+.3f}, p={p_s:.4f}")

    # All other dimensions
    print('\n========== agc-judge vs other measures ==========')
    for col in ['dsi', 'word_count', 'perceptual_details']:
        s = sub.dropna(subset=[col])
        rho, _ = spearmanr(s['pred_C'], s[col])
        rp, _ = pearsonr(s['pred_C'], s[col])
        print(f'  agc-judge x {col:<22s}: rho={rho:+.3f} r={rp:+.3f}')

    # Per-condition means
    print('\n========== Per-condition means (does agc-judge show source bias?) ==========')
    print(f'{"condition":<10} {"n":>4} {"agc-judge":>10} {"human-cret":>11} {"GPT_score":>10}')
    for cond in ['human', 'GPT-3', 'GPT-4']:
        c = sub[sub['condition'] == cond]
        print(f'  {cond:<10} {len(c):>4} {c["pred_C"].mean():>10.2f} {c["creativity"].mean():>11.2f} {c["GPT_score"].mean():>10.2f}')

    # Within-condition correlations (control for source)
    print('\n========== Within-condition (controlling for source) ==========')
    for cond in ['human', 'GPT-3', 'GPT-4']:
        c = sub[sub['condition'] == cond]
        if len(c) >= 30:
            r_h, _ = spearmanr(c['pred_C'], c['creativity'])
            r_g, _ = spearmanr(c['pred_C'], c['GPT_score'])
            print(f'  {cond:<10} (n={len(c):>3}): agc x human-cret rho={r_h:+.3f}  |  agc x GPT_score rho={r_g:+.3f}')


if __name__ == '__main__':
    main()
