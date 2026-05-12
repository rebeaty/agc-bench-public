# Cascade-impact analysis — three DQ scenarios

Recomputed under three handlings of the 32 cells flagged by both heuristic and audit signals.


## A. as-is

- **Models in composite**: 83
- **Models in c-factor (complete domains)**: 83
- **First eigenvalue**: 4.921
- **Cronbach α**: 0.956
- **% variance (eig1 / 6)**: 82.0
- **Loadings**: Brainstorming: -0.67, Figurative Language: -0.57, Humor: -0.53, Problem Solving: -0.53, STEM: -0.47, Story / Narrative: -0.55
- **Intelligence ρ**: AA Intelligence: ρ=+0.778 (r=+0.680, n=74), MMLU-Pro: ρ=+0.627 (r=+0.533, n=57), GPQA-Diamond: ρ=+0.769 (r=+0.660, n=75), HLE: ρ=+0.638 (r=+0.588, n=75)

**Top 10 leaderboard:**
   1. anthropic/claude-opus-4.7                +0.804
   2. openai/gpt-5.4                           +0.737
   3. openai/gpt-5.5                           +0.736
   4. z-ai/glm-5.1                             +0.711
   5. z-ai/glm-5-turbo                         +0.668
   6. anthropic/claude-opus-4.6-fast           +0.663
   7. z-ai/glm-5                               +0.651
   8. openai/gpt-5.1                           +0.643
   9. google/gemma-4-31b-it                    +0.642
  10. moonshotai/kimi-k2.5                     +0.635

## B. mask 32 BOTH cells

- **Models in composite**: 83
- **Models in c-factor (complete domains)**: 83
- **First eigenvalue**: 4.911
- **Cronbach α**: 0.955
- **% variance (eig1 / 6)**: 81.9
- **Loadings**: Brainstorming: -0.69, Figurative Language: -0.55, Humor: -0.51, Problem Solving: -0.52, STEM: -0.45, Story / Narrative: -0.55
- **Intelligence ρ**: AA Intelligence: ρ=+0.778 (r=+0.681, n=74), MMLU-Pro: ρ=+0.629 (r=+0.536, n=57), GPQA-Diamond: ρ=+0.770 (r=+0.663, n=75), HLE: ρ=+0.638 (r=+0.589, n=75)

**Top 10 leaderboard:**
   1. anthropic/claude-opus-4.7                +0.804
   2. openai/gpt-5.4                           +0.737
   3. openai/gpt-5.5                           +0.736
   4. z-ai/glm-5.1                             +0.711
   5. z-ai/glm-5-turbo                         +0.668
   6. anthropic/claude-opus-4.6-fast           +0.663
   7. z-ai/glm-5                               +0.651
   8. openai/gpt-5.1                           +0.643
   9. google/gemma-4-31b-it                    +0.642
  10. moonshotai/kimi-k2.5                     +0.635

## C. drop 3 flagged models

- **Models in composite**: 80
- **Models in c-factor (complete domains)**: 80
- **First eigenvalue**: 4.908
- **Cronbach α**: 0.955
- **% variance (eig1 / 6)**: 81.8
- **Loadings**: Brainstorming: -0.60, Figurative Language: -0.57, Humor: -0.49, Problem Solving: -0.50, STEM: -0.45, Story / Narrative: -0.53
- **Intelligence ρ**: AA Intelligence: ρ=+0.778 (r=+0.680, n=74), MMLU-Pro: ρ=+0.627 (r=+0.533, n=57), GPQA-Diamond: ρ=+0.769 (r=+0.660, n=75), HLE: ρ=+0.638 (r=+0.588, n=75)

**Top 10 leaderboard:**
   1. anthropic/claude-opus-4.7                +0.804
   2. openai/gpt-5.4                           +0.737
   3. openai/gpt-5.5                           +0.736
   4. z-ai/glm-5.1                             +0.711
   5. z-ai/glm-5-turbo                         +0.668
   6. anthropic/claude-opus-4.6-fast           +0.663
   7. z-ai/glm-5                               +0.651
   8. openai/gpt-5.1                           +0.643
   9. google/gemma-4-31b-it                    +0.642
  10. moonshotai/kimi-k2.5                     +0.635

## Primary-Number Deltas

| metric | A as-is | B mask-32 | Δ | C drop-3 | Δ |
|---|---|---|---|---|---|
| eigenvalue_1 | 4.921 | 4.911 | -0.010 | 4.908 | -0.013 |
| alpha | 0.956 | 0.955 | -0.001 | 0.955 | -0.001 |
| var % | 82.0 | 81.9 | -0.2 | 81.8 | -0.2 |
| AA Intelligence ρ | +0.778 | +0.778 | -0.000 | +0.778 | +0.000 |
| MMLU-Pro ρ | +0.627 | +0.629 | +0.002 | +0.627 | +0.000 |
| GPQA-Diamond ρ | +0.769 | +0.770 | +0.000 | +0.769 | +0.000 |
| HLE ρ | +0.638 | +0.638 | -0.000 | +0.638 | +0.000 |

## Rank shifts under masking (B vs A) for any affected model

| model | rank A | mean A | rank B | mean B | Δ rank |
|---|---|---|---|---|---|
| deepcogito/cogito-v2.1-671b | 44 | +0.073 | 40 | +0.145 | -4 |
| tencent/hunyuan-a13b-instruct | 68 | -0.446 | 66 | -0.386 | -2 |
| meta-llama/llama-3.2-11b-vision-instruct | 71 | -0.578 | 72 | -0.574 | +1 |
| meta-llama/llama-3.1-8b-instruct | 73 | -0.671 | 73 | -0.663 | +0 |
| google/gemma-2-27b-it | 75 | -0.891 | 69 | -0.483 | -6 |
| qwen/qwen3-14b | 78 | -1.012 | 77 | -0.997 | -1 |
| z-ai/glm-4.5v | 79 | -1.074 | 79 | -1.015 | +0 |
| meta-llama/llama-3.2-3b-instruct | 80 | -1.076 | 80 | -1.049 | +0 |
| morph/morph-v3-fast | 82 | -1.497 | 82 | -1.418 | +0 |
