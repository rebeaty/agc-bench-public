# Letter-String Analogies (LSA) — methods note

A quick reference for the LSA experiment we ran on the AGC release set, paired with `lsa_per_model.png`.

## What it is

The Lewis–Mitchell counterfactual letter-string analogy task — a Hofstadter-style fluid-reasoning probe. Each item shows a model an example transformation (e.g. `abc → abd`) and asks it to apply the abstract rule to a new input (`ijk → ?`). The **counterfactual** variant uses permuted alphabets, which strips away surface-pattern shortcuts and forces the model to abstract a real rule. That makes it a clean fluid-intelligence (gf) test in the LLM literature.

## How we ran it

- **500 stratified items**, balanced across 19 transformation types and the seen / unseen testlet split.
- Implementation matches the published LSA reference code byte-for-byte at the prompt and scoring level — single-attempt direct evaluation, temperature = 0, **reasoning OFF** for fairness across model families (Anthropic Claude variants where reasoning toggles were unreliable were excluded).
- 83 strict-coverage models from the AGC release set (the same 83 used for the AGC text-only primary analysis).
- Strict accuracy: `accuracy_strict = exact match between model output and reference D`. We also compute `mean_lev_to_D` (Levenshtein distance to the reference), while the primary reported value is strict accuracy.

## What we use it for in the AGC paper

- The pure-gf indicator alongside the broader Artificial Analysis intelligence battery (which is mixed Gq / Gc, not pure gf).
- Primary correlation: **AGC composite × LSA = +0.51 Pearson** across the release set.
- Supports the paper's claim that creativity in LLMs (the c-factor in AGC) is correlated with but distinguishable from fluid reasoning (LSA).

## Release-Set Summary

- **n = 83 models**
- Release-set mean accuracy = **0.27**
- Range = **0.00 – 0.77**
- Top 3: `qwen3.5-flash-02-23` (0.77), `gpt-5.5` (0.65), `claude-opus-4.5` (0.54)
- Bottom 3: `morph-v3-fast` (0.00), `nemotron-nano-12b-v2-vl` (0.01), `nemotron-3-nano-30b-a3b` (0.04)

## File

`lsa_per_model.png` (and `.pdf`): horizontal bar chart, models sorted ascending by accuracy, family-colored, release-set mean dotted line.
