# Release-Level Scoring Notes

Two cross-benchmark patterns are worth flagging up-front, with per-benchmark
detail in `audit/fidelity/<bench>/REPORT.md`.

## 1. LLM-judge layer (24 of 67 text-only benchmarks)

The 24 benchmarks whose canonical metric is LLM-judge (`jrt_corrected = True`
in `dataset_metadata.csv`) are scored under a fixed three-vendor
**Judge Response Theory** panel: `google/gemini-3-flash-preview`,
`x-ai/grok-4.1-fast`, `openai/gpt-4.1-mini`. Per-judge severity is calibrated
via a Bayesian Graded Response Model (`scripts/fit_jrt_grm.py`); the released
`dataset_z` for these cells comes from GRM theta. Released long-table rows
for these cells carry `score_source = 'jrt'`. The long-table
`n_score_sources` column is aggregate-level bookkeeping: it records how many
judge identities contributed to the model-dataset aggregate, not the item-level
planned-missing rater count.

The `registry_metrics.yaml` `judge_model_name` field records each benchmark's
source-paper canonical judge for fidelity. The released `dataset_z` for the
24 LLM-judge cells reflects the JRT three-vendor calibration above, not any
one source-paper judge endpoint.

The remaining 43 benchmarks use formula-based or model-based metrics (no LLM
judge involved); their long-table rows carry `score_source = 'raw'`.

## 2. Embedding-backend routing (`metrics/embedder_factory.py`)

Nine metric modules call out to `metrics/embedder_factory.py`, which uses
a single Gemini embedding backend (`gemini-embedding-001`) for release-set
comparability:

```
metrics/sdat_metric.py
metrics/conceptual_design_metric.py
metrics/creative_process_metric.py
metrics/creativity_score_metric.py
metrics/mops_diversity_metric.py
metrics/sentence_bert_metric.py
metrics/semantic_diversity_metric.py
metrics/slang_generation_metric.py
metrics/matdesign_metric.py
```

The `registry_metrics.yaml` `metric_model` field records each metric's
source-paper-canonical embedding model (e.g. SDAT lists
`ibm-granite/granite-embedding-278m-multilingual`); the runtime embedder
may differ from that field. Substitutions that affect a released
`dataset_z` are flagged at MEDIUM or HIGH in the corresponding
`audit/fidelity/<bench>/REPORT.md`.

For the 24 LLM-judge benchmarks scored under JRT, the embedding backend
has no effect on the released `dataset_z` — the calibrated theta
supersedes embedding-derived scores. Embedding routing is release-visible
on `raw`-source cells where a model-based metric contributes to the
canonical mean. The most prominent of these is `sdat`, where every
canonical metric is embedding-driven (`audit/fidelity/sdat/REPORT.md`).

## Release Scope

The released bundle contains the 83 strict-coverage models used for the primary
AGC-Bench release analyses. Every model-keyed CSV in `release_data/` and
`analysis/` contains rows for these 83 release models (or the AA-overlap subset
where the intelligence join requires it). All released numbers are computed on
this release set.

## Authority order: run_specs are canonical, registry is descriptive

Where `data/registry/registry_metrics.yaml` and
`run_specs/<bench>_run_specs.py` declare different metric layouts, the
**run_spec is authoritative** for what was computed in the released runs.
The registry catalogues the source paper's available metric surface (BLEU,
ROUGE, embedding models, candidate LLM-judge entries) for documentation;
the run_spec wires the specific MetricSpec / AnnotatorSpec instances HELM
actually evaluated. Per-benchmark `audit/fidelity/<bench>/REPORT.md` files
document the wiring under "Implementation audited" and flag drift from
registry under "Deviations found".

## Tier-3 sensitivity

Eight of the nine Tier-3 benchmarks in `audit/fidelity/SUMMARY.md` are
included in the 67-dataset text-only primary analysis as proxy or adapted
benchmarks (`mars` is excluded for unrelated multimodal-only reasons).
The Tier-3 label flags HIGH paper-vs-implementation deviations on at
least one axis (typically metric/scoring fidelity) where the implemented
metric differs in substantive construct from the paper's primary metric;
the implemented evaluation still produces a coherent signal that
contributes to the release set's per-domain composites.

To make the inclusion explicit, the primary numbers were recomputed
under a sensitivity check that drops the eight included Tier-3 cells:

| metric | 67 datasets | drop 8 Tier-3 (59 ds) | delta |
|---|---|---|---|
| C-factor eigenvalue | 4.89 | 4.22 | -0.67 |
| Cronbach α | 0.96 | 0.91 | -0.04 |
| % variance | 81.5 % | 70.4 % | -11.1 pp |
| AA Intelligence ρ | +0.77 | +0.77 | -0.00 |
| MMLU-Pro ρ | +0.63 | +0.62 | -0.01 |
| GPQA-Diamond ρ | +0.77 | +0.76 | -0.01 |
| HLE ρ | +0.64 | +0.63 | -0.01 |

C-factor magnitude shrinks (the eight benchmarks contribute about 12 %
of the release set and load on c), but the structure stays unidimensional
with α = 0.91 and 70.4 % variance on the first PC. Intelligence ρ values
shift by at most 0.01 across all four indicators. Readers preferring a
strict exclusion can read off the "drop 8 Tier-3" column directly.

## Reproducibility scope

`reproduce_paper_results.sh` checks the main-text numerical claims
against the in-bundle 83-model release set: eigenvalue 4.89, α 0.96,
variance 81.5 %, intelligence ρ values (AA Intelligence ρ = 0.77,
GPQA-Diamond ρ = 0.76), raw-vs-JRT ρ = 0.94, AGC-Judge held-out splits
(0.83 / 0.94 / 0.94), LLM vs human α (0.64 vs 0.42, p = 0.028). The c-factor
and CAP within-instrument numbers are recomputed live from raw inputs;
the raw-vs-JRT ρ is read from the shipped `analysis/leaderboard_raw_vs_jrt.csv`
artifact and the spearman is recomputed on its 83 release-model rows.

Two of these reproduce to within rounding rather than exact match against
the paper text:

- C-factor eigenvalue / variance: 4.89 / 81.5 % here matches the paper
  text. The release set applies the data-quality mask documented in
  `audit/dq_sweep/`; the paper reports the post-mask values throughout.
- Raw-vs-JRT ρ: 0.94 here on the shipped 83-model pre-/post-JRT comparison
  artifact (`analysis/leaderboard_raw_vs_jrt.csv`), versus the paper text's
  rounded 0.95 from the submission analysis snapshot. The substantive
  conclusion (raw and JRT-corrected leaderboards converge at high rank
  correlation) holds either way.

Appendix-specific analyses — App D domain panel κ, App G DQ audit,
App H PM1-50 + JRT GRM + AGC-Judge held-out + Orwig validation, App I
MuCE / LSA / interventions, App J robustness sweeps, App K AA-battery
table + parameter-count scaling, App L fairness audit + style flip,
App M qualitative samples — are checked end-to-end by
`reproduce_appendix.sh`, which recomputes a panel of independently checkable
numerical claims from the in-bundle artifacts and prints paper-stated
vs computed for each (the script prints the current check count in its
summary).
