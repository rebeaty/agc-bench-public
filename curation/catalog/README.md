# AGC-Bench Catalog Audit

This directory documents the benchmark-catalog stage of AGC-Bench.

The paper reports two related catalog counts:

- **497** unique benchmark records after deduplication.
- **432** creativity-relevant candidates after filtering out adjacent/general
  NLP, CV, reasoning, intelligence, and auxiliary benchmarks that were
  co-extracted from the reviewed papers.

The release file [`../../release_data/benchmark_catalog_497.csv`](../../release_data/benchmark_catalog_497.csv)
puts both counts in one table. It has one row per deduplicated benchmark
record and a `creativity_relevant` flag. The 432 creativity-relevant candidates
are the rows where `creativity_relevant = true`; the remaining 65 rows are
retained for auditability.

The 497-row catalog is intentionally narrow: benchmark name/aliases,
creativity-relevance flag, source paper, modality, runnability tier, and
source-paper scoring protocol. It does not include the preliminary
catalog-stage domain labels because the paper's final six-domain taxonomy
applies to the 78 onboarded release benchmarks; those final domains remain in
[`../../croissant/agc-bench-catalog.csv`](../../croissant/agc-bench-catalog.csv)
and `release_data/dataset_metadata.csv`.

This v1.0.1 metadata update does not change model outputs, leaderboard scores,
validation analyses, released benchmark implementations, or paper headline
numbers.

## Reproducible Catalog Chain

The catalog-stage provenance files are in [`source/`](source/):

| Stage | File | Count |
|---|---|---:|
| Gemini full-text extraction | `source/extracted_benchmarks_merged.jsonl` | 283 papers; 546 benchmark mentions; 1,160 task records |
| Hybrid deduplication | `source/unique_benchmarks.csv` | 497 unique benchmark records |
| Creativity relevance filter | `source/benchmark_catalog_432.csv` | 432 creativity-relevant candidates |
| Release joined catalog | `../../release_data/benchmark_catalog_497.csv` | 497 rows; 432 relevant; 65 filtered |

`source/deduplication_results.json` records the deduplication groups behind the
497-row catalog.

The release package starts at the benchmark-catalog stage, where the checked-in
files reproduce the catalog counts exactly. Earlier paper-harvest and screening
counts are documented in the paper, but those transient logs are not included as
canonical release files.

## Regeneration

Run from the repository root:

```bash
python3 scripts/build_benchmark_catalog_497.py
```

The script expects the checked-in source files at `curation/catalog/source/`
by default. Use `--archive` to rebuild from a compatible source archive with
the original pipeline directory layout.
