# `data/` - bundled inputs and registries

This directory contains small files needed by the runnable benchmark package.
The primary released score tables are in `release_data/`, and validation
artifacts are in `analysis/`.

| Path | Purpose |
|---|---|
| `registry/registry_master.yaml` | Source benchmark metadata: names, source papers, licenses, modality, and release status. |
| `registry/registry_metrics.yaml` | Source-paper scoring metrics and AGC-Bench metric mappings. |
| `registry/registry_inference.yaml` | Per-benchmark inference settings used by the HELM run specs. |
| `registry/registry_rubrics.yaml` | Structured scoring rubrics for benchmark-specific LLM judges. |
| `muce_balanced30.parquet` | Small MuCE subset used by released convergent-validity checks. |

Third-party corpora that are not redistributed are referenced by scripts under
their expected local paths, such as `data/external/...`, when users obtain them
separately.
