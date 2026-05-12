# Changelog

## v1.0.1 - Package Update

This release improves the repository and Hugging Face package. Scores,
validation artifacts, and benchmark implementations are unchanged.

- Added `release_data/benchmark_catalog_497.csv`, the 497-row deduplicated
  benchmark catalog used before benchmark selection and implementation.
- Added `release_data/model_dataset_scores.csv`, a compact score table for the
  Hugging Face Dataset Viewer.
- Added Hugging Face dataset-card YAML so the viewer opens on the compact score
  table and also exposes the benchmark catalog.
- Added `curation/catalog/` and `curation/onboarder/`, which document catalog
  construction and benchmark onboarding.
- Added `scripts/consolidate_generations.py` and documented the hosted
  `generations/` corpus of release-set prompts and model completions.
- Added short README maps for the main repository directories.
- Updated the aggregate and per-benchmark Croissant manifests for the files
  shipped in this package.
