# `generations/` - model-output corpus

This directory is hosted on the Hugging Face dataset release. It contains the
derived item-level model outputs behind the 83-model release set where those
records are available.

| Path | Purpose |
|---|---|
| `<provider>/<model>/<benchmark>.parquet` | Per-(model, benchmark) prompts, completions, item identifiers, and available per-item score metadata. |
| `prompts/<benchmark>.parquet` | De-duplicated prompt bank for each benchmark. |
| `_consolidation_audit.csv` | Build log for the generation corpus: source run paths, retained cells, row counts, and missing-file notes. |

The canonical aggregate scores remain in `release_data/`. The generation corpus
is provided for inspection and secondary analysis, not as the default landing
table for the dataset viewer.
