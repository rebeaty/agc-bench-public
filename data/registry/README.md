# `data/registry/` - benchmark registry

The registry files are the compact, machine-readable map from source benchmark
papers to the AGC-Bench HELM implementation.

| File | Purpose |
|---|---|
| `registry_master.yaml` | Benchmark identity, source paper, source repository, license, modality, status, and release notes. |
| `registry_metrics.yaml` | Canonical source-paper metrics and the corresponding AGC-Bench metric implementation. |
| `registry_inference.yaml` | Inference settings, including output counts and generation parameters used by the run specs. |
| `registry_rubrics.yaml` | Rubrics used by benchmark-specific LLM-as-judge annotators. |

These files document implementation choices. They are not aggregate results;
those live in `release_data/`.
