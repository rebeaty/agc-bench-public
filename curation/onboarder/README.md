# Benchmark Onboarding Workflow

This directory documents how accepted catalog entries were converted into
HELM-compatible AGC-Bench scenarios.

| File | Purpose |
| --- | --- |
| `SKILL.md` | Operational instructions used during scenario implementation. |
| `helm-template.md` | HELM scenario patterns and code conventions. |
| `benchmarks.json` | Queue and status records for candidate benchmarks. |
| `examples/` | Example scenario patterns for supported modalities. |

The workflow prioritizes source-paper fidelity: prompts are taken from source
materials when available, dataset fields are checked against sample records, and
scoring notes are preserved for downstream metric configuration. Benchmarks were
reviewed before entering the release set.
