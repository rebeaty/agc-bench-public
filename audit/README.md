# `audit/` — release audit materials

This directory gathers release audit materials and supporting checks behind
the released artifacts.

| Path | Purpose |
|---|---|
| `fidelity/` | Per-benchmark paper-vs-implementation audit reports and summary indexes. Start with `fidelity/INDEX.md`. |
| `dq_sweep/` | Release data-quality sweep, including the 32-cell `dq_masked` decision path and cascade-impact checks. Start with `dq_sweep/REPORT.md`. |
| `dq_audit/` | LLM on-task audit outputs used by the data-quality sweep. |
| `judge_prompts/` | Verbatim LLM-judge prompts for scoring, audits, domain classification, AGC-Human fairness checks, and intervention analyses. |

The audit files are supporting evidence for the released artifacts in
`release_data/` and `analysis/`; the quickest way to reproduce the main
results remains `bash reproduce_paper_results.sh` from the repository root.
