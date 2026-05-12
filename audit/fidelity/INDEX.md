# AGC-Bench Fidelity Audit (v1 Release)

Per-benchmark fidelity audit covering all 78 onboarded benchmarks in the v1
release. Each `<bench>/REPORT.md` documents the four-axis paper-vs-implementation
audit (instance source, prompt fidelity, metric/scoring fidelity, generation
config) with severity-tagged deviations.

- [AUDIT_METHODOLOGY.md](AUDIT_METHODOLOGY.md) — what each report covers and how it was produced
- [SUMMARY.md](SUMMARY.md) — **per-benchmark table** with tier (1 / 2 / 3 / ?), confidence, and one-line recommendation. Direct links to each `<bench>/REPORT.md`.
- `<bench>/REPORT.md` — per-benchmark deviation list

**Tier distribution at v1:** 19 Tier-1 / 50 Tier-2 / 9 Tier-3 / 0 Tier-?. See [SUMMARY.md](SUMMARY.md) for the per-benchmark table and [AUDIT_METHODOLOGY.md](AUDIT_METHODOLOGY.md) for the tier-rule definitions (Tier 3 = retained as proxy / adapted with deviation documented; sensitivity check in [../../release_data/SCORING_NOTES.md](../../release_data/SCORING_NOTES.md)).

**Note on registry vs. run_spec.** A handful of per-benchmark reports
(notably `cpers`, `creatset`, `slang_generation`) discuss
`registry_metrics.yaml` as the canonical metric surface. The registry
documents the source paper's available metrics; what the released runs
actually computed is determined by the corresponding
`run_specs/<bench>_run_specs.py` (which wires the specific MetricSpec
and AnnotatorSpec instances into HELM). Where the two diverge, the run-spec
is authoritative for the released `dataset_z` and the registry entry is
descriptive. See [../../release_data/SCORING_NOTES.md](../../release_data/SCORING_NOTES.md)
for the release-level conventions.

Substitutions users are most likely to check — judge-model swaps to
`gemini-3-flash-preview` / `gpt-4` family, embedding-backend routing through
`metrics/embedder_factory.py` (default Gemini), prompt paraphrasing where the
source paper format isn't compatible with HELM — are flagged in the relevant
per-benchmark report under "Deviations found". Release-level patterns are
covered in [../../release_data/SCORING_NOTES.md](../../release_data/SCORING_NOTES.md).

## Companion audits

- `audit/dq_sweep/` — release data-quality sweep that cross-referenced
  heuristic flags (empty / refusal / repetitive responses) with the on-task
  LLM-judge audit. Identified 32 cells flagged by both signals; those cells
  are masked at the score level (`dq_masked = True` in the long table).
  Cascade-impact analysis confirmed primary numbers shift by at most 0.03.
- `audit/judge_prompts/` — verbatim text of every LLM-judge prompt used in
  AGC-Bench: per-benchmark scoring rubrics, the DQ on-task audit prompt, the
  3-LLM domain-classifier panel, the AGC-Human fairness-aware judge, the
  CAP validity gate, the be-creative + reasoning intervention prompts, and
  the MuCE judgment prompt.
